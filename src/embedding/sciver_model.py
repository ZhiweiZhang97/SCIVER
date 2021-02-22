import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel, BertModel, BertPreTrainedModel
from typing import Iterable, Dict


# Siamese Network
class SiameseModel(nn.Module):
    def __init__(self, args):
        super(SiameseModel, self).__init__()

        self.ebd = nn.Embedding(args.vocab_size, args.embed_size)
        self.ebd_dim = self.ebd.embedding_dim

        self.C = args.num_label
        self.hidden_dim = args.hidden_dim
        self.single_model = CNN(args)
        self.sigmoid = nn.Sigmoid()
        self.linear_OneHot = nn.Linear(args.num_label, self.hidden_dim)  # number of label 32, 1024
        self.linear_out = nn.Linear(self.hidden_dim, self.C)  # 1024, 300

    def forward(self, claim, sentence, label=None):
        # out1 = self.single_model(sentence)
        # out1 = self.sigmoid(out1)
        out2 = self.single_model(sentence)
        out2 = self.sigmoid(out2)
        out1 = out2

        distance = self.l1_distance(out1, out2)
        out = self.linear_out(distance)
        if label is not None:
            shifted_prediction_scores = out.contiguous()
            labels = label.contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_prediction_scores, labels)
            return loss, out
        return out

    def l1_distance(self, output1, output2):
        """ Manhattan distance: element-wise absolute difference """
        return torch.abs(output1 - output2)

    def cos_distance(self, output1, output2):
        """ Cosine distance: measure the distance of two vecotrs accordings to the angle between them """
        return F.cosine_similarity(output1, output2, dim=0)

    def contrastive_loss(self, output1, output2, label, margin=2.0):
        """ calculate loss for siamese models without output dense layer (deprecated) """
        # Find the pairwise distance or eucledian distance of two output feature vectors
        euclidean_distance = F.pairwise_distance(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)))
        return loss_contrastive


#  CNN model.
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.embed_size = args.embed_size
        self.args = args
        self.ebd = nn.Embedding(args.vocab_size, args.embed_size)
        self.in_channel = 1
        self.out_channel = args.cnn_num_filters
        self.kernel_size = args.cnn_filter_sizes
        self.dropout_keep = args.dropout
        self.hidden_dim = args.hidden_dim
        self.class_num_label = args.class_num_label
        self.C = args.num_label

        # convert layer in 1, out 50, kernel size [3, 4, 5]
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                               kernel_size=(self.kernel_size[0], self.embed_size), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                               kernel_size=(self.kernel_size[1], self.embed_size), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                               kernel_size=(self.kernel_size[2], self.embed_size), stride=1, padding=0)
        self.dropout = nn.Dropout(self.dropout_keep)
        self.dropout_half = nn.Dropout(self.dropout_keep / 2.0)
        self.linear = nn.Linear(self.out_channel * len(self.kernel_size), self.hidden_dim)  # 50 * 3,  1024
        self.linear_label = nn.Linear(self.hidden_dim, self.C)  # 1024, 300
        self.sigmoid = nn.Sigmoid()
        self.linear_OneHot = nn.Linear(args.num_label, self.hidden_dim)  # Number of label. 32, 1024
        self.relu = nn.ReLU()

    def conv_pooling(self, x, conv):
        out = conv(x)
        activation = F.relu(out.squeeze(3))
        out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return out

    def forward(self, data, labels=None):
        # one_hot = data['label'].float()
        x = self.ebd(data)
        x = self.dropout_half(x)
        x = x.unsqueeze(1)
        out1 = self.conv_pooling(x, self.conv1)
        out2 = self.conv_pooling(x, self.conv2)
        out3 = self.conv_pooling(x, self.conv3)
        out = torch.cat((out1, out2, out3), 1)

        out = self.dropout(out)
        out = self.linear(out)

        output = self.linear_label(out)
        cnn_out = self.linear_label(out)
        # hc = self.linear_OneHot(one_hot)
        # hc = self.relu(hc / math.sqrt(self.C))
        if labels is not None:
            shifted_prediction_scores = output.contiguous()
            labels = labels.contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_prediction_scores, labels)
            return loss, output
        return out


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=3):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.num_channels = config.hidden_size // 3
        self.bert = BertModel(config)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(2, config.hidden_size))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(3, config.hidden_size))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(4, config.hidden_size))

        # self.pool1 = nn.MaxPool1d(kernel_size=config.max_position_embeddings - 2 + 1)
        # self.pool2 = nn.MaxPool1d(kernel_size=config.max_position_embeddings - 3 + 1)
        # self.pool3 = nn.MaxPool1d(kernel_size=config.max_position_embeddings - 4 + 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def conv_pooling(self, x, conv):
        out = conv(x)
        activation = F.relu(out.squeeze(3))
        out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return out

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = pooled_output[0]
        # pooled_output, _ = self.Robert(x)
        pooled_output = pooled_output.unsqueeze(1)

        h1 = self.conv_pooling(pooled_output, self.conv1)
        h2 = self.conv_pooling(pooled_output, self.conv2)
        h3 = self.conv_pooling(pooled_output, self.conv3)

        pooled_output = torch.cat([h1, h2, h3], 1).squeeze()

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            shifted_prediction_scores = logits.contiguous()
            labels = labels.contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_prediction_scores, labels)
            return loss, logits
        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class CosineSimilarityLoss(nn.Module):
    def __init__(self, model, loss_fct=nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.view(-1))