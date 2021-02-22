import torch
import datetime
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoModel
from embedding.sciver_model import CNN, SiameseModel
from embedding.sciver_model import BertForSequenceClassification


def get_model(args):
    print("{}, Building embedding".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    if args.embedding == 'sciver_model':
        model = CNN(args)
    elif args.embedding == 'roberta':
        config = AutoConfig.from_pretrained('roberta-base', num_labels=args.num_label)
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base', config=config)
    elif args.embedding == 'bert_cnn':
        config = AutoConfig.from_pretrained('bert-base-uncased', num_labels=args.num_label, output_hidden_states=False)
        # print(config)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    elif args.embedding == 'SciBert':
        config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=args.num_label)
        model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=config)
    else:
        model = SiameseModel(args)

    # if args.snapshot != '':
    #     # load pretrained models
    #     print("{}, Loading pretrained embedding from {}".format(
    #         datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
    #         args.snapshot + '.ebd'
    #     ))
    #     model.load_state_dict(torch.load(args.snapshot + '.ebd'))

    # # load pre-trained cnn model
    # if args.embedding == 'hscnn' and args.snapshot == '' and args.cnn_model_dict != '':
    #     siamese_model_dict = model.state_dict()
    #     cnn_model_dict = torch.load(args.cnn_model_dict + '.ebd')
    #     state_dict = {k: v for k, v in cnn_model_dict.items() if k in siamese_model_dict.keys()}
    #     siamese_model_dict.update(state_dict)
    #     model.load_state_dict(siamese_model_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.cuda(device)
