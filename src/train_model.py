import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import time
from pathlib import Path

from src.dataset.encode import encode_sen_pair, encode_sentence
from src.evaluation.evaluation_model import evaluate_label, evaluate_rationale


def train_rationale_selection(model, train_set, dev_set, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(os.path.curdir, "tmp-runs")
    # model: RoBerta
    # create tokenizer and model.
    # tokenizer = AutoTokenizer.from_pretrained('roberta-base')  # used to produce sentence vectors.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    # specify different learning rates for different parts of the network.
    optimizer = torch.optim.Adam([
        # If you are using non-roberta based models, change this to point to the right base
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    #  Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    #  increases linearly between 0 and the initial lr set in the optimizer.
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)  # Learning rate warm-up
    best_f1 = 0
    for e in range(args.epochs):
        model.train()
        t = tqdm(DataLoader(train_set, batch_size=args.batch_size_gpu, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode_sen_pair(tokenizer, batch['claim'], batch['sentence'])
            # label: true or false
            if args.embedding == 'bert_cnn':
                loss, logits = model(input_ids=encoded_dict['input_ids'],
                                     attention_mask=encoded_dict['attention_mask'],
                                     token_type_ids=encoded_dict['token_type_ids'],
                                     labels=batch['evidence'].long().to(device))
            elif args.embedding == 'hscnn':
                claim_encoded = encode_sentence(tokenizer, batch['claim'])
                sentence_encoded = encode_sentence(tokenizer, batch['sentence'])
                if len(sentence_encoded['input_ids'][0]) <= 5:  # [".", ".", ".", ".", ".", ".", ".", "."]
                    continue
                loss, logits = model(claim_encoded['input_ids'],
                                     sentence_encoded['input_ids'], label=batch['evidence'].long().to(device))
            else:
                # loss, logits = model(**encoded_dict, labels=batch['evidence'].long().to(device))
                output = model(input_ids=encoded_dict['input_ids'],
                               attention_mask=encoded_dict['attention_mask'],
                               labels=batch['evidence'].long().to(device))
                loss = output[0]
                # logits = output[1]
            # print(loss)
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()
        train_score = evaluate_rationale(model, train_set, args, tokenizer)
        print(f'Epoch {e}, train f1: %.4f, precision: %.4f, recall: %.4f' % train_score)
        dev_score = evaluate_rationale(model, dev_set, args, tokenizer)
        print(f'Epoch {e}, dev f1: %.4f, precision: %.4f, recall: %.4f' % dev_score)
        # Save model
        # print(dev_score)
        if train_score[0] > best_f1:
            best_f1 = dev_score[0]
            best_tokenizer = tokenizer
            best_model = model
        save_path = os.path.join(out_dir, str(int(time.time() * 1e7)) + f'-rationale-f1-{int(dev_score[0] * 1e4)}')
        os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        # model.save_pretrained(save_path)
        torch.save(model, save_path+'/pytorch_model.bin')
    save_path = os.path.join(args.save, f'rationale_best_model_SciBert')
    if not Path(save_path).exists():
        os.makedirs(save_path)
    if args.embedding == 'SciBert':
        best_tokenizer.save_pretrained(save_path)
        best_model.save_pretrained(save_path)
    else:
        best_tokenizer.save_pretrained(save_path)
        # best_model.save_pretrained(save_path)
        torch.save(best_model, save_path+'/pytorch_model.bin')
    return save_path


def train_label_prediction(model, train_set, dev_set, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(os.path.curdir, "tmp-runs")
    args.batch_size_gpu = 4
    # tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # config = AutoConfig.from_pretrained(args.model, num_labels=3)
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).to(device)
    optimizer = torch.optim.Adam([
        # If you are using non-roberta based models, change this to point to the right base
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)

    best_f1 = 0
    for e in range(args.epochs):
        model.train()
        t = tqdm(DataLoader(train_set, batch_size=args.batch_size_gpu, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode_sen_pair(tokenizer, batch['claim'], batch['rationale'])
            if args.embedding == 'bert_cnn':
                loss, logits = model(input_ids=encoded_dict['input_ids'],
                                     attention_mask=encoded_dict['attention_mask'],
                                     token_type_ids=encoded_dict['token_type_ids'],
                                     labels=batch['label'].long().to(device))
            else:
                # loss, logits = model(**encoded_dict, labels=batch['label'].long().to(device))
                output = model(input_ids=encoded_dict['input_ids'],
                               attention_mask=encoded_dict['attention_mask'],
                               labels=batch['label'].long().to(device))
                loss = output[0]
                # logits = output[1]
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()
        # Eval
        train_score = evaluate_label(model, train_set, args, tokenizer)
        print(f'Epoch {e} train score:')
        print(train_score)
        dev_score = evaluate_label(model, dev_set, args, tokenizer)
        print(f'Epoch {e} dev score:')
        print(dev_score)
        # Save
        if dev_score["macro_f1"] > best_f1:
            best_f1 = dev_score["macro_f1"]
            best_tokenizer = tokenizer
            best_model = model
        save_path = os.path.join(out_dir, str(int(time.time() * 1e7))
                                 + f'-label-f1-{int(dev_score["macro_f1"] * 1e4)}')
        os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        # model.save_pretrained(save_path)
        torch.save(model, save_path+'/pytorch_model.bin')
    save_path = os.path.join(args.save, f'label_best_model_SciBert')
    if not Path(save_path).exists():
        os.makedirs(save_path)
    if args.embedding == 'SciBert':
        best_tokenizer.save_pretrained(save_path)
        best_model.save_pretrained(save_path)
    else:
        best_tokenizer.save_pretrained(save_path)
        # best_model.save_pretrained(save_path)
        torch.save(best_model, save_path+'/pytorch_model.bin')
    return save_path
