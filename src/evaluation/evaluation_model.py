from collections import Counter
import json
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch
from torch.utils.data import DataLoader

import src.dataset.loader as loader
from src.dataset.save_file import merge
from src.evaluation.metrics import compute_f1, compute_metrics
from src.evaluation.data import GoldDataset, PredictedDataset
from src.dataset.encode import encode_sen_pair, encode_sentence


def is_correct(pred_sentence, pred_sentences, gold_sets):
    """
    A predicted sentence is correctly identified if it is part of a gold
    rationale, and all other sentences in the gold rationale are also
    predicted rationale sentences.
    """
    for gold_set in gold_sets:
        gold_sents = gold_set["sentences"]
        if pred_sentence in gold_sents:
            if all([x in pred_sentences for x in gold_sents]):
                return True
            else:
                return False

    return False


def evaluate_rationale_selection(args, rationale_results):
    '''
    # ================================================================================================================ #
    # evaluate rationale selection results.
    # ================================================================================================================ #
    '''
    evaluation_set = args.claim_dev_path
    dataset = loader.loader_json(evaluation_set)
    counts = Counter()
    for data, retrieval in zip(dataset, rationale_results):
        assert data['id'] == retrieval['claim_id']

        # Count all the gold evidence sentences.
        for doc_key, gold_rationales in data["evidence"].items():
            for entry in gold_rationales:
                counts["relevant"] += len(entry["sentences"])

        for doc_id, pred_sentences in retrieval['evidence'].items():
            true_evidence_sets = data['evidence'].get(doc_id) or []

            for pred_sentence in pred_sentences:
                counts["retrieved"] += 1
                if is_correct(pred_sentence, pred_sentences, true_evidence_sets):
                    counts["correct"] += 1

    rationale_metrics = compute_f1(counts)
    print(f'F1:                {round(rationale_metrics["f1"], 4)}')
    print(f'Precision:         {round(rationale_metrics["precision"], 4)}')
    print(f'Recall:            {round(rationale_metrics["recall"], 4)}')
    print()


def evaluate_label_predictions(args, label_results):
    '''
    # ================================================================================================================ #
    # evaluate label predictions results.
    # ================================================================================================================ #
    '''
    evaluation_set = args.claim_dev_path
    # evaluation
    corpus = loader.get_corpus(args.corpus_path)
    dataset = loader.loader_json(evaluation_set)
    # label_prediction = loader.loader_json(args.output_label)
    pred_labels = []
    true_labels = []

    LABELS = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

    for data, prediction in zip(dataset, label_results):
        assert data['id'] == prediction['claim_id']

        if args.filter:
            prediction['labels'] = {doc_id: pred for doc_id, pred in prediction['labels'].items()
                                    if corpus[int(doc_id)]['structured'] is (args.filter == 'structured')}
        if not prediction['labels']:
            continue

        # claim_id = data['id']
        for doc_id, pred in prediction['labels'].items():
            pred_label = pred['label']
            true_label = {es['label'] for es in data['evidence'].get(doc_id) or []}
            assert len(true_label) <= 1, 'Currently support only one label per doc'
            true_label = next(iter(true_label)) if true_label else 'NOT_ENOUGH_INFO'
            pred_labels.append(LABELS[pred_label])
            true_labels.append(LABELS[true_label])
    # sentence_labels = [0, 1, 2] if include_nei else [0, 2]
    print(
        f'Accuracy           '
        f'{round(sum([pred_labels[i] == true_labels[i] for i in range(len(pred_labels))]) / len(pred_labels), 4)}')
    print(f'Macro F1:          {f1_score(true_labels, pred_labels, average="macro").round(4)}')
    print(f'Macro F1 w/o NEI:  {f1_score(true_labels, pred_labels, average="macro", labels=[0, 2]).round(4)}')
    print()
    # if include_nei:
    #     print('                   [C      N      S     ]')  # C: CONTRADICT; N: NOT_ENOUGH_INFO; S: SUPPORT
    # else:
    print('                   [C      S     ]')
    print(f'F1:                {f1_score(true_labels, pred_labels, average=None, labels=[0, 2]).round(4)}')
    print(f'Precision:         {precision_score(true_labels, pred_labels, average=None, labels=[0, 2]).round(4)}')
    print(f'Recall:            {recall_score(true_labels, pred_labels, average=None, labels=[0, 2]).round(4)}')
    print()
    print('Confusion Matrix:')
    print(confusion_matrix(true_labels, pred_labels))
    print()


def merge_rationale_label(rationale_results, label_results, args, state='valid', gold=''):
    '''
    # ================================================================================================================ #
    # merge rationale and label predictions.
    # evaluate final predictions.
    # ================================================================================================================ #
    '''
    print('evaluate final predictions result...')
    merge(rationale_results, label_results, args.merge_results)

    if state == 'valid':
        import pandas as pd
        import numpy as np
        np.set_printoptions(threshold=np.inf)

        pd.set_option('display.width', 300)  # 设置字符显示宽度
        pd.set_option('display.max_rows', None)  # 设置显示最大行
        pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列

        data = GoldDataset(args.corpus_path, gold)
        predictions = PredictedDataset(data, args.merge_results)
        res = compute_metrics(predictions)
        if args.output is not None:
            with open(args.output, "w") as f:
                json.dump(res.to_dict(), f, indent=2)
        print(res)
    else:
        print('')


def evaluate_rationale(model, dataset, args, tokenizer):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_dict = encode_sen_pair(tokenizer, batch['claim'], batch['sentence'])
            if args.embedding == 'bert_cnn':
                logits = model(input_ids=encoded_dict['input_ids'],
                               attention_mask=encoded_dict['attention_mask'],
                               token_type_ids=encoded_dict['token_type_ids'])
            elif args.embedding == 'hscnn':
                claim_encoded = encode_sentence(tokenizer, batch['claim'])
                sentence_encoded = encode_sentence(tokenizer, batch['sentence'])
                if len(sentence_encoded['input_ids'][0]) <= 5:
                    continue
                logits = model(claim_encoded['input_ids'], sentence_encoded['input_ids'])
            else:
                logits = model(**encoded_dict)[0]
                # logits = model(input_ids=encoded_dict['input_ids'], attention_mask=encoded_dict['attention_mask'])
            targets.extend(batch['evidence'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    # print('targets:', len(targets))
    # print(90 * "=")
    # print('outputs:', len(outputs))
    f1 = f1_score(targets, outputs, zero_division=0)
    P = precision_score(targets, outputs, zero_division=0)
    R = recall_score(targets, outputs, zero_division=0)
    return f1, P, R


def evaluate_label(model, dataset, args, tokenizer):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_dict = encode_sen_pair(tokenizer, batch['claim'], batch['rationale'])
            if args.embedding == 'bert_cnn':
                logits = model(input_ids=encoded_dict['input_ids'],
                               attention_mask=encoded_dict['attention_mask'],
                               token_type_ids=encoded_dict['token_type_ids'])
            else:
                logits = model(**encoded_dict)[0]
                # logits = model(input_ids=encoded_dict['input_ids'], attention_mask=encoded_dict['attention_mask'])
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return {
        'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
        'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None))
    }
