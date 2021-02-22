'''
Evaluate full-pipeline predictions. Example usage:

python verisci/evaluate/pipeline.py \
    --gold data/claims_dev.jsonl \
    --corpus data/corpus.jsonl \
    --prediction prediction/merged_predictions.jsonl \
    --output predictions/metrics.json
'''

import argparse
import json

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

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    parser.add_argument('--gold', type=str, default='../data/claims_dev.jsonl',
                        help='The gold labels.')
    parser.add_argument('--corpus', type=str, default='../data/corpus.jsonl')
    parser.add_argument('--prediction', type=str, default='prediction/merged_predictions.jsonl',
                        help='The predictions.')
    parser.add_argument('--output', type=str, default=None,
                        help='If provided, save metrics to this file.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    data = GoldDataset(args.corpus, args.gold)
    predictions = PredictedDataset(data, args.prediction)

    res = compute_metrics(predictions)
    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(res.to_dict(), f, indent=2)
    else:
        print(res)


if __name__ == "__main__":
    main()
