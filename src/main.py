import torch
import argparse

from src.dataset.loader import SciFactRationaleSelectionDataset, SciFactLabelPredictionDataset
from train_model import train_rationale_selection, train_label_prediction
from evaluation.evaluation_model import evaluate_rationale_selection, evaluate_label_predictions, merge_rationale_label
from get_prediction import get_rationale, get_labels
import embedding.factory as factory
from src.dataset.save_file import split_dataset, tfidf_abstract


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    # dataset parameters.
    parser.add_argument('--corpus_path', type=str, default='../data/corpus.jsonl',
                        help='The corpus of documents.')
    parser.add_argument('--claim_train_path', type=str,
                        default='../data/claims_train_retrieval.jsonl')
    parser.add_argument('--claim_dev_path', type=str,
                        default='../data/claims_dev_retrieval.jsonl')
    parser.add_argument('--claim_test_path', type=str,
                        default='../data/claims_test.jsonl')
    parser.add_argument('--gold', type=str, default='../data/claims_dev.jsonl')
    parser.add_argument('--abstract_retrieval', type=str,
                        default='prediction/abstract_retrieval.jsonl')
    parser.add_argument('--rationale_selection', type=str,
                        default='prediction/rationale_selection.jsonl')
    parser.add_argument('--save', type=str, default='model/',
                        help='Folder to save the weights')
    parser.add_argument('--output_label', type=str, default='prediction/label_predictions.jsonl')
    parser.add_argument('--merge_results', type=str, default='prediction/merged_predictions.jsonl')
    parser.add_argument('--output', type=str, default='prediction/result_evaluation.json',
                        help='The predictions.')
    parser.add_argument('--rationale_selection_tfidf', type=str, default='prediction/rationale_selection_tfidf.jsonl')

    # model parameters.
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--rationale_model', type=str, default='')
    parser.add_argument('--label_model', type=str, default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.5, required=False)
    parser.add_argument('--only_rationale', action='store_true')
    parser.add_argument('--batch_size_gpu', type=int, default=8,
                        help='The batch size to send through GPU')
    parser.add_argument('--batch-size-accumulated', type=int, default=256,
                        help='The batch size for each gradient update')
    parser.add_argument('--lr-base', type=float, default=1e-5)
    parser.add_argument('--lr-linear', type=float, default=1e-3)
    parser.add_argument('--mode', type=str, default='claim_and_rationale',
                        choices=['claim_and_rationale', 'only_claim', 'only_rationale'])
    parser.add_argument('--filter', type=str, default='structured',
                        choices=['structured', 'unstructured'])

    parser.add_argument('--embedding', type=str, default='roberta')

    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension")
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--num_label", type=int, default=2, help="numbers of the label")
    parser.add_argument("--class_num_label", type=int, default=1,
                        help="max number of the label for one class")
    parser.add_argument("--embed_size", type=int, default=300, help="embedding size")
    parser.add_argument("--cnn_num_filters", type=int, default=128,
                        help="Num of filters per filter size [default: 50]")
    parser.add_argument("--cnn_filter_sizes", type=int, nargs="+",
                        default=[3, 4, 5],
                        help="Filter sizes [default: 3]")
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument("--dropout", type=float, default=0.5, help="drop rate")
    parser.add_argument('--k', type=int, default=10, help="tfidf")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    # loader dataset
    split = True
    if split:
        split_dataset('../data/claims_train_retrieval.jsonl')
        claim_train_path = '../data/train_data.jsonl'
        claim_dev_path = '../data/dev_data.jsonl'
    else:
        claim_train_path = args.claim_train_path
        claim_dev_path = args.claim_dev_path

    rationale_train_set = SciFactRationaleSelectionDataset(args.corpus_path, claim_train_path)
    rationale_dev_set = SciFactRationaleSelectionDataset(args.corpus_path, claim_dev_path)
    label_train_set = SciFactLabelPredictionDataset(args.corpus_path, claim_train_path)
    label_dev_set = SciFactLabelPredictionDataset(args.corpus_path, claim_dev_path)

    # training model
    # args.rationale_model = 'model/rationale_best_model_SciBert/'
    # args.label_model = 'model/label_best_model_SciBert/'
    # args.rationale_model = 'model/rationale_roberta_large_scifact/'
    # args.label_model = 'model/label_roberta_large_fever_scifact/'
    args.embedding = 'SciBert'
    args.model = 'allenai/scibert_scivocab_cased'
    # args.embedding = 'roberta'
    # args.model = 'roberta-base'
    args.vocab_size = 50265
    args.lr_base = 1e-2  # 0.01
    args.k = 3
    # print(factory.get_model(args))
    if args.rationale_model == '' and args.label_model == '':
        print('training rationale selection model...')
        args.rational_model = train_rationale_selection(factory.get_model(args),
                                                        rationale_train_set, rationale_dev_set, args)
        print('training label prediction model...')
        args.num_label = 3
        args.label_model = train_label_prediction(factory.get_model(args), label_train_set, label_dev_set, args)
        args.rationale_model = 'model/rationale_best_model_SciBert/'
        args.label_model = 'model/label_best_model_SciBert/'
        # args.rationale_model = 'model/rationale_best_model_roberta/'
        # args.label_model = 'model/label_best_model_roberta/'

    tfidf_abstract(args.claim_dev_path, args.abstract_retrieval, 1, 2, args)  # 使用tfidf获得相关性高的前k(k=10)个摘要
    print('selection rationales...')
    rationale_results = get_rationale(args, args.claim_dev_path, args.rationale_model)
    evaluate_rationale_selection(args, rationale_results)  # evaluation rationale selection
    print(f'prediction labels({args.mode})...')
    label_results = get_labels(args, args.claim_dev_path, args.label_model)
    evaluate_label_predictions(args, label_results)  # evaluate label predictions

    print('merging predictions...')
    merge_rationale_label(args.rationale_selection, args.output_label, args, state='valid',
                          gold=args.gold)
    # model = torch.load(args.label_model+'pytorch_model.bin')
    # print(model.C)


if __name__ == "__main__":
    main()
