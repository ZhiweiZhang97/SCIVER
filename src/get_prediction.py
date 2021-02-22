import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
from tqdm import tqdm
import jsonlines

from src.dataset import loader, save_file
from src.dataset.save_file import save_rationale_selection, save_label_predictions
from src.dataset.encode import encode_sentence


def get_rationale(args, input_set, rationale_model, save_results=True, k=None):
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus_path)}
    dataset = jsonlines.open(input_set)
    abstract_retrieval = jsonlines.open(args.abstract_retrieval)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device "{device}"')

    tokenizer = AutoTokenizer.from_pretrained(rationale_model)
    model = AutoModelForSequenceClassification.from_pretrained(rationale_model).to(device).eval()

    results = []

    with torch.no_grad():
        for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
            assert data['id'] == retrieval['claim_id']
            claim = data['claim']

            evidence_scores = {}
            for doc_id in retrieval['doc_ids']:
                doc = corpus[doc_id]
                sentences = doc['abstract']

                encoded_dict = tokenizer.batch_encode_plus(
                    list(zip(sentences, [claim] * len(sentences))) if not args.only_rationale else sentences,
                    padding=True,
                    return_tensors='pt'
                )
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                sentence_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[:, 1].detach().cpu().numpy()
                evidence_scores[doc_id] = sentence_scores
            '''
            selection rationale from abstract.
            if the sentence score is greater than threshold(0.5), then it is evidence.
            '''
            if k:
                evidence = {doc_id: list(sorted(sentence_scores.argsort()[-k:][::-1].tolist()))
                            for doc_id, sentence_scores in evidence_scores.items()}
            else:
                evidence = {doc_id: (sentence_scores >= args.threshold).nonzero()[0].tolist()
                            for doc_id, sentence_scores in evidence_scores.items()}
            results.append({
                'claim_id': retrieval['claim_id'],
                'evidence': evidence
            })
    # save result of rational selection.
    if save_results:
        save_rationale_selection(args.rationale_selection, results)
    results = loader.loader_json(args.rationale_selection)

    return results


def encode(sentences, claims, args, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text = {
        "claim_and_rationale": list(zip(sentences, claims)),
        "only_claim": claims,
        "only_rationale": sentences
    }[args.mode]
    encoded_dict = tokenizer.batch_encode_plus(
        text,
        padding=True,
        return_tensors='pt'
    )
    if encoded_dict['input_ids'].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
            text,
            max_length=512,
            padding=True,
            truncation_strategy='only_first',
            return_tensors='pt'
        )
    encoded_dict = {key: tensor.to(device)
                    for key, tensor in encoded_dict.items()}
    return encoded_dict


def get_labels(args, input_set, label_model, save_results=True):
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus_path)}
    dataset = jsonlines.open(input_set)
    rationale_selection = jsonlines.open(args.rationale_selection)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device "{device}"')

    tokenizer = AutoTokenizer.from_pretrained(label_model)
    config = AutoConfig.from_pretrained(label_model, num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(label_model, config=config).eval().to(device)

    LABELS = ['CONTRADICT', 'NOT_ENOUGH_INFO', 'SUPPORT']
    label_results = []
    with torch.no_grad():
        for data, selection in tqdm(list(zip(dataset, rationale_selection))):
            assert data['id'] == selection['claim_id']

            claim = data['claim']
            results = {}
            for doc_id, indices in selection['evidence'].items():
                if not indices:
                    results[doc_id] = {'label': 'NOT_ENOUGH_INFO', 'confidence': 1}
                else:
                    evidence = ' '.join([corpus[int(doc_id)]['abstract'][i] for i in indices])
                    encoded_dict = encode([evidence], [claim], args, tokenizer)
                    label_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[0]
                    label_index = label_scores.argmax().item()
                    label_confidence = label_scores[label_index].item()
                    results[doc_id] = {'label': LABELS[label_index], 'confidence': round(label_confidence, 4)}
            label_results.append({
                'claim_id': data['id'],
                'labels': results
            })
    # save prediction result of label.
    if save_results:
        save_label_predictions(args.output_label, label_results)
    label_results = loader.loader_json(args.output_label)

    return label_results
