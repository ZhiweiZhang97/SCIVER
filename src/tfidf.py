import argparse
import jsonlines
import numpy as np
from statistics import mean, median
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='../data/corpus.jsonl')
parser.add_argument('--dataset', type=str, default='../data/claims_dev.jsonl')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--min-gram', type=int, default=1)
parser.add_argument('--max-gram', type=int, default=2)
parser.add_argument('--output', type=str, default='prediction/abstract_retrieval_tfidf.jsonl')
args = parser.parse_args()

corpus = list(jsonlines.open(args.corpus))
dataset = list(jsonlines.open(args.dataset))
output = jsonlines.open(args.output, 'w')
k = args.k

vectorizer = TfidfVectorizer(stop_words='english',
                             ngram_range=(args.min_gram, args.max_gram))

doc_vectors = vectorizer.fit_transform([doc['title'] + ' '.join(doc['abstract'])
                                        for doc in corpus])
# print(doc_vectors)
doc_ranks = []

for data in dataset:
    claim = data['claim']
    claim_vector = vectorizer.transform([claim]).todense()
    doc_scores = np.asarray(doc_vectors @ claim_vector.T).squeeze()
    doc_indices_rank = doc_scores.argsort()[::-1].tolist()
    doc_id_rank = [corpus[idx]['doc_id'] for idx in doc_indices_rank]
    # for gold_doc_id in data['evidence'].keys():
    #     print('gold doc id:', gold_doc_id)
    #     rank = doc_id_rank.index(int(gold_doc_id))
    #     print('rank:', rank)
    #     print(200*'*')
    #     doc_ranks.append(rank)
#
#     output.write({
#         'claim_id': data['id'],
#         'doc_ids': doc_id_rank[:k]
#     })
#
# print(f'Mid reciprocal rank: {median(doc_ranks)}')
# print(f'Avg reciprocal rank: {mean(doc_ranks)}')
# print(f'Min reciprocal rank: {min(doc_ranks)}')
# print(f'Max reciprocal rank: {max(doc_ranks)}')
# print(doc_ranks)
"""
Performs sentence retrieval with oracle on SUPPORT and CONTRADICT claims,
and tfidf on NOT_ENOUGH_INFO claims
"""

# import argparse
# import jsonlines
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--corpus', type=str, default='../data/corpus.jsonl')
# parser.add_argument('--dataset', type=str, default='../data/claims_dev.jsonl')
# parser.add_argument('--abstract-retrieval', type=str, default='prediction/abstract_retrieval_tfidf.jsonl')
# parser.add_argument('--output', type=str, default='prediction/abstract_retrieval_ora_tfidf.jsonl')
# args = parser.parse_args()
#
# corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
# abstract_retrieval = jsonlines.open(args.abstract_retrieval)
# dataset = jsonlines.open(args.dataset)
# output = jsonlines.open(args.output, 'w')
#
# for data, retrieval in zip(dataset, abstract_retrieval):
#     assert data['id'] == retrieval['claim_id']
#
#     evidence = {}
#
#     for doc_id in retrieval['doc_ids']:
#         if data['evidence'].get(str(doc_id)):
#             evidence[doc_id] = [s for es in data['evidence'][str(doc_id)] for s in es['sentences']]
#         else:
#             sentences = corpus[doc_id]['abstract']
#             vectorizer = TfidfVectorizer(stop_words='english')
#             sentence_vectors = vectorizer.fit_transform(sentences)
#             claim_vector = vectorizer.transform([data['claim']]).todense()
#             sentence_scores = np.asarray(sentence_vectors @ claim_vector.T).squeeze()
#             top_sentence_indices = sentence_scores.argsort()[-2:][::-1].tolist()
#             top_sentence_indices.sort()
#             evidence[doc_id] = top_sentence_indices
#
#     output.write({
#         'claim_id': data['id'],
#         'evidence': evidence
#     })

