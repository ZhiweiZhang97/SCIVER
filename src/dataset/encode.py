import torch
from typing import List


def encode_sen_pair(tokenizer, claims: List[str], sentences: List[str]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoded_dict = tokenizer.batch_encode_plus(
        list(zip(claims, sentences)),
        padding=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            list(zip(claims, sentences)),
            max_length=512,
            truncation_strategy='only_first',
            padding=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def encode_sentence(tokenizer, sentences: List[str]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoded_dict = tokenizer.batch_encode_plus(
        sentences,
        pad_to_max_length=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            sentences,
            max_length=512,
            truncation_strategy='only_first',
            pad_to_max_length=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict

# def encode(tokenizer, claims: List[str], sentences: List[str], max_sent_len=512):
#     def truncate(input_ids, max_length, sep_token_id, pad_token_id):
#         def longest_first_truncation(sentence, obj):
#             sent_lens = [len(sent) for sent in sentence]
#             while np.sum(sent_lens) > obj:
#                 max_position = np.argmax(sent_lens)
#                 sent_lens[max_position] -= 1
#             return [sentence[:length] for sentence, length in zip(sentence, sent_lens)]
#
#         all_paragraphs = []
#         for paragraph in input_ids:
#             valid_paragraph = paragraph[paragraph != pad_token_id]
#             if valid_paragraph.size(0) <= max_length:
#                 all_paragraphs.append(paragraph[:max_length].unsqueeze(0))
#             else:
#                 sep_token_idx = np.arange(valid_paragraph.size(0))[(valid_paragraph == sep_token_id).numpy()]
#                 idx_by_sentence = []
#                 prev_idx = 0
#                 for idx in sep_token_idx:
#                     idx_by_sentence.append(paragraph[prev_idx:idx])
#                     prev_idx = idx
#                 objective = max_length - 1 - len(idx_by_sentence[0])  # The last sep_token left out
#                 truncated_sentences = longest_first_truncation(idx_by_sentence[1:], objective)
#                 truncated_paragraph = torch.cat([idx_by_sentence[0]]
#                                                 + truncated_sentences + [torch.tensor([sep_token_id])], 0)
#                 all_paragraphs.append(truncated_paragraph.unsqueeze(0))
#
#         return torch.cat(all_paragraphs, 0)
#
#     encoded_dict = tokenizer.batch_encode_plus(
#         zip(claims, sentences),
#         pad_to_max_length=True,
#         add_special_tokens=True,
#         return_tensors='pt')
#     if encoded_dict['input_ids'].size(1) > max_sent_len:
#         if 'token_type_ids' in encoded_dict:
#             encoded_dict = {
#                 "input_ids": truncate(encoded_dict['input_ids'], max_sent_len,
#                                       tokenizer.sep_token_id, tokenizer.pad_token_id),
#                 'token_type_ids': encoded_dict['token_type_ids'][:, :max_sent_len],
#                 'attention_mask': encoded_dict['attention_mask'][:, :max_sent_len]
#             }
#         else:
#             encoded_dict = {
#                 "input_ids": truncate(encoded_dict['input_ids'], max_sent_len,
#                                       tokenizer.sep_token_id, tokenizer.pad_token_id),
#                 'attention_mask': encoded_dict['attention_mask'][:, :max_sent_len]
#             }
#     return encoded_dict
