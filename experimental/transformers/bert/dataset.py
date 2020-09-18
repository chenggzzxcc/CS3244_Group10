import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
from transformers import BertTokenizerFast
import os
from utils.generic_utils import read_pairs
import random

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


class FRDataset():
    def __init__(self, data):
        self.data = data

    @classmethod
    def create_data(cls, path='data/train.csv', input_lengths=(64, 128, 256, 384, 512), overlap_ratios=(0.0, 0.15, 0.5, 0.9), max_length=24000, tokenizer=tokenizer, split=(1,), shuffle=True, testing=False):
        assert sum(split) == 1
        data = []
        cls_token = torch.tensor(101).unsqueeze(0)
        sep_token = torch.tensor(102).unsqueeze(0)
        if testing:
            raw_data = dict(read_pairs(path, cast=(int, str), offset=1))
            for i, (_id, text) in enumerate(raw_data.items()):
                sample = {}
                encodings = torch.tensor(tokenizer.encode(text, add_special_tokens=False))[:max_length]
                for overlap_ratio in overlap_ratios:
                    for input_length in input_lengths:
                            idx = 0
                            for j in range(0, len(encodings), int((input_length - 2) * (1 - overlap_ratio))):
                                encoding = torch.cat([cls_token, encodings[j: j + input_length - 2], sep_token])
                                sample['tokens_tensor'] = encoding
                                sample['tokens_type_id'] = torch.zeros_like(encoding, dtype=torch.long)
                                sample['attention_mask'] = torch.ones_like(encoding, dtype=torch.long)
                                sample['_id'] = _id
                                sample['text'] = text
                                sample['idx'] = idx
                                data.append(sample)
                                idx += 1
                                if j + input_length - 1 > encodings.shape[0]:
                                    break
        else:
            raw_data = dict(read_pairs(path, cast=(str, int), offset=1))
            for i, (text, label) in enumerate(raw_data.items()):
                sample = {}
                encodings = torch.tensor(tokenizer.encode(text, add_special_tokens=False))[:max_length]
                for overlap_ratio in overlap_ratios:
                    for input_length in input_lengths:
                        idx = 0
                        for j in range(0, len(encodings), int((input_length - 2) * (1 - overlap_ratio))):
                            encoding = torch.cat([cls_token, encodings[j: j + input_length - 2], sep_token])
                            sample['tokens_tensor'] = encoding
                            sample['tokens_type_id'] = torch.zeros_like(encoding, dtype=torch.long)
                            sample['attention_mask'] = torch.ones_like(encoding, dtype=torch.long)
                            sample['real'] = torch.tensor(label)
                            sample['text'] = text
                            sample['idx'] = idx
                            data.append(sample)
                            idx += 1
                            if j + input_length - 1 > encodings.shape[0]:
                                break
        if shuffle:
            random.shuffle(data)
        splits = []
        begin_idx = 0
        for i, s in enumerate(split):
            if i == len(split) - 1:
                end_idx = len(data)
            else:
                end_idx = round(begin_idx + len(data) * s)
            splits.append(FRDataset(data[begin_idx: end_idx]))
            begin_idx = end_idx
        return splits[0] if len(split) == 1 else splits

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, i):
        return self.data[i]
