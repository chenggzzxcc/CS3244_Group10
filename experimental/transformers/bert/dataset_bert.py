import torch
from torch.utils.data import Dataset
from utils.generic_utils import read_pairs
import random
from base_model import tokenizer
from tqdm import tqdm


class FRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    @classmethod
    def create_data(cls, path='data/train.csv', input_lengths=(64, 256, 512), overlap_ratios=(0.2, 0.4), max_length=24000, tokenizer=tokenizer, split=(1,), shuffle=True, testing=False):
        assert sum(split) == 1
        data = []
        cls_token = torch.tensor(101).unsqueeze(0)
        sep_token = torch.tensor(102).unsqueeze(0)
        count = 0
        if testing:
            raw_data = read_pairs(path, cast=(int, str), offset=1)
            for i, (_id, text) in enumerate(tqdm(raw_data)):
                sample = {}
                encodings = torch.tensor(tokenizer.encode(text, add_special_tokens=False))[:max_length]
                for overlap_ratio in overlap_ratios:
                    for input_length in input_lengths:
                        for j in range(0, len(encodings), int((input_length - 2) * (1 - overlap_ratio))):
                            encoding = torch.cat([cls_token, encodings[j: j + input_length - 2], sep_token])
                            sample['input_ids'] = encoding
                            sample['tokens_type_ids'] = torch.zeros_like(encoding, dtype=torch.long)
                            sample['attention_masks'] = torch.ones_like(encoding, dtype=torch.long)
                            sample['_id'] = _id
                            sample['id'] = count
                            count += 1
                            data.append(sample)
                            if j + input_length - 1 > encodings.shape[0]:
                                break
        else:
            raw_data = read_pairs(path, cast=(str, int), offset=1)
            for i, (text, label) in enumerate(tqdm(raw_data)):
                sample = {}
                encodings = torch.tensor(tokenizer.encode(text, add_special_tokens=False))[:max_length]
                for overlap_ratio in overlap_ratios:
                    for input_length in input_lengths:
                        for j in range(0, len(encodings), int((input_length - 2) * (1 - overlap_ratio))):
                            encoding = torch.cat([cls_token, encodings[j: j + input_length - 2], sep_token])
                            sample['input_ids'] = encoding
                            sample['tokens_type_ids'] = torch.zeros_like(encoding, dtype=torch.long)
                            sample['attention_masks'] = torch.ones_like(encoding, dtype=torch.long)
                            sample['news'] = torch.tensor(label)
                            sample['id'] = count
                            count += 1
                            data.append(sample)
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
            subdata = data[begin_idx: end_idx]
            splits.append(FRDataset(subdata))
            begin_idx = end_idx
        return splits[0] if len(split) == 1 else splits

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, i):
        return self.data[i]
