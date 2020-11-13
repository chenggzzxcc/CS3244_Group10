import torch
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from dataset_bert import FRDataset
from model_bert import FR, DataParallel
from utils.model_utils import loop, collate_dict, get_device_ids

cuda = True
mixed_precision = False
clear_cache_every = 50
max_len = 512

device_ids = get_device_ids(cuda=cuda)
print('Using devices: {}'.format(device_ids))

BATCH_SIZE = 32
print('Batch size: {}'.format(BATCH_SIZE))
MINI_BATCH_SIZE = min((len(device_ids) << 4) if cuda else 32, BATCH_SIZE)
print('Mini batch size for training: {}'.format(MINI_BATCH_SIZE))
BATCH_SIZE_EVAL = (len(device_ids) << 5) if cuda else 64
print('Batch size for evaluation: {}'.format(BATCH_SIZE_EVAL))

assert BATCH_SIZE % MINI_BATCH_SIZE == 0
batch_iters = BATCH_SIZE // MINI_BATCH_SIZE

workers = max(min(16, MINI_BATCH_SIZE >> 3), 4)
workers_eval = max(min(8, BATCH_SIZE_EVAL >> 3), 4)

train, val = FRDataset.create_data('data/fakenewskdd2020/train.csv', split=(0.9, 0.1))
dataloader_train = DataLoader(train, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False, collate_fn=collate_dict)
dataloader_val = DataLoader(val, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=workers_eval, pin_memory=True, drop_last=False, collate_fn=collate_dict)

acc_best = 0
cnt = 0
patience = -1
epochs = 1
steps = epochs * len(dataloader_train)

model = FR(num_training_steps=steps, lr=1e-5, device_idxs=device_ids, mixed_precision=mixed_precision, cuda=cuda)

if cuda:
    # if len(device_ids) > 1:
    #     model = DataParallel(model, device_ids=device_ids, output_device=device_ids[-1])
    model.to('cuda:' + str(device_ids[0]))

for i in range(epochs):
    loop(model, dataloader_train, batch_iters=batch_iters, clear_cache_every=clear_cache_every, train=True, cuda=cuda, model_name='FR')
    loop(model, dataloader_val, batch_iters=1, clear_cache_every=clear_cache_every, train=False, cuda=cuda, model_name='FR', save_results=True)
