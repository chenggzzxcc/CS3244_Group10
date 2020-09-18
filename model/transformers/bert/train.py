import torch
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from model.transformers.bert.dataset import FRDataset
from model.transformers.bert.model import FakeReal, FRDataParallel
from utils.generic_utils import get_gpu_memory_usage
from tqdm import tqdm
import sys


def get_empty_cuda_devices():
    ret = []
    devices = get_gpu_memory_usage()
    for device in devices:
        if devices[device]['used'] < 13:
            ret.append(device)
    return ret


print('Using cuda')
all_devices = get_empty_cuda_devices()
device_ids = input('Found {} idle device(s): {}\nPlease choose which one(s) to use (default: \'all\'): '.format(len(all_devices), all_devices))
if not device_ids or device_ids.lower() == 'all':
    device_ids = all_devices
else:
    device_ids = sorted(set(int(i) for i in device_ids.split()))
print('Using devices: {}'.format(device_ids))
BATCH_SIZE = 32
print('Batch size: {}'.format(BATCH_SIZE))
MINI_BATCH_SIZE = 32
print('Mini batch size for training: {}'.format(MINI_BATCH_SIZE))
MINI_BATCH_SIZE_EVAL = 32
print('Mini batch size for evaluation: {}'.format(MINI_BATCH_SIZE_EVAL))
assert BATCH_SIZE % MINI_BATCH_SIZE == 0
batch_iters = BATCH_SIZE // MINI_BATCH_SIZE
workers = max(min(16, MINI_BATCH_SIZE >> 3), 4)
workers_eval = max(min(8, MINI_BATCH_SIZE_EVAL >> 3), 4)

clear_cache_every = 20


def collate(batch):
    return batch


train, dev = FRDataset.create_data('data/fakenewskdd2020/train.csv', split=(0.9, 0.1))
dataloader_train = DataLoader(train, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False, collate_fn=collate)
dataloader_dev = DataLoader(dev, batch_size=MINI_BATCH_SIZE_EVAL, shuffle=False, num_workers=workers_eval, pin_memory=True, drop_last=False, collate_fn=collate)

acc_best = 0
cnt = 0
patience = -1
epochs = 1
steps = epochs * len(dataloader_train)

model = FakeReal(batch_size=BATCH_SIZE, num_training_steps=steps, lr=1e-6, device_idxs=device_ids, mixed_precision=False)

if len(device_ids) > 1:
    model = FRDataParallel(model, device_ids=device_ids, output_device=device_ids[-1])
model.to('cuda:' + str(device_ids[0]))

for epoch in range(epochs):
    torch.cuda.empty_cache()
    model.reset()
    model.zero_grad()
    pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
    for i, batch in pbar:
        model.train_batch(batch)
        if i + batch_iters > len(dataloader_train):
            total_mini_batches = (len(dataloader_train) % batch_iters) or batch_iters
        else:
            total_mini_batches = batch_iters
        model.backward(r=1/total_mini_batches, l2=False)
        if (i + 1) % batch_iters == 0:
            model.optimize()
            model.zero_grad()
        if cuda and (i + 1) % clear_cache_every == 0:
            torch.cuda.empty_cache()
        pbar.set_description(model.print_loss())
    if len(dataloader_train) % batch_iters:
        model.optimize()
        model.zero_grad()
    acc = model.evaluate(dataloader_dev, 0, epoch + 1)

    if(acc >= acc_best):
        acc_best = acc
        cnt = 0
        best_model = model
        print(f'Best acc so far: {acc_best}')
    else:
        cnt += 1

    if cnt == patience or acc == 1.0: 
        print("Ran out of patient, early stop...")  
        break 
sys.exit(0)
