import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from transformers import BertModel, BertConfig, AdamW
from torch.optim import lr_scheduler
from torch.nn.functional import pad
from activations import MishME
from utils.json_utils import save
from collections import defaultdict
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils.generic_utils import current_time


class FRDataParallel(nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.module, name)


def inserted_func(*args, **kwargs):
    with autocast(value):
        return func(*args, **kwargs)


def insert_autocast(func, value=True):
    globals()['func'] = func
    globals()['value'] = value
    return inserted_func


class FakeReal(nn.Module):
    def __init__(self, batch_size=64, lr=1e-6, scale_lr_with_batch_size=True, dropout_bert=0.25, dropout_linear=0.5, warmup_ratio=0.1, num_training_steps=1000, device_idxs=[], mixed_precision=False):
        super().__init__()
        self.mixed_precision = mixed_precision
        self.config = BertConfig(hidden_dropout_prob=dropout_bert, attention_probs_dropout_prob=dropout_bert)
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=self.config)
        self.devices = device_idxs
        self.model_device = device_idxs[0]
        self.bert = FRDataParallel(self.bert, device_ids=device_idxs, output_device=self.model_device)
        if self.mixed_precision:
            self.scaler = GradScaler()
            self.bert.forward = insert_autocast(self.bert.forward)
        self.dropout = nn.Dropout(dropout_linear, inplace=False)
        self.linear = nn.Linear(self.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss()
        if scale_lr_with_batch_size:
            lr *= batch_size
        self.num_warmup_steps = int(warmup_ratio * num_training_steps)
        self.num_training_steps = num_training_steps
        self.optimizer = AdamW(self.parameters(), lr=lr)
        self.scheduler = self.linear_scheduler(self.optimizer)

    def linear_scheduler(self, optimizer, last_epoch=-1):
        return lr_scheduler.LambdaLR(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        )

    def forward(self, tokens_tensors, tokens_type_ids, attention_masks):
        logits = self.bert(tokens_tensors, token_type_ids=tokens_type_ids, attention_mask=attention_masks)[1]
        logits = self.dropout(logits)
        return self.linear(logits)

    def reset(self):
        self.loss, self.loss_fake_real, self.iter = 0, 0, 1

    def print_loss(self):
        loss_avg = self.loss / self.iter
        loss_fake_real = self.loss_fake_real / self.iter
        self.iter += 1
        return 'L:{:.2f} LFR:{:.3f}'.format(loss_avg, loss_fake_real)

    def backward(self, r=1, l2=True):
        self.loss_grad = self.loss_grad * r
        if l2:
            if self.mixed_precision:
                grad_params = torch.autograd.grad(self.scaler.scale(self.loss_grad), self.parameters(), create_graph=True)
                inv_scale = 1 / self.scaler.get_scale()
                grad_params = [p * inv_scale for p in grad_params]
            else:
                grad_params = torch.autograd.grad(self.loss_grad, self.parameters(), create_graph=True)
            with autocast(self.mixed_precision):
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                self.loss_grad = self.loss_grad + grad_norm
        if self.mixed_precision:
            self.scaler.scale(self.loss_grad).backward()
        else:
            self.loss_grad.backward()


    def optimize(self, clip=True):
        if clip:
            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()

    def add_padding(self, tokens_tensors, tokens_type_ids, attention_masks):
        max_len = max(tensor.shape[0] for tensor in tokens_tensors)
        for i, tensor in enumerate(tokens_tensors):
            tokens_tensors[i] = pad(tensor, pad=(0, max_len - tensor.shape[-1]))
            tokens_type_ids[i] = pad(tokens_type_ids[i], pad=(0, max_len - tokens_type_ids[i].shape[-1]))
            attention_masks[i] = pad(attention_masks[i], pad=(0, max_len - attention_masks[i].shape[-1]))

    def prepare(self, batch):
        tokens_tensors = [sample['tokens_tensor'] for sample in batch]
        tokens_type_ids = [sample['tokens_type_id'] for sample in batch]
        attention_masks = [sample['attention_mask'] for sample in batch]
        self.add_padding(tokens_tensors, tokens_type_ids, attention_masks)
        tokens_tensors = torch.stack(tokens_tensors)
        tokens_type_ids = torch.stack(tokens_type_ids)
        attention_masks = torch.stack(attention_masks)
        return tokens_tensors, tokens_type_ids, attention_masks

    def train_batch(self, batch):
        self.train()
        loss = 0
        tokens_tensors, tokens_type_ids, attention_masks = self.prepare(batch)
        tokens_tensors = tokens_tensors.to(self.model_device)
        tokens_type_ids = tokens_type_ids.to(self.model_device)
        attention_masks = attention_masks.to(self.model_device)
        fake_real_labels = torch.stack([sample['real'] for sample in batch])
        fake_real_labels = fake_real_labels.to(self.model_device)

        with autocast(self.mixed_precision):
            logits = self(tokens_tensors, tokens_type_ids, attention_masks)
            loss_fake_real = self.cross_entropy(logits, fake_real_labels)

            loss = loss + loss_fake_real
            self.loss_fake_real = self.loss_fake_real + loss_fake_real.item()
            self.loss_grad = loss
            self.loss = self.loss + loss.data

    def evaluate(self, dev, matric_best, epoch=0):
        self.eval()

        with torch.no_grad():
            print("STARTING EVALUATION")
            all_predictions = defaultdict(list)
            pbar = tqdm(enumerate(dev), total=len(dev))
            for i, batch in pbar:
                tokens_tensors, tokens_type_ids, attention_masks = self.prepare(batch)
                logits = self(tokens_tensors, tokens_type_ids, attention_masks)

                fake_real = self.softmax(logits)
                fake_real = fake_real.argmax(dim=-1).cpu().numpy()

                for j, sample in enumerate(batch):
                    fake_real_prediction = True if fake_real[j] == 1 else False
                    fake_real_gt = True if sample['real'].item() else False

                    all_predictions[sample['text']].append({
                        'fake_real_prediction': fake_real_prediction,
                        'fake_real_gt': fake_real_gt,
                        'idx': sample['idx']
                    })
            loss_avg = self.loss / self.iter
            now = current_time().replace('/', '-')
            save(all_predictions, f'prediction/all_prediction_{epoch}_{loss_avg}_{now}.json')

            joint_acc_fake_real_score, joint_acc_score = FakeReal.evaluate_metrics(all_predictions)

            evaluation_metrics = {'Joint Acc Fake Real': joint_acc_fake_real_score,
                                  'Joint Acc': joint_acc_score}
            print(evaluation_metrics)

            if (joint_acc_score >= matric_best):
                self.save_model('FakeReal_ACC-{:.4f} '.format(joint_acc_score) + current_time().replace('/', '-'))
                print("MODEL SAVED")
            return joint_acc_score

    def save_model(self, name):
        torch.save(self, os.path.join('checkpoint', name + '.th'))

    @classmethod
    def evaluate_metrics(cls, all_predictions):
        joint_acc_fake_real, joint_acc, total = 0, 0, 0
        for predictions in all_predictions.values():
            total += 1
            fake_real_correct = True
            for prediction in predictions:
                fake_real_gt = prediction['fake_real_gt']
                fake_real_prediction = prediction['fake_real_prediction']
                if fake_real_prediction != fake_real_gt:
                    fake_real_correct = False

            if fake_real_correct:
                joint_acc_fake_real += 1

            if fake_real_correct:
                joint_acc += 1

        joint_acc_fake_real_score = joint_acc_fake_real / total if total != 0 else -1
        joint_acc_score = joint_acc / total if total != 0 else -1

        return joint_acc_fake_real_score, joint_acc_score

    def predict(self, test):
        self.eval()

        with torch.no_grad():
            print("STARTING TESTING")
            all_predictions = defaultdict(list)
            pbar = tqdm(enumerate(test), total=len(test))
            for i, batch in pbar:
                tokens_tensors, tokens_type_ids, attention_masks = self.prepare(batch)
                tokens_tensors = tokens_tensors.to(self.model_device)
                tokens_type_ids = tokens_type_ids.to(self.model_device)
                attention_masks = attention_masks.to(self.model_device)
                logits = self(tokens_tensors, tokens_type_ids, attention_masks)

                fake_real = self.softmax(logits)
                fake_real = fake_real.argmax(dim=-1).cpu().numpy()

                for j, sample in enumerate(batch):
                    fake_real_prediction = True if fake_real[j] == 1 else False

                    all_predictions[sample['_id']].append({
                        'fake_real_prediction': fake_real_prediction
                    })
            results = {}
            for _id, predictions in all_predictions.items():
                fake_real_prediction = bool(round(torch.tensor([prediction['fake_real_prediction'] for prediction in predictions]).float().sum().item() / len(predictions)))
                results[_id] = fake_real_prediction
            return results
