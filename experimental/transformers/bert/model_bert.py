from transformers import AutoModel, AdamW, AutoConfig
from collections import defaultdict
from torch.nn import functional as F
from tqdm import tqdm

from base_model import *


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx).float()
    if mask.sum().item() == 0:
        return 0
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask
    loss = losses.sum() / mask.sum()
    return loss


class FR(BaseModule):
    def __init__(self,
                 transformer_model=transformer_model,
                 feature_extractor=None,
                 lr=5e-5,
                 transformer_dropout=0.1,
                 soft_classifier_dropout=0.4,
                 cuda=True,
                 warmup_ratio=0.1,
                 num_training_steps=1000,
                 device_idxs=(),
                 mixed_precision=False):
        super().__init__(cuda=cuda, warmup_ratio=warmup_ratio, num_training_steps=num_training_steps,
                         device_idxs=device_idxs, mixed_precision=mixed_precision)

        self.config = AutoConfig.from_pretrained(transformer_model,
                                                 hidden_dropout_prob=transformer_dropout,
                                                 attention_probs_dropout_prob=transformer_dropout)

        # Main transformer model
        self.transformer = AutoModel.from_pretrained(transformer_model, config=self.config)

        # Head layers
        self.news_classifier = nn.Linear(self.config.hidden_size, 2)

        # Loss Function
        self.cross_entropy = nn.CrossEntropyLoss()

        # Other parameters
        self.soft_dropout = nn.Dropout(soft_classifier_dropout)
        self.reset()

        # Multi-GPU training support
        if self.cuda:
            self.transformer = DataParallel(self.transformer,
                                            device_ids=device_idxs,
                                            output_device=self.model_device)

        # Mixed precision training support
        if self.mixed_precision:
            self.transformer.forward = insert_autocast(self.transformer.forward)

        # Optimizer
        self.optimizer = AdamW(self.parameters(), lr=lr)
        self.scheduler = self.linear_scheduler(self.optimizer)

        self.main_losses = {'news'}

    def forward(self, input_ids, tokens_type_ids=None, attention_masks=None):
        input_ids = input_ids.to(self.model_device)
        tokens_type_ids = tokens_type_ids.to(self.model_device)
        attention_masks = attention_masks.to(self.model_device)

        pooled_output = self.transformer(input_ids, token_type_ids=tokens_type_ids, attention_mask=attention_masks)[1]

        # Classification
        news = self.news_classifier(self.soft_dropout(pooled_output)).squeeze(1)  # B,2

        return {
            'news': news
        }

    def reset(self):
        self.loss, self.loss_news, self.iter = 0, 0, 1

    def print_loss(self):
        loss_avg = self.loss / self.iter
        loss_news = self.loss_news / self.iter
        self.iter += 1
        return 'L:{:.2f} LN:{:.3f}'.format(loss_avg, loss_news)

    def prepare(self, batch):
        input_ids = batch['input_ids']
        tokens_type_ids = batch['tokens_type_ids']
        ids = batch['id']

        if 'news' in batch:
            news_labels = batch['news']
            news_labels = BaseModule.pad_seq(news_labels, val=self.cross_entropy.ignore_index)
        else:
            news_labels = None
        input_ids = BaseModule.pad_seq(input_ids)
        tokens_type_ids = BaseModule.pad_seq(tokens_type_ids, val=1)
        attention_masks = (input_ids != 0).float()

        return {
            'input_ids': input_ids,
            'tokens_type_ids': tokens_type_ids,
            'attention_masks': attention_masks
               }, {
            'news_labels': news_labels,
            'ids': ids
        }

    def calculate_loss(self, outputs, extras):
        news = outputs['news']
        news_labels = extras['news_labels']

        # News loss
        # return masked_cross_entropy_for_value(gens.contiguous(), gen_labels.to(self.model_device).contiguous())
        return F.cross_entropy(news, news_labels.to(self.model_device))

    def accumulate_loss(self, outputs, extras):
        # News loss
        batch_loss_news = self.calculate_loss(outputs, extras)

        # Add all losses
        loss = batch_loss_news
        self.loss_news += batch_loss_news.item()

        self.loss_grad = loss
        self.loss += loss.data

    def make_results(self, outputs, extras):
        news = outputs['news']
        news_labels = extras['news_labels']
        ids = extras['ids']

        news = news.argmax(-1).long()

        results = defaultdict(list)
        for i, news_label in enumerate(news_labels):
            news_gt = news_label.detach().cpu()
            news_prediction = news[i].detach().cpu()

            results[ids[i]].append({
                'predictions': {
                    'news': news_prediction
                },
                'gts': {
                    'news': news_gt
                }
            })

        return results

    def predict(self, test, thresh=0.5):
        self.eval()

        with torch.no_grad():
            print("STARTING TESTING")
            all_predictions = defaultdict(list)
            pbar = tqdm(enumerate(test), total=len(test))
            for i, batch in pbar:
                inputs, extras = self.prepare(batch)
                outputs = self(**inputs)
                fake_real = outputs['news'].detach().cpu().argmax(dim=-1).numpy()

                for j, _id in enumerate(batch['_id']):
                    fake_real_prediction = True if fake_real[j] == 1 else False

                    all_predictions[_id].append({
                        'fake_real_prediction': fake_real_prediction
                    })
            results = {}
            for _id, predictions in all_predictions.items():
                fake_real_prediction = torch.tensor([prediction['fake_real_prediction'] for prediction in predictions]).float().sum().item() / len(predictions) > thresh
                if not fake_real_prediction:
                    print(_id)
                results[_id] = fake_real_prediction
            return results
