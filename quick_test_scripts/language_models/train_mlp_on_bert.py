import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from utils import write_yaml
from osil.debug import register_pdb_hook
register_pdb_hook()

from sklearn.metrics.pairwise import pairwise_distances

from datasets import load_dataset

# parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', '-mn', default='roberta-base', type=str)
parser.add_argument('--glue_id', default='qqp', choices=['mrpc', 'stsb', 'qqp'], type=str)
parser.add_argument('--batch_size', '-bs', default=128, type=int)
parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float)
parser.add_argument('--weight_decay', '-wc', default=0, type=float)
parser.add_argument('--num_proc', '-np', default=0, type=int)
parser.add_argument('--max_steps', default=50, type=int)
parser.add_argument('--display_examples', default=-1, type=int) # number of examples to show
parser.add_argument('--max_samples', default=-1, type=int) # max number of samples
parser.add_argument('--frac', default=1., type=float)
parser.add_argument('--ckpt', type=str)

pargs = parser.parse_args()

####################################################
root = Path('./lm_cache')
model_name = pargs.model_name
glue_id = pargs.glue_id

class ParaphraseEmbDataset(Dataset):

    def __init__(self, split='train') -> None:
        super().__init__()

        # name = f'glue-{glue_id}-{split}-{model_name}'
        name = 'glue-mrpc-validation-text-similarity-ada-001'
        path = root / f'{name}.pt'
        dataset = torch.load(path)

        self.seq_a, self.seq_b, self.labels = dataset['seq_a'], dataset['seq_b'], dataset['labels']

        self.seq_a = (self.seq_a - self.seq_a.mean(0)) / self.seq_a.std(0)
        self.seq_b = (self.seq_b - self.seq_b.mean(0)) / self.seq_b.std(0)

        self.inds = np.where(self.labels)[0]
    
    def __len__ (self):
        return len(self.inds)
        # return len(self.seq_a)
    
    def __getitem__(self, idx):
        index = self.inds[idx]
        # index = idx
        seq_a = torch.as_tensor(self.seq_a[index], dtype=torch.float)
        seq_b = torch.as_tensor(self.seq_b[index], dtype=torch.float)
        label = torch.as_tensor(self.labels[index], dtype=torch.float)

        return seq_a, seq_b, label
        # return seq_a, seq_b, label

class TransformEmbs(pl.LightningModule):

    def __init__(self, lr, wc, in_dim, out_dim, h_dim) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.wc = wc
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, out_dim),
            nn.Tanh(),
        )

        self.temp = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.xent = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wc)

    def forward(self, batch, compute_loss=False):
        """
        Bert embeddings on their own are not good for the prediction of this label.
        The sentences are too close to eachother to be separated out without any fine-tuning on the backbone.
        But maybe infoNCE loss will encourage similarity vs. disimilarity more?
        """
        seq_a, seq_b, label = batch
        z_a = self.net(seq_a)
        z_b = self.net(seq_b)
        if compute_loss:
            B, dim = z_a.size()
            labels = torch.arange(B).to(z_a.device).long()

            # normalize the norm to map it onto a unit sphere
            norm_z_a = torch.bmm(z_a.view(B, 1, dim), z_a.view(B, dim, 1)).view(B, 1) ** 0.5 # (B, 1)
            norm_z_b = torch.bmm(z_b.view(B, 1, dim), z_b.view(B, dim, 1)).view(B, 1) ** 0.5
            z_a = z_a / norm_z_a
            z_b = z_b / norm_z_b

            # inner product weighted by the trainable temprature
            logits = (z_a @ z_b.T) # * self.temp.exp() # (B, B)
            l_r = self.xent(logits, labels)
            l_c = self.xent(logits.T, labels)
            loss = 0.5 * (l_r + l_c)
            # loss = l_r

            # if self.training and self.global_step > 200:
            #     dist_mat = seq_a @ seq_b.T
            #     breakpoint()

            return z_a, z_b, loss
        return z_a, z_b

    def training_step(self, batch, batch_dx):
        _, _, loss = self(batch[0], compute_loss=True)
        return dict(loss=loss)

    def validation_step(self, batch, batch_dx):
        _, _, loss = self(batch, compute_loss=True)
        return dict(loss=loss)

    def training_epoch_end(self, outputs) -> None:
        train_loss_epoch = torch.mean(torch.stack([output['loss'] for output in outputs], 0))
        self.log('train_loss_epoch', train_loss_epoch)

    def validation_epoch_end(self, outputs) -> None:
        valid_loss_epoch = torch.mean(torch.stack([output['loss'] for output in outputs], 0))
        self.log('valid_loss_epoch', valid_loss_epoch)
    
total_set = ParaphraseEmbDataset('validation')
train_set = Subset(total_set, range(int(pargs.frac * len(total_set))))
valid_set = ParaphraseEmbDataset('validation')
sample = train_set[0][0]

# seq_a, seq_b, label = next(iter(DataLoader(total_set, batch_size=1000)))
# import matplotlib.pyplot as plt

# dist = ((seq_a - seq_b) ** 2).sum(-1)
# plt.hist(dist[label==0].numpy(), bins=50, range=(0,1), density=True, color='orange', alpha=0.5, label='not_eq')
# plt.hist(dist[label==1].numpy(), bins=50, range=(0,1), density=True, color='green', alpha=0.5, label='eq')
# plt.legend()
# # x = torch.concat([seq_a, seq_b], -1).numpy()

# # from sklearn.manifold import TSNE
# # x2d = TSNE(n_components=2).fit_transform(x)

# # plt.scatter(x2d[:, 0], x2d[:, 1], c=label.long().numpy(), s=5)
# plt.savefig('qqp.png')
# breakpoint()

train_loader = DataLoader(train_set, shuffle=True, batch_size=pargs.batch_size, num_workers=pargs.num_proc)
valid_loader = DataLoader(valid_set, shuffle=False, batch_size=pargs.batch_size, num_workers=pargs.num_proc)

if not pargs.ckpt:
    logger = WandbLogger(project='finetune-bert-emb')
    trainer = pl.Trainer(max_steps=pargs.max_steps, gpus=1, logger=logger)
    model = TransformEmbs(
        lr=pargs.learning_rate,
        wc=pargs.weight_decay,
        in_dim=sample.shape[-1],
        h_dim=1024,
        out_dim=sample.shape[-1],
    )
    trainer.fit(model, train_dataloaders=[train_loader], val_dataloaders=[valid_loader])
else:
    model = TransformEmbs.load_from_checkpoint(pargs.ckpt)

# """
# test the retriever based on cosine similarity
# get the test embs
print('Computing the test embeddings ... ')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_a_list, z_b_list, labels = [], [], []
model.to(device)
with torch.no_grad():
    for batch in valid_loader:
        seq_a, seq_b, label = batch
        z_a, z_b = model((seq_a.to(device), seq_b.to(device), label.to(device)))
        z_a_list.append(z_a.detach().cpu().numpy())
        z_b_list.append(z_b.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

z_a = np.concatenate(z_a_list, 0)
z_b = np.concatenate(z_b_list, 0)
labels = np.concatenate(labels, 0)
print('Test embeddings computed.')

print('Loading test dataset ...')
raw_datasets = load_dataset('glue', glue_id, split='validation')
seq_identifier = 'sentence' if glue_id in ('mrpc', 'stsb') else 'question'
seq_a_str = f'{seq_identifier}1'
seq_b_str = f'{seq_identifier}2'
print('Test dataset loaded.')

max_n = pargs.max_samples if pargs.max_samples > 0 else len(z_a)
dist_matrix = pairwise_distances(z_a[:max_n], z_b[:max_n], metric='cosine')
top_k_list = [1, 3, 5, 10, 20]

acc_list = []
print(f'{"top_k":10}, {"accuracy":10}')

raw_datasets = raw_datasets.select(range(max_n))
for top_k in top_k_list:
    success_list = []
    count = 0
    for seq_a_idx, label in enumerate(labels[:max_n]):
        if not label:
            continue 

        sorted_inds = dist_matrix[seq_a_idx].argsort()

        # select subset of dataset
        actual_pair = raw_datasets[seq_a_idx][seq_b_str]
        top_candidates = set([raw_datasets[int(top_idx)][seq_b_str] for top_idx in sorted_inds[:top_k]])
        success = actual_pair in top_candidates

        if count < pargs.display_examples and not success:
            print('-'*30)
            print(f'success = {success}')
            print(f'seq_a: {raw_datasets[seq_a_idx][seq_a_str]} -- {raw_datasets[seq_a_idx][seq_b_str]}')
            print('seq_b:')
            for top_idx in sorted_inds[:top_k]:
                print(f'[{dist_matrix[seq_a_idx][top_idx]:.4f}] {raw_datasets[int(top_idx)][seq_b_str]} -- {raw_datasets[int(top_idx)][seq_a_str]}')

        success_list.append(success)
        count += not success

    acc = sum(success_list) / len(success_list)
    acc_list.append(acc)

    print(f'{top_k:10d}, {acc:10.4f}')

name = f'glue-{glue_id}-{model_name}+mlp'
write_yaml(root / 'results' / f'summary-{name}.yaml', dict(acc_list=acc_list, top_k_list=top_k_list))
# """



