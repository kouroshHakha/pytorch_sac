from nbformat import write
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from pathlib import Path

from osil.debug import register_pdb_hook
register_pdb_hook()

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from utils import write_yaml

import argparse

# from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from datasets import load_dataset, list_datasets, list_metrics, load_metric

# parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', '-mn', type=str)
parser.add_argument('--dataset', default='imdb', type=str)
pargs = parser.parse_args()

####################################################
root = Path('./lm_cache')
root.mkdir(exist_ok=True)
model_name = pargs.model_name
dataset_name = pargs.dataset
cache_fname = root / f'{dataset_name}-{model_name}.pt'

if not cache_fname.exists():
    configuration = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw_datasets = load_dataset(dataset_name)
    # print(raw_datasets)

    train_texts = raw_datasets['train']['text']
    train_labels = raw_datasets['train']['label']
    test_texts = raw_datasets['test']['text']
    test_labels = raw_datasets['test']['label']

    # computing some basic stats
    train_seq_lens = [len(x) for x in train_texts]
    test_seq_lens = [len(x) for x in test_texts]

    print('-'*30 + ' // train dataset')
    print(f'size: {len(train_texts)}, max_seq_len: {max(train_seq_lens)}, min_seq_len: {min(train_seq_lens)}, avg_seq_len: {np.mean(train_seq_lens)}')
    print(f'positive ratio: {sum(train_labels) / len(train_labels)}')


    print('-'*30 + ' // test dataset')
    print(f'size: {len(test_seq_lens)}, max_seq_len: {max(test_seq_lens)}, min_seq_len: {min(test_seq_lens)}, avg_seq_len: {np.mean(test_seq_lens)}')
    print(f'positive ratio: {sum(test_labels) / len(test_labels)}')


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'], device=device)
    full_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'], device=device)


    train_dset = full_train_dataset
    tloader = torch.utils.data.DataLoader(train_dset, batch_size=128)

    # train set features
    train_feats = []
    train_labels = []
    model.to(device)

    with torch.no_grad():
        for batch in tqdm.tqdm(tloader):
            xin = batch['input_ids']
            attn_mask = batch['attention_mask']
            labels = batch['label']

            output = model(input_ids=batch['input_ids'], attention_mask=attn_mask)
            train_feats.append(output['pooler_output'])
            train_labels.append(labels)

    train_feats = torch.cat(train_feats, 0).detach().cpu().numpy()
    train_labels = torch.cat(train_labels, 0).detach().cpu().numpy()


    test_dset = full_eval_dataset
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=128)

    # test set features
    test_feats = []
    test_labels = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            xin = batch['input_ids']
            attn_mask = batch['attention_mask']
            labels = batch['label']

            output = model(input_ids=batch['input_ids'], attention_mask=attn_mask)
            test_feats.append(output['pooler_output'])
            test_labels.append(labels)

    test_feats = torch.cat(test_feats, 0).detach().cpu().numpy()
    test_labels = torch.cat(test_labels, 0).detach().cpu().numpy()
    
    to_save = dict(
        train_feats=train_feats,
        train_labels=train_labels,
        test_feats=test_feats,
        test_labels=test_labels,
    )
    torch.save(to_save, cache_fname)
else:
    to_load = torch.load(cache_fname)
    train_feats = to_load['train_feats']
    train_labels = to_load['train_labels']
    test_feats = to_load['test_feats']
    test_labels = to_load['test_labels']

##################### train knn

k_list = [1, 3, 5, 8, 10, 15, 20, 25, 30]
train_acc_list = []
test_acc_list = []
for k in k_list:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_feats, train_labels)

    test_preds = knn_model.predict(test_feats)
    train_preds = knn_model.predict(train_feats)
    train_acc = metrics.accuracy_score(train_labels, train_preds)
    test_acc = metrics.accuracy_score(test_labels, test_preds)
    print('-'*15 + f'// k = {k}')
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)

    train_acc_list.append(float(train_acc))
    test_acc_list.append(float(test_acc))

write_yaml(root / 'results' / f'summary-{dataset_name}-{model_name}.yaml', dict(train_acc=train_acc_list, test_acc=test_acc_list, ks=k_list))

