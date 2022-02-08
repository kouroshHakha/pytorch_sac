from collections import defaultdict
from nbformat import write
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from pathlib import Path

from osil.debug import register_pdb_hook
register_pdb_hook()

from sklearn.metrics.pairwise import pairwise_distances

from utils import write_yaml

import argparse

from transformers import AutoConfig, AutoModel, AutoTokenizer
from datasets import load_dataset

# parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', '-mn', type=str)
parser.add_argument('--glue_id', default='qqp', choices=['mrpc', 'stsb', 'qqp'], type=str)
parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean', 'mlp'], type=str)
parser.add_argument('--num_proc', '-np', default=1, type=int)
parser.add_argument('--split', default='validation', type=str)
parser.add_argument('--display_examples', default=-1, type=int) # number of examples to show
parser.add_argument('--max_samples', default=-1, type=int) # max number of samples
parser.add_argument('--batch_size', '-bs', default=128, type=int)

pargs = parser.parse_args()

####################################################
root = Path('./lm_cache')
root.mkdir(exist_ok=True)
model_name = pargs.model_name
glue_id = pargs.glue_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

splits = [pargs.split]

# load tokenizer, model, and dataset
configuration = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
raw_datasets = load_dataset('glue', glue_id, split=splits)

seq_identifier = 'sentence' if glue_id in ('mrpc', 'stsb') else 'question'
seq_a_str = f'{seq_identifier}1'
seq_b_str = f'{seq_identifier}2'

# tokenize and preprocess the input data
def tokenize_function(examples):
    seq_a_tokens = tokenizer(examples[seq_a_str], padding="max_length", truncation=True)
    seq_b_tokens = tokenizer(examples[seq_b_str], padding="max_length", truncation=True)
    
    data = {f'{k}_a': v for k, v in seq_a_tokens.items()}
    data.update({f'{k}_b': v for k, v in seq_b_tokens.items()})
    return data

for idx, split_n in enumerate(splits):
    name = f'glue-{glue_id}-{split_n}-{model_name}'
    cache_fname = root / f'{name}.pt'
    if not cache_fname.exists():
        dataset = {}
        model.to(device)

        print('Loading the tokenized dataset ...')
        dset = raw_datasets[idx].map(tokenize_function, batched=False, num_proc=pargs.num_proc)
        dset = dset.remove_columns([seq_a_str, seq_b_str, 'idx'])
        dset.set_format(type='torch', columns=['input_ids_a', 'input_ids_b', 'attention_mask_a', 'attention_mask_b', 'label'], device=device)
        print('Tokenized dataset loaded.')

        loader = torch.utils.data.DataLoader(dset, batch_size=pargs.batch_size)

        seq_a_list, seq_b_list = [], []
        labels = []
        print('Getting the encodings ...')
        with torch.no_grad():
            for batch in tqdm.tqdm(loader):
                seq_a = batch['input_ids_a']
                seq_b = batch['input_ids_b']

                attn_mask_a = batch['attention_mask_a']
                attn_mask_b = batch['attention_mask_b']

                label_b = batch['label']

                output_a = model(input_ids=seq_a, attention_mask=attn_mask_a)
                output_b = model(input_ids=seq_b, attention_mask=attn_mask_b)

                seq_a_list.append(output_a['pooler_output'])
                seq_b_list.append(output_b['pooler_output'])
                labels.append(label_b)
        print('Encodings obtained.')

        dataset['seq_a'] = torch.cat(seq_a_list, 0).detach().cpu().numpy()
        dataset['seq_b'] = torch.cat(seq_b_list, 0).detach().cpu().numpy()
        dataset['labels'] = torch.cat(labels, 0).detach().cpu().numpy()

        torch.save(dataset, cache_fname)
    else:
        print('Reusing the saved embeddings ...')
        dataset = torch.load(cache_fname)
        print('Saved embeddings loaded.')


    ##################### train retreiver (based on heurstic distance)
    if pargs.metric in ['cosine', 'euclidean']:
        seq_a, seq_b, labels = dataset['seq_a'], dataset['seq_b'], dataset['labels']
        max_n = pargs.max_samples if pargs.max_samples > 0 else len(seq_a)
        dist_matrix = pairwise_distances(seq_a[:max_n], seq_b[:max_n], metric=pargs.metric)
        top_k_list = [1, 3, 5, 10, 20]
        # top_k_list = [20]

        acc_pair_list, acc_unpair_list = [],  []
        print(f'[{split_n}]')
        print(f'{"top_k":10}, {"accuracy_pair":10}, {"accuracy_unpair":10}')
        
        raw_datasets[idx] = raw_datasets[idx].select(range(max_n))
        for top_k in top_k_list:
            pairing_success_list = []
            unpairing_success_list = []
            count = 0
            for seq_a_idx, label in enumerate(labels[:max_n]):
                actual_pair = raw_datasets[idx][seq_a_idx][seq_b_str]
                sorted_inds = dist_matrix[seq_a_idx].argsort()
                top_candidates = set([raw_datasets[idx][int(top_idx)][seq_b_str] for top_idx in sorted_inds[:top_k]])
                
                if not label:
                    # the actual pair should not be in top_k
                    success = actual_pair not in top_candidates
                    unpairing_success_list.append(success)
                
                else:
                    # the actual pair should be in top_k
                    success = actual_pair in top_candidates
                    pairing_success_list.append(success)

                if count < pargs.display_examples and not success:
                    print('-'*30)
                    print(f'label = {label}, success = {success}')
                    print(f'seq_a: {raw_datasets[idx][seq_a_idx][seq_a_str]} -- {raw_datasets[idx][seq_a_idx][seq_b_str]}')
                    print('seq_b:')
                    for top_idx in sorted_inds[:top_k]:
                        print(f'[{dist_matrix[seq_a_idx][top_idx]:.4f}] {raw_datasets[idx][int(top_idx)][seq_b_str]} -- {raw_datasets[idx][int(top_idx)][seq_a_str]}')

                count += not success

            acc_pair = sum(pairing_success_list) / len(pairing_success_list)
            acc_unpair = sum(unpairing_success_list) / len(unpairing_success_list)
            acc_pair_list.append(acc_pair)
            acc_unpair_list.append(acc_unpair)

            print(f'{top_k:10d}, {acc_pair:10.4f}, {acc_unpair:10.4f}')

        write_yaml(root / 'results' / f'summary-{name}-{pargs.metric}.yaml', dict(acc_pair_list=acc_pair_list, acc_unpair_list=acc_unpair_list, top_k_list=top_k_list))
    



