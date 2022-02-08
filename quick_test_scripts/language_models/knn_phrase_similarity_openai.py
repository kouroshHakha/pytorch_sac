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

import openai
from datasets import load_dataset

# parser
parser = argparse.ArgumentParser()

""" 
model_names:
text-similarity-ada-001 (cheap)
text-similarity-babbage-001
text-similarity-curie-001
text-similarity-davinci-001 (expensive)
"""
parser.add_argument('--model_name', '-mn', type=str)
parser.add_argument('--glue_id', default='qqp', choices=['mrpc', 'stsb', 'qqp'], type=str)
parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'], type=str)
# parser.add_argument('--num_proc', '-np', default=1, type=int)
# parser.add_argument('--with_train', action='store_true') # if true will include train split as well as validation
parser.add_argument('--display_examples', default=-1, type=int) # number of examples to show


pargs = parser.parse_args()

####################################################
root = Path('./lm_cache')
root.mkdir(exist_ok=True)
model_name = pargs.model_name
glue_id = pargs.glue_id

splits = ['validation']
# if pargs.with_train:
#     splits.append('train')

raw_datasets = load_dataset('glue', glue_id, split=splits)

seq_identifier = 'sentence' if glue_id in ('mrpc', 'stsb') else 'question'
seq_a_str = f'{seq_identifier}1'
seq_b_str = f'{seq_identifier}2'


def get_embedding(text_list):
    text_list = [text.replace("\n", " ") for text in text_list]
    # num_chars = sum([sum([len(word) for word in text]) for text in text_list])
    # print(num_chars)
    response = openai.Embedding.create(input=text_list, engine=pargs.model_name)['data']

    return np.array([response[i]['embedding'] for i in range(len(text_list))])

for idx, split_n in enumerate(splits):
    name = f'glue-{glue_id}-{split_n}-{model_name}'
    cache_fname = root / f'{name}.pt'
    if not cache_fname.exists():
        dataset = {}

        print('Getting the encodings ...')
        dset = raw_datasets[idx].select(range(1000))
        seq_a_list = dset[seq_a_str]
        seq_b_list = dset[seq_b_str]
        dataset['seq_a'] = get_embedding(seq_a_list)
        dataset['seq_b'] = get_embedding(seq_b_list)
        dataset['labels'] = dset['label']
        print('Encodings obtained.')


        torch.save(dataset, cache_fname)
    else:
        print('Reusing the saved embeddings ...')
        dataset = torch.load(cache_fname)
        print('Saved embeddings loaded.')

    ##################### train retreiver
    seq_a, seq_b, labels = dataset['seq_a'], dataset['seq_b'], dataset['labels']
    dist_matrix = pairwise_distances(seq_a, seq_b, metric=pargs.metric)
    top_k_list = [1, 3, 5, 10, 20]
    # top_k_list = [20]

    acc_pair_list, acc_unpair_list = [],  []
    print(f'[{split_n}]')
    print(f'{"top_k":10}, {"accuracy_pair":10}, {"accuracy_unpair":10}')
    
    for top_k in top_k_list:
        pairing_success_list = []
        unpairing_success_list = []
        count = 0
        for seq_a_idx, label in enumerate(labels):
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
    