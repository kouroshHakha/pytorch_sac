
import argparse
from pprint import pprint
import numpy as np
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from osil.data import collate_fn_for_supervised_osil, PointMazePairedDataset
from osil.nets import TOsilv1, TOsilSemisupervised
from utils import read_yaml, write_yaml
from osil.debug import register_pdb_hook
register_pdb_hook()

def get_ckpt_list(ckpt_yaml):
    content = read_yaml(ckpt_yaml)

    orders = list(map(int, [re.compile('.*step=(\d+).*').match(key)[1] for key in content]))
    order_map = dict(zip(content.keys(), orders))
    historically_sorted = sorted(content.keys(), key=lambda x: order_map[x])
    best_idx = np.argmin([content[key] for key in historically_sorted])

    return sorted(orders), historically_sorted, best_idx

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='osil', choices=['osil', 'semi-osil'])
    parser.add_argument('--dataset_path', default='./maze2d-open-v0_osil_short_trajs_v2', type=str)
    parser.add_argument('--ckpt_yaml', type=str)

    return parser.parse_args()

def main(pargs):
    pl.seed_everything(10)

    data_path = pargs.dataset_path
    output_dir = Path(pargs.ckpt_yaml).parent
    print(f'Running embedding plotting on {str(output_dir)} ...')
    
    steps, ckpt_list, best_ckpt_idx = get_ckpt_list(pargs.ckpt_yaml)

    pprint(ckpt_list)
    print(f'best_idx = {best_ckpt_idx}')

    # create a flattened dataset with class_ids
    dset = PointMazePairedDataset(data_path=data_path, mode='train')
    SPLITS = {
        'train': [3, 7, 12, 6, 8, 2, 10, 5, 11, 14, 1, 0], 
        'valid': [4], 
        'test':  [13, 9],
    }
    raw_data = dset.raw_data
    
    class_id = 0
    demo_states, demo_actions, demo_masks = [], [], []
    classes = []
    for task_id in raw_data:
        for var_id in raw_data[task_id]:
            episodes = [
                dict(
                    context_s=torch.as_tensor(ep['state'], dtype=torch.float),
                    context_a=torch.as_tensor(ep['action'], dtype=torch.float),
                )
            for ep in raw_data[task_id][var_id]
            ]
            episodes_padded = collate_fn_for_supervised_osil(episodes)
            demo_states.append(episodes_padded['context_s'])
            demo_actions.append(episodes_padded['context_a'])
            demo_masks.append(episodes_padded['attention_mask'])
            classes.append(torch.tensor([class_id]*len(episodes)))

            class_id += 1

    demo_states = torch.cat(demo_states, 0)
    demo_actions = torch.cat(demo_actions, 0)
    demo_masks = torch.cat(demo_masks, 0)
    classes = torch.cat(classes, 0).long()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fix the colors
    colors = np.zeros_like(classes)
    for idx, mode in enumerate(SPLITS):
        for c in SPLITS[mode]:
            colors[classes==c] = idx

    # get pca vectors of the best ckpt
    pca = None
    ckpt_order_of_pca = [ckpt_list[best_ckpt_idx]] + ckpt_list

    demo_2ds = []
    for idx, ckpt in tqdm.tqdm(enumerate(ckpt_order_of_pca)):
        if pargs.model == 'osil':
            agent = TOsilv1.load_from_checkpoint(ckpt)
        elif pargs.model == 'semi-osil':
            agent = TOsilSemisupervised.load_from_checkpoint(ckpt)
        else:
            raise ValueError('Unknown model')
    
        agent.to(device)
        # Putting model in eval mode is the key since it has dropout/norm layers
        agent.eval()

        dset = TensorDataset(demo_states.to(device), demo_actions.to(device), demo_masks.to(device))
        dloader = DataLoader(dset, batch_size=32, num_workers=0, shuffle=False)

        embs = []
        with torch.no_grad():
            for c_s, c_a, c_m in dloader:
                emb = agent.get_task_emb(c_s, c_a, c_m)
                embs.append(emb)

        demo_embs = torch.cat(embs, 0).detach().cpu().numpy()

        if pca is None:
            pca = PCA(n_components=2, svd_solver='full').fit(demo_embs)
        else:
            demo_2d = pca.transform(demo_embs)
            demo_2ds.append(demo_2d)
    
    cat_demos = np.concatenate(demo_2ds, 0)
    max_xy = cat_demos.max(0)
    min_xy = cat_demos.min(0)
    for idx, demo_2d in enumerate(demo_2ds):
        plt.close()
        s_plt = plt.scatter(demo_2d[:, 0], demo_2d[:, 1], s=5, c=colors, cmap=mcolors.ListedColormap(['blue', 'orange', 'green']))
        h,l = s_plt.legend_elements()
        plt.xlim(min_xy[0]*1.05, max_xy[0]*1.05)
        plt.ylim(min_xy[1]*1.05, max_xy[1]*1.05)
        plt.legend(handles = h, labels=['train', 'valid', 'test'])
        plt.title(f'step = {steps[idx]:4d}')
        plt.savefig(output_dir / f'demo_embs_{idx}.png')

    # print('Running KNN for k = [1, 3, 5, 10, 25, 50, 100]')
    # k_list = [1, 3, 5, 10, 25, 50, 100]

    # X_train, X_test, y_train, y_test = train_test_split(demo_embs, classes, test_size=0.2, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    # train_accs, valid_accs, test_accs = [], [], []
    # for k in tqdm.tqdm(k_list):
    #     classifier = KNeighborsClassifier(k)
    #     classifier.fit(X_train, y_train)
    #     ## train 
    #     pred_train = classifier.predict(X_train)
    #     acc = accuracy_score(y_train, pred_train)
    #     train_accs.append(float(acc))
    #     ## valid
    #     pred_val = classifier.predict(X_val)
    #     acc = accuracy_score(y_val, pred_val)
    #     valid_accs.append(float(acc))

    #     ## test
    #     pred_test = classifier.predict(X_test)
    #     acc = accuracy_score(y_test, pred_test)
    #     test_accs.append(float(acc))

    # opt_index = int(np.argmax(valid_accs))
    # k_opt = k_list[opt_index]
    # print(f'{"k":2}, {"train_acc":10}, {"valid_acc":10}, {"test_acc":10}')
    # for k, t_acc, v_acc, test_acc in zip(k_list, train_accs, valid_accs, test_accs):
    #     print(f'{k:2d}, {t_acc:10.3f}, {v_acc:10.3f}, {test_acc:10.3f}')
    # print('---------------')
    # print(f'best model, k = {k_opt}, test_acc = {test_accs[opt_index]:.3f}')

    # results = dict(
    #     k_list=k_list,
    #     train_accs=train_accs,
    #     valid_accs=valid_accs,
    #     test_accs=test_accs,
    #     best=dict(
    #         k=k_opt,
    #         train_acc=train_accs[opt_index],
    #         valid_acc=valid_accs[opt_index],
    #         test_acc=test_accs[opt_index],
    #     )   
    # )
    # write_yaml(output_dir / 'knn_results.yaml', results)


    # # compute trajectory retrieval scores
    # print('Computing trajectory retrieval score ...')
    # # last k should be 99 since except the index itself there are 99 others
    # k_list = [1, 3, 5, 10, 25, 50, 100]

    # tr_score_list = []
    # for k in tqdm.tqdm(k_list):
    #     query_acc_list = []
    #     for idx in range(len(demo_embs)):
    #         dist = np.sqrt(((demo_embs - demo_embs[idx]) ** 2).sum(-1))
    #         dist[idx] = float('inf')
    #         cand_inds = np.argsort(dist)[:k]
    #         retrieved_classes = classes[cand_inds]
    #         query_acc = (retrieved_classes == classes[idx]).sum() / k
    #         query_acc_list.append(query_acc)
    #     tr_score = np.mean(query_acc_list)
    #     tr_score_list.append(float(tr_score))

    # print(f'{"k":2}, {"tr_score":10}')
    # for k, tr_score in zip(k_list, tr_score_list):
    #     print(f'{k:2d}, {tr_score:10.3f}')

    # results = dict(k_list=k_list, tr_scores=tr_score_list)
    # write_yaml(output_dir / 'tr_results.yaml', results)

if __name__ == '__main__':
    main(_parse_args())