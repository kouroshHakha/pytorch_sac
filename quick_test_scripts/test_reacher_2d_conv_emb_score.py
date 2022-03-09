
import argparse
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from osil.data import collate_fn_for_supervised_osil
from osil.datasets.reacher2d.gcbc import Reacher2DGCBCDataset
from osil.nets import GCBCv5
from utils import write_yaml


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--dataset_path', default='./reacher_2d_train_v3', type=str)
    # parser.add_argument('--max_padding', default=50, type=int)
    parser.add_argument('--metric', default='euclidean', type=str)
    parser.add_argument('--random', action='store_true')

    return parser.parse_args()

def main(pargs):
    print(f'Running embedding evalution on {pargs.ckpt} ...')
    pl.seed_everything(10)

    data_path = Path(pargs.dataset_path)
    output_dir = Path(pargs.ckpt).parent

    conf = None
    if pargs.random:
        conf = torch.load(pargs.ckpt)['hyper_parameters']

    agent = GCBCv5.load_from_checkpoint(pargs.ckpt)
    
    # create a flattened dataset with class_ids
    dset = Reacher2DGCBCDataset(data_path=data_path, image_based=True, task_size=16)
    
    class_id = 0
    classes = []
    demo_goals = []
    for trajs in dset.obses:
        for traj in trajs:
            demo_goals.append(torch.as_tensor(traj[-1]))
            classes.append(torch.tensor([class_id]))
        class_id += 1

    demo_goals = torch.stack(demo_goals, 0)
    classes = torch.cat(classes, 0).long()

    # get the embeddings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.to(device)

    dset = TensorDataset(demo_goals.to(device))
    dloader = DataLoader(dset, batch_size=32, num_workers=0)

    embs = []
    agent.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dloader):
            goals = batch[0]
            emb = agent.enc(goals)
            embs.append(emb)

    demo_embs = torch.cat(embs, 0).detach().cpu().numpy()
    classes = classes.detach().cpu().numpy()

    fname_suf = data_path.stem
    if pargs.random:
        fname_suf += '_random'

    print(f'Number of trajectories: {len(demo_embs)}')
    print('Running TSNE ...')
    demo_2d = TSNE(n_components=2).fit_transform(demo_embs)
    s_plt = plt.scatter(demo_2d[:, 0], demo_2d[:, 1], s=5, c=classes) #, cmap=mcolors.ListedColormap(label_colors))
    h,l = s_plt.legend_elements()
    # plt.legend(handles = h, labels=labels)
    plt.savefig(output_dir / f'tsne_demo_embs_{fname_suf}.png' if fname_suf else 'tsne_demo_embs.png')

    plt.close()

    print('Running PCA ...')

    demo_2d = PCA(n_components=2).fit_transform(demo_embs)
    s_plt = plt.scatter(demo_2d[:, 0], demo_2d[:, 1], s=5, c=classes) #, cmap=mcolors.ListedColormap(label_colors))
    h,l = s_plt.legend_elements()
    # plt.legend(handles = h, labels=labels)
    plt.savefig(output_dir / f'pca_demo_embs_{fname_suf}.png' if fname_suf else 'pca_demo_embs.png')

    breakpoint()

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


    # compute trajectory retrieval scores
    print('Computing trajectory retrieval score ...')
    # last k should be 99 since except the index itself there are 99 others
    # k_list = [1, 2, 3, 4, 5, 10]
    k_list = [2]

    tr_score_list = []
    for k in tqdm.tqdm(k_list):
        query_acc_list = []
        for idx in range(len(demo_embs)):
            if pargs.metric == 'euclidean':
                dist = np.sqrt(((demo_embs - demo_embs[idx]) ** 2).sum(-1))
            else:
                norm_embs = np.sqrt((demo_embs**2).sum(-1))
                norm_emb = np.sqrt((demo_embs[idx]**2).sum(-1))
                dist = 1 - (demo_embs @ demo_embs[idx].T) / norm_embs / norm_emb
            dist[idx] = float('inf')
            cand_inds = np.argsort(dist)[:k]
            retrieved_class = classes[cand_inds]
            query_acc = (np.all(retrieved_class == classes[idx], -1)).sum() / k
            query_acc_list.append(query_acc)
        tr_score = np.mean(query_acc_list)
        tr_score_list.append(float(tr_score))
        print(tr_score)

    print(f'{"k":2}, {"tr_score":10}')
    for k, tr_score in zip(k_list, tr_score_list):
        print(f'{k:2d}, {tr_score:10.3f}')

    results = dict(k_list=k_list, tr_scores=tr_score_list)

    # if pargs.metric == 'euclidean':
    #     write_yaml(output_dir / f'tr_results_{fname_suf}.yaml' if fname_suf else 'tr_results.yaml', results)
    # else:
    #     write_yaml(output_dir / f'tr_results_cosine_{fname_suf}.yaml' if fname_suf else 'tr_results_cosine.yaml', results)

if __name__ == '__main__':
    main(_parse_args())