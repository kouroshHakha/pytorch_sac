
import warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
import tqdm

from torch.utils.data import DataLoader, TensorDataset

from utils import read_pickle, write_pickle, write_yaml

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from osil.nets import GCBCv2, TOsilv1, TOsilSemisupervised
from osil.utils import ParamDict
from osil.eval import EvaluatorPointMazeBase
from osil.data import collate_fn_for_supervised_osil, OsilPairedDataset


from osil.debug import register_pdb_hook
register_pdb_hook()

from torch.utils.data import Dataset
import torch
import d4rl; import gym

from osil_gen_data.data_collector import OsilDataCollector


"""
# at decoder level we will have variable batch size, but that's ok
# for probing: context_cls_id, target_cls_id
# reconstruct target trajectory based on context_emb
dict(
    context_s=..., # sample from its random nearest neighbours
    context_a=...,
    context_class_id=...,
    context_emb=...,
    target_s=..., # index from trajectory
    target_a=...,
    target_class_id=...,
    target_emb=...,
)

we can plot estimate of acc for picking the correct context_s, context_a as a function of epoch ...
"""

class PointmassEmbBCDataset(Dataset):

    def __init__(
        self,
        data_path,
        embs,
        states,
        actions,
        classes,
        mode='train', # valid / test are also posssible
        seed=0,
        goal_is_self_embedding=True, # goal is the self trajectory embedding otherwise its random neighbor embedding
        metric='euclidean',
        k_neighbor=100,
    ):  
        # to enable backward compatible comparision with the other experiment
        SPLITS = {
            'train': [3, 7, 12, 6, 8, 2, 10, 5, 11, 14, 1, 0], 
            'valid': [4], 
            'test': [13, 9],
        }

        self.data_path = Path(data_path)
        collector = OsilDataCollector.load(data_path)
        self.raw_data = collector.data
        self.goal_is_self_embedding = goal_is_self_embedding

        # create this allowed ids to make it compatible with previously implemented evaluation functions
        class_to_task_map = {}
        class_id = 0
        for task_id in self.raw_data:
            for var_id in self.raw_data[task_id]:
                class_to_task_map[class_id] = (task_id, var_id)
                class_id += 1
        # task_to_class_map = {v: k for k, v in class_to_task_map.items()}
        self.allowed_ids = [class_to_task_map[i] for i in SPLITS[mode]]
        
        states_flat, actions_flat, target_flat, class_flat = [], [], [], []
        self.nn_s, self.nn_a = [], []

        self.class_ids = []
        self.states = []
        self.actions = []
        self.embs = []

        # nearest neighbors per each trajectory
        self.nn_s = []
        self.nn_a = []
        self.nn_class_ids = []
        self.nn_embs = []


        for idx, (emb, state, action, class_id) in enumerate(zip(embs, states, actions, classes)):
            if class_id not in SPLITS[mode]:
                continue

            self.states.append(state)
            self.actions.append(action)
            self.class_ids.append(class_id)
            self.embs.append(emb)

            if self.goal_is_self_embedding:
                self.nn_s.append([state])
                self.nn_a.append([action])
                self.nn_class_ids.append([class_id])
                self.nn_embs.append([emb])
            else:
                if metric == 'euclidean':
                    dist = np.sqrt(((embs - embs[idx]) ** 2).sum(-1))
                else:
                    norm_embs = np.sqrt((embs**2).sum(-1))
                    norm_emb = np.sqrt((emb**2).sum(-1))
                    dist = (embs @ embs[idx].T) / norm_embs / norm_emb
                dist[idx] = float('inf')
                cand_inds = np.argsort(dist)[:k_neighbor]

                c_s, c_a, c_cls_ids, c_embs = [], [], [], []
                for k in cand_inds:
                    c_s.append(states[k])
                    c_a.append(actions[k])
                    c_cls_ids.append(classes[k])
                    c_embs.append(embs[k])
                
                self.nn_s.append(c_s)        
                self.nn_a.append(c_a)        
                self.nn_class_ids.append(c_cls_ids) 
                self.nn_embs.append(c_embs) 

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        t_s = torch.as_tensor(self.states[idx], dtype=torch.float)
        t_a = torch.as_tensor(self.actions[idx], dtype=torch.float)
        t_emb = torch.as_tensor(self.embs[idx], dtype=torch.float)
        t_class_id = torch.as_tensor(self.class_ids[idx], dtype=torch.long)

        num_neis = len(self.nn_s[idx])
        rand_nei_idx = np.random.randint(num_neis)

        c_s = torch.as_tensor(self.nn_s[idx][rand_nei_idx], dtype=torch.float)
        c_a = torch.as_tensor(self.nn_a[idx][rand_nei_idx], dtype=torch.float)
        c_emb = torch.as_tensor(self.nn_embs[idx][rand_nei_idx], dtype=torch.float)
        c_class_id = torch.as_tensor(self.nn_class_ids[idx][rand_nei_idx], dtype=torch.long)

        # reconstruct target trajectory based on context_emb
        return dict(
            context_s=c_s, # sample from its random nearest neighbours
            context_a=c_a,
            context_class_id=c_class_id,
            context_emb=c_emb,
            target_s=t_s, # index from trajectory
            target_a=t_a,
            target_class_id=t_class_id,
            target_emb=t_emb,
        )

class GCBCDatasetWrapper(Dataset):

    def __init__(self, emb_dataset) -> None:
        super().__init__()
        self.emb_dataset = emb_dataset

    def __len__(self):
        return len(self.emb_dataset)
    
    def __getitem__(self, index):
        sample = self.emb_dataset[index]
        # s, g, a
        return sample['target_s'], sample['target_emb'], sample['target_a']

    def __getattr__(self, attribute_name):
        if attribute_name not in self.__dict__:
            return getattr(self.emb_dataset, attribute_name)
        return self.__dict__[attribute_name]


def collate_fn(batch):
    STATE = 0
    GOAL = 1
    ACTION = 2
    s = torch.cat([b[STATE] for b in batch], 0)
    a = torch.cat([b[ACTION] for b in batch], 0)
    g = torch.cat([b[GOAL].tile(b[STATE].shape[0], 1) for b in batch])
    return s, g, a

class Evaluator(EvaluatorPointMazeBase):

    def _get_goal(self, demo_state, demo_action):
        episode = [dict(
            context_s=torch.as_tensor(demo_state, dtype=torch.float),
            context_a=torch.as_tensor(demo_action, dtype=torch.float),
        )]
        episode = collate_fn_for_supervised_osil(episode)
        device = self.agent['encoder'].device
        with torch.no_grad():
            c_s = episode['context_s'].to(device)
            c_a = episode['context_a'].to(device)
            c_m = episode['attention_mask'].to(device)

            emb = self.agent['encoder'].get_task_emb(c_s, c_a, c_m)[0]
        return emb.detach().cpu().numpy()

    def _get_action(self, state, goal):
        decoder = self.agent['decoder']
        device = decoder.device
        state_tens = torch.as_tensor(state[None], dtype=torch.float, device=device)
        goal_tens = torch.as_tensor(goal[None], dtype=torch.float, device=device)
        pred_ac = decoder(state_tens, goal_tens)
        a = pred_ac.squeeze(0).detach().cpu().numpy()
        return a


def _parse_args():

    parser = argparse.ArgumentParser()
    # basic common params
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight_decay', '-wc', default=1e-4, type=float)
    parser.add_argument('--hidden_dim', '-hd', default=256, type=int)
    parser.add_argument('--batch_size', '-bs', default=1024, type=int)
    parser.add_argument('--lr', '-lr', default=3e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--env_name', type=str)
    # other params
    # parser.add_argument('--num_shots', default=-1, type=int, 
    #                     help='number of shots per each task variation \
    #                         (-1 means max number of shots available in the dataset)')
    # parser.add_argument('--gd', '-gd', default=-1, type=int)
    # encoder checkpoint parameters
    parser.add_argument('--self_emb', action='store_true') # use the self embedding or cross embedding as goal
    parser.add_argument('--k_neighbor', '-kn', default=100, type=int) # the number of trajs for retrieval system 
    parser.add_argument('--encoder_ckpt', type=str)
    parser.add_argument('--model', type=str, choices=['osil', 'semi-osil'])
    # checkpoint resuming and testing
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_path', type=str)
    # wandb
    parser.add_argument('--use_wandb', '-wb', action='store_true')
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')

    return parser.parse_args()


def main(pargs):
    exp_name = f'gcbcv2+emb_pm'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)

    data_path = pargs.dataset_path
    args_var = vars(pargs)
    ckpt = args_var.get('ckpt', '')
    resume = args_var.get('resume', False)
    train = (ckpt and resume) or not ckpt
    
    emb_file = args_var.get('emb_file', Path(pargs.encoder_ckpt).parent / 'dataset_emb_file.pickle')
    compute_traj_embs = train and not emb_file.exists()
    
    if pargs.model == 'osil':
        encoder = TOsilv1.load_from_checkpoint(pargs.encoder_ckpt)
    elif pargs.model == 'semi-osil':
        encoder = TOsilSemisupervised.load_from_checkpoint(pargs.encoder_ckpt)
    else:
        raise ValueError('Unknown model')
    
    if compute_traj_embs:
        print('Computing the trajectory embeddings of the dataset ...')

        # create a flattened dataset with class_ids
        dset = OsilPairedDataset(data_path=data_path, mode='train')
        raw_data = dset.raw_data
        
        class_id = 0
        demo_states, demo_actions, demo_masks = [], [], []
        states, actions = [], []
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
                states += [ep['state'] for ep in raw_data[task_id][var_id]]
                actions += [ep['action'] for ep in raw_data[task_id][var_id]]

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

        # get the embeddings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder.to(device)

        dset = TensorDataset(demo_states.to(device), demo_actions.to(device), demo_masks.to(device))
        dloader = DataLoader(dset, batch_size=32, num_workers=0)

        embs = []
        with torch.no_grad():
            for c_s, c_a, c_m in tqdm.tqdm(dloader):
                emb = encoder.get_task_emb(c_s, c_a, c_m)
                embs.append(emb)

        demo_embs = torch.cat(embs, 0).detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        # # using 2d TSNE projections as embs??
        # demo_2d = TSNE(n_components=2).fit_transform(demo_embs)
        # emb_content = dict(embs=demo_2d, states=states, actions=actions, classes=classes)
        emb_content = dict(embs=demo_embs, states=states, actions=actions, classes=classes)
        write_pickle(emb_file, emb_content)
    else:
        print('Loading pre-computed trajectory embeddings of the dataset ...')
        emb_content = read_pickle(emb_file)
    
    train_emb_dataset = PointmassEmbBCDataset(data_path, **emb_content, mode='train', goal_is_self_embedding=pargs.self_emb, k_neighbor=pargs.k_neighbor)
    valid_emb_dataset = PointmassEmbBCDataset(data_path, **emb_content, mode='valid', goal_is_self_embedding=pargs.self_emb, k_neighbor=pargs.k_neighbor)

    train_dataset = GCBCDatasetWrapper(train_emb_dataset)
    valid_dataset = GCBCDatasetWrapper(valid_emb_dataset)

    # # calculate the accuracy per number of epoch
    # n_epoch = 200
    # correct_list = []
    # acc_list = []
    # epoch_list = []
    # wrong_samples = []
    # for epoch in range(n_epoch):
    #     for idx in range(len(train_emb_dataset)):
    #         sample = train_emb_dataset[idx]

    #         dist = torch.linalg.norm(sample['context_s'][-1, :2] - sample['target_s'][-1, :2])
    #         success_from_s = bool(dist < 0.2)

    #         success_from_labels = bool(sample['context_class_id'] == sample['target_class_id'])
    #         assert success_from_s == success_from_labels

    #         if not success_from_s:
    #             wrong_samples.append(sample)
    #         correct_list.append(success_from_s)

    #     acc_list.append(np.mean(correct_list))
    #     epoch_list.append(epoch)
    
    # plt.plot(epoch_list, acc_list)
    # plt.savefig('debug_gcbcv2_w_osil_emb_acc_vs_ep.png')
    # breakpoint()

    # ###### visualize the data
    # plt.close()
    # _, axes = plt.subplots(4, 4, figsize=(15, 8), squeeze=False)
    # axes = axes.flatten()
    # for idx, sample in enumerate(wrong_samples[:16]):
    #     s = sample['target_s']
    #     axes[idx].plot(s[:, 0], s[:, 1], color='red', linewidth=5, label='query')
    #     axes[idx].scatter([s[-1, 0]], [s[-1, 1]], color='green', s=320, marker='*', label='q_end')

    #     state_seq = sample['context_s']
    #     axes[idx].plot(state_seq[:, 0], state_seq[:, 1], color='orange', alpha=0.5)
    #     axes[idx].scatter([state_seq[-1, 0]], [state_seq[-1, 1]], color='orange', marker='.', s=50, alpha=0.5)
    #     axes[idx].set_xlim(0, 4)
    #     axes[idx].set_ylim(0, 6)
    
    # plt.legend()
    # plt.tight_layout()    
    # plt.savefig(f'debug_gcbcv2_w_osil_emb_failed_modes.png')

    # ## visualize goals
    # print('Running TSNE ...')
    # points = []
    # colors = []
    # for idx in range(len(train_emb_dataset)):
    #     data = train_emb_dataset[idx]
    #     points.append(data['context_emb'].numpy())
    #     colors.append(int(data['context_class_id']))
    
    # colors = np.array(colors)
    # points = np.stack(points, 0)
    # plt.close()
    # for c in [4, 9, 13]:#set(colors):
    #     x2d = points[colors==c]
    #     plt.scatter(x2d[:, 0], x2d[:, 1], s=5, label=f'class = {c}')
    #     print(c, len(x2d))
    # plt.legend()
    # plt.savefig('debug_gcbcv2_w_osil_emb_neighbors_tsne_gaols.png')

    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=0, collate_fn=collate_fn)
    vloader = DataLoader(valid_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=0, collate_fn=collate_fn)
    obs, goal, act = train_dataset[0]

    assert goal.shape[-1] == 2

    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        obs_dim=obs.shape[-1],
        ac_dim=act.shape[-1],
        goal_dim=goal.shape[-1],
        lr=pargs.lr,
        wd=pargs.weight_decay,
    )


    decoder = GCBCv2.load_from_checkpoint(ckpt) if ckpt else GCBCv2(config)
    decoder = decoder.to(device=pargs.device)

    if pargs.use_wandb and train:
        import wandb
        run_name = exp_name if not pargs.run_name else f'{exp_name}_{pargs.run_name}'
        wandb_run = wandb.init(
            project='osil',
            name=run_name,
            dir='./wandb_logs',
            id=pargs.wandb_id,
            resume='allow',
            config=dict(seed=pargs.seed),
        )
        logger = WandbLogger(experiment=wandb_run, save_dir='./wandb_logs')
    else:
        logger = TensorBoardLogger(save_dir='tb_logs', name=exp_name)

    ckpt_callback = ModelCheckpoint(
                monitor='valid_loss',
                filename='cgl-{step}-{valid_loss:.4f}-{epoch:02d}',
                save_last=True,
                save_on_train_epoch_end=True,
                mode='min',
            )

    trainer = pl.Trainer(
        max_epochs=pargs.max_epochs,
        max_steps=pargs.max_steps,
        resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
        callbacks=[ckpt_callback],
    )

    eval_output_dir = ''
    if train:
        trainer.fit(decoder, train_dataloaders=[tloader], val_dataloaders=[vloader])
        eval_output_dir = Path(trainer.checkpoint_callback.best_model_path).parent
        decoder = decoder.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        eval_output_dir = pargs.eval_path
        if ckpt and not pargs.eval_path:
            eval_output_dir = str(Path(ckpt).parent.resolve())
            warnings.warn(f'Checkpoint is given for evaluation, but evaluation path is not determined. Using {eval_output_dir} by default')
    
    test_dataset  = GCBCDatasetWrapper(PointmassEmbBCDataset(data_path, **emb_content, mode='test'))
    evaluator = Evaluator(pargs, {'decoder': decoder, 'encoder': encoder}, eval_output_dir, test_dataset)
    evaluator.eval()



if __name__ == '__main__':
    main(_parse_args())
