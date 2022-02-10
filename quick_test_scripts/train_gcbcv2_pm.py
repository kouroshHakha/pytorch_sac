
from gc import callbacks
import warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset

from utils import write_pickle, write_yaml

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from osil.data import GCPM, TestOsilPM
from osil.nets import GCBCv2
from osil.utils import ParamDict
from osil.eval import EvaluatorPointMazeBase

from osil.debug import register_pdb_hook
register_pdb_hook()

from torch.utils.data import Dataset
import torch
import d4rl; import gym

from osil_gen_data.data_collector import OsilDataCollector

class PointmassBCDataset(Dataset):

    def __init__(
        self,
        data_path,
        mode='train', # valid / test are also posssible
        seed=0,
        nshots_per_task=-1, # sets the number of examples per task variation, -1 means to use the max
        goal_dim=-1,
    ):
        SPLITS = {'train': (0, 0.8), 'valid': (0.8, 0.9), 'test': (0.9, 1)}

        self.data_path = Path(data_path)
        collector = OsilDataCollector.load(data_path)
        self.raw_data = collector.data
        self.nshots_per_task = nshots_per_task

        # fake goal dim
        self.goal_dim = goal_dim

        task_name_list = []
        for task_id in self.raw_data:
            for var_id in self.raw_data[task_id]:
                task_name_list.append((task_id, var_id))
        np.random.seed(seed+10)
        inds = np.random.permutation(np.arange(len(task_name_list)))

        sratio, eratio = SPLITS[mode]
        s = int(sratio * len(inds))
        e = int(eratio * len(inds))
        self.allowed_ids = [task_name_list[int(i)] for i in inds[s:e]]
        foo = [task_name_list[i] for i in inds]
        
        states, actions, targets = [], [], []
        for task_id, var_id in self.allowed_ids:
            episodes = self.raw_data[task_id][var_id]
            max_ep_idx = len(episodes) if nshots_per_task == -1 else nshots_per_task
            if max_ep_idx < 2:
                print(f'You need at least two examples per task to make an osil statement, using two instead.')
                max_ep_idx = 2
            if max_ep_idx > len(episodes):
                print(f'Using fewer than {max_ep_idx}, since there are not too many samples for task {task_id}_{var_id}.')
            for ep in episodes[:max_ep_idx]:
                states.append(ep['state'])
                actions.append(ep['action'])
                # targets.append(ep['target'])
                targets.append(np.tile(ep['state'][-1, :2], (len(ep['state']), 1)))

        self.states = np.concatenate(states, 0)
        self.actions = np.concatenate(actions, 0)
        self.targets = np.concatenate(targets, 0)
        
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.states[idx], dtype=torch.float)
        g = torch.as_tensor(self.targets[idx], dtype=torch.float)
        y = torch.as_tensor(self.actions[idx], dtype=torch.float)

        if self.goal_dim != -1:
            nrepeats = self.goal_dim // g.shape[-1]
            g = torch.tile(g, (nrepeats, ))

        return x, g, y

class Evaluator(EvaluatorPointMazeBase):

    def _get_goal(self, demo_state, demo_action):
        g = demo_state[-1, :2]
        if self.conf.gd != -1:
            nrepeats = self.conf.gd // g.shape[-1]
            g = np.tile(g, (nrepeats, ))

        return g

    def _get_action(self, state, goal):
        device = self.agent.device
        state_tens = torch.as_tensor(state[None], dtype=torch.float, device=device)
        goal_tens = torch.as_tensor(goal[None], dtype=torch.float, device=device)
        pred_ac = self.agent(state_tens, goal_tens)
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
    parser.add_argument('--num_shots', default=-1, type=int, 
                        help='number of shots per each task variation \
                            (-1 means max number of shots available in the dataset)')
    parser.add_argument('--gd', '-gd', default=-1, type=int)
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
    exp_name = f'gcbcv2_pm'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    data_path = pargs.dataset_path
    train_dataset = PointmassBCDataset(data_path=data_path, mode='train', nshots_per_task=pargs.num_shots, goal_dim=pargs.gd)
    valid_dataset = PointmassBCDataset(data_path=data_path, mode='valid', goal_dim=pargs.gd)
    test_dataset = PointmassBCDataset(data_path=data_path, mode='test', goal_dim=pargs.gd)
    breakpoint()

    # ###### visualize the data
    # tbatch_all = next(iter(DataLoader(train_dataset, shuffle=True, batch_size=len(train_dataset), num_workers=0)))
    # vbatch_all = next(iter(DataLoader(valid_dataset, shuffle=True, batch_size=len(valid_dataset), num_workers=0)))
    # t_state, t_goal, t_action = tbatch_all
    # v_state, v_goal, v_action = vbatch_all

    # # plot the distribution of states and compare it to validation
    # plt.close()
    # plt.scatter(t_state[:, 0], t_state[:, 1], color='blue', alpha=0.5, s=5, label='train')
    # plt.scatter(v_state[:, 0], v_state[:, 1], color='orange', alpha=0.5, s=5, label='valid')
    # plt.xlim(0, 4)
    # plt.ylim(0, 6)
    # plt.legend()
    # plt.savefig('debug_gcbcv2_xy_train_valid.png')

    # from utils import read_pickle
    # example_trajs = read_pickle('wandb_logs/osil/5p1rsl4m/checkpoints/example_trajs.pkl')
    # rollouts_100 = np.concatenate([exmp['visited_xys'] for exmp in example_trajs[:100]])
    # rollouts_200 = np.concatenate([exmp['visited_xys'] for exmp in example_trajs[100:]])
    # plt.close()
    # plt.scatter(rollouts_100[:, 0], rollouts_100[:, 1], color='blue', alpha=0.5, s=5, label='tested rollouts first 100')
    # plt.scatter(rollouts_200[:, 0], rollouts_200[:, 1], color='orange', alpha=0.5, s=5, label='tested rollouts second 100')
    # plt.xlim(0, 4)
    # plt.ylim(0, 6)
    # plt.legend()
    # plt.savefig('debug_gcbcv2_xy_tested_rollouts2.png')

    # breakpoint()
    
    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=0)
    vloader = DataLoader(valid_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=0)
    obs, goal, act = train_dataset[0]

    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        obs_dim=obs.shape[-1],
        ac_dim=act.shape[-1],
        goal_dim=goal.shape[-1],
        lr=pargs.lr,
        wd=pargs.weight_decay,
    )

    ######## check model's input to output dependency
    args_var = vars(pargs)
    ckpt = args_var.get('ckpt', '')
    resume = args_var.get('resume', False)
    
    agent = GCBCv2.load_from_checkpoint(ckpt) if ckpt else GCBCv2(config)
    agent = agent.to(device=pargs.device)

    train = (ckpt and resume) or not ckpt
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
        trainer.fit(agent, train_dataloaders=[tloader], val_dataloaders=[vloader])
        eval_output_dir = Path(trainer.checkpoint_callback.best_model_path).parent
        agent = agent.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        eval_output_dir = pargs.eval_path
        if ckpt and not pargs.eval_path:
            eval_output_dir = str(Path(ckpt).parent.resolve())
            warnings.warn(f'Checkpoint is given for evaluation, but evaluation path is not determined. Using {eval_output_dir} by default')
    

    evaluator = Evaluator(pargs, agent, eval_output_dir, test_dataset)
    evaluator.eval()



if __name__ == '__main__':
    main(_parse_args())
