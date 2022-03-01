
from gc import callbacks
from multiprocessing.sharedctypes import Value
import warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.nn as nn

from utils import write_pickle, write_yaml

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from osil.nets import GCBC, GCBCv2
from osil.utils import ParamDict
from osil.eval import EvaluatorReacher2D_GCBC, EvaluationCallback

from osil.debug import register_pdb_hook
register_pdb_hook()

from torch.utils.data import Dataset
import torch
import d4rl; import gym
import envs

import tqdm

from osil_gen_data.data_collector import OsilDataCollector
from osil.data import SPLITS


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
    parser.add_argument('--task_size', default=-1, type=int)

    parser.add_argument('--gd', '-gd', default=-1, type=int)
    # checkpoint resuming and testing
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--eval_every_nsteps', default=1000, type=int)
    # wandb
    parser.add_argument('--use_wandb', '-wb', action='store_true')
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')

    return parser.parse_args()


class GCBCDataset(Dataset):

    def __init__(
        self,
        data_path,
        seed=0,
        nshots_per_task=-1, # sets the number of examples per task variation, -1 means to use the max
        task_size=-1,
    ):

        self.data_path = Path(data_path)
        collector = OsilDataCollector.load(data_path)
        self.raw_data = collector.data
        self.nshots_per_task = nshots_per_task

        # create this allowed ids to make it compatible with previously implemented evaluation functions
        class_to_task_map = {}
        class_id = 0
        for task_id in self.raw_data:
            task_size = len(self.raw_data[task_id]) if task_size == -1 else task_size
            for var_id in sorted(self.raw_data[task_id].keys()):
                if var_id < task_size:
                    class_to_task_map[class_id] = (task_id, var_id)
                    class_id += 1
        self.allowed_ids = list(class_to_task_map.values())

        states, actions, targets = [], [], []
        for task_id, var_id in self.allowed_ids:
            episodes = self.raw_data[task_id][var_id]

            if len(episodes) < 2:
                print(f'Skipping task {task_id}-{var_id}, because of insufficient examples')
                continue

            max_ep_idx = len(episodes) if nshots_per_task == -1 else nshots_per_task
            if max_ep_idx < 2:
                print(f'You need at least two examples per task to make an osil statement, using two instead.')
                max_ep_idx = 2
            if max_ep_idx > len(episodes):
                print(f'Using fewer than {max_ep_idx}, since there are not too many samples for task {task_id}_{var_id}.')
            for ep in episodes[:max_ep_idx]:
                # select the length of each episode based on how large actions are
                # the last large action set the end of an episode for imitation learning
                # cond = (ep['action']**2).sum(-1)**0.5 > 0.05
                # if cond.any():
                    # add n steps after that so it learns to stay there for at least n more steps
                    # last_step = min(np.where(cond)[0][-1] + 20, len(cond) - 1)
                    # last_step = len(cond) - 1
                # else:
                    # continue
                # states.append(ep['state'][:last_step + 1])
                # actions.append(ep['action'][:last_step + 1])
                states.append(ep['state'])
                actions.append(ep['action'])
                target = np.tile(ep['target'], (len(states[-1]), 1))
                # location of eef
                # target = np.tile(ep['state'][last_step, 4:6], (len(states[-1]), 1))
                # target = np.tile(ep['state'][-1, 4:6], (len(states[-1]), 1))
                targets.append(target)

        self.states = np.concatenate(states, 0)
        self.actions = np.concatenate(actions, 0)
        self.targets = np.concatenate(targets, 0)
        assert self.states.shape[0] == self.actions.shape[0] == self.targets.shape[0]
        
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.states[idx], dtype=torch.float)
        g = torch.as_tensor(self.targets[idx], dtype=torch.float)
        y = torch.as_tensor(self.actions[idx], dtype=torch.float)

        return x, g, y

def main(pargs):
    exp_name = f'gcbcv2_reacher2d'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    train_path = 'reacher_2d_train_v2'
    test_path = 'reacher_2d_test_v2'
    train_dataset = GCBCDataset(data_path=train_path, nshots_per_task=pargs.num_shots, task_size=pargs.task_size)
    test_dataset = GCBCDataset(data_path=test_path)

    print('training set size:', len(train_dataset))
    print('test set size:',     len(test_dataset))


    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=0)
    vloader = DataLoader(test_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=0)
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
    evaluator_cls = EvaluatorReacher2D_GCBC

    
    # agent = Reacher2DModel.load_from_checkpoint(ckpt) if ckpt else Reacher2DModel(config)
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

    # setting up the call backs
    ckpt_callback_valid = ModelCheckpoint(
                monitor='valid_loss',
                filename='cgl-{step}-{valid_loss:.4f}-{epoch:02d}',
                save_last=True,
                save_on_train_epoch_end=True,
                mode='min',
            )

    ckpt_callback_train = ModelCheckpoint(
            monitor='train_loss_epoch',
            filename='cgl-{step}-{train_loss_epoch:.4f}-{epoch:02d}',
            save_last=True,
            save_on_train_epoch_end=True,
            mode='min',
        )

    # evaluation callbacks
    train_evaluator = evaluator_cls(
        ParamDict(max_eval_episodes=100, env_name='Reacher2D-v1'),
        train_dataset, mode='train'
    )
    eval_ckpt_train = EvaluationCallback(
        train_evaluator, 
        eval_every_n_updates=pargs.eval_every_nsteps, 
        dirpath=ckpt_callback_valid.dirpath,
    )

    test_evaluator = evaluator_cls(
        ParamDict(max_eval_episodes=100, env_name='Reacher2DTest-v1'),
        test_dataset, mode='test',
    )
    eval_ckpt_test = EvaluationCallback(test_evaluator, 
        eval_every_n_updates=pargs.eval_every_nsteps,
        dirpath=ckpt_callback_valid.dirpath,
    )


    trainer = pl.Trainer(
        max_epochs=pargs.max_epochs,
        max_steps=pargs.max_steps,
        resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
        callbacks=[ckpt_callback_valid, ckpt_callback_train, eval_ckpt_train, eval_ckpt_test],
    )

    eval_output_dir = ''
    if train:
        trainer.fit(agent, train_dataloaders=[tloader], val_dataloaders=[vloader])
        # eval_output_dir = Path(ckpt_callback_valid.best_model_path).parent
        # print(f'Evaluating with {ckpt_callback_valid.best_model_path} ...')
        # agent = agent.load_from_checkpoint(ckpt_callback_valid.best_model_path)
        eval_output_dir = Path(eval_ckpt_test.best_model_path).parent
        print(f'Evaluating with {eval_ckpt_test.best_model_path} ...')
        agent = agent.load_from_checkpoint(eval_ckpt_test.best_model_path)
    else:
        eval_output_dir = pargs.eval_path
        if ckpt and not pargs.eval_path:
            eval_output_dir = str(Path(ckpt).parent.resolve())
            warnings.warn(f'Checkpoint is given for evaluation, but evaluation path is not determined. Using {eval_output_dir} by default')
    

    conf = ParamDict(max_eval_episodes=100, max_render=10, verbose=True, env_name='Reacher2D-v1')
    evaluator_cls(conf, train_dataset, eval_output_dir, mode='train').eval(agent)
    conf = ParamDict(max_eval_episodes=100, max_render=10, verbose=True, env_name='Reacher2DTest-v1')
    evaluator_cls(conf, test_dataset, eval_output_dir, mode='test').eval(agent)

if __name__ == '__main__':
    main(_parse_args())
