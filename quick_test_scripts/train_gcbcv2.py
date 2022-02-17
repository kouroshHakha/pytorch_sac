
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

from osil.nets import GCBCv2
from osil.utils import ParamDict
from osil.eval import EvaluatorPointMazeBase, EvaluatorReacherSawyer

from osil.debug import register_pdb_hook
register_pdb_hook()

from torch.utils.data import Dataset
import torch
import d4rl; import gym

from osil_gen_data.data_collector import OsilDataCollector



class GCBCDataset(Dataset):

    def __init__(
        self,
        data_path,
        mode='train', # valid / test are also posssible
        seed=0,
        nshots_per_task=-1, # sets the number of examples per task variation, -1 means to use the max
        env_name='maze2d-open-v0'
    ):
                # to enable backward compatible comparision with the other experiment
        SPLITS = {
            'reacher_7dof-v1': {
                'train': np.arange(12, 64).tolist(),
                'valid': [6, 1, 5, 8, 0, 11],
                'test': [4, 10, 3, 9, 7, 2],
            }, 
            'maze2d-open-v0': {
                'train': [3, 7, 12, 6, 8, 2, 10, 5, 11, 14, 1, 0], 
                'valid': [4], 
                'test': [13, 9],
            }
        }
        self.splits = SPLITS[env_name]
        # SPLITS = {'train': (0, 0.8), 'valid': (0.8, 0.9), 'test': (0.9, 1)}

        self.data_path = Path(data_path)
        collector = OsilDataCollector.load(data_path)
        self.raw_data = collector.data
        self.nshots_per_task = nshots_per_task

        # create this allowed ids to make it compatible with previously implemented evaluation functions
        class_to_task_map = {}
        class_id = 0
        for task_id in self.raw_data:
            for var_id in self.raw_data[task_id]:
                class_to_task_map[class_id] = (task_id, var_id)
                class_id += 1
        # task_to_class_map = {v: k for k, v in class_to_task_map.items()}
        self.allowed_ids = [class_to_task_map[i] for i in self.splits[mode]]

        # np.random.seed(seed+10)
        # inds = np.random.permutation(np.arange(len(task_name_list)))

        # sratio, eratio = SPLITS[mode]
        # s = int(sratio * len(inds))
        # e = int(eratio * len(inds))
        # self.allowed_ids = [task_name_list[int(i)] for i in inds[s:e]]
        
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
                # target = ep['target']
                if env_name == 'maze2d-open-v0':
                    # the xy location of the last state
                    target = np.tile(ep['state'][-1, :2], (len(ep['state']), 1))
                elif env_name == 'reacher_7dof-v1':
                    # the 3d eef at the last state
                    target = np.tile(ep['state'][-1][-3:], (len(ep['state']), 1))

                # target = OneHotEncoder(categories=[np.arange(15)]).fit_transform([[task_to_class_map[(task_id, var_id)]]])
                # target = np.tile(target.toarray(), (len(ep['state']), 1))
                targets.append(target)

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


def get_action(state, goal, agent):
    device = agent.device
    state_tens = torch.as_tensor(state[None], dtype=torch.float, device=device)
    goal_tens = torch.as_tensor(goal[None], dtype=torch.float, device=device)
    pred_ac = agent(state_tens, goal_tens)
    a = pred_ac.squeeze(0).detach().cpu().numpy()
    return a

class EvaluatorPM(EvaluatorPointMazeBase):
    def _get_goal(self, demo_state, demo_action):
        g = demo_state[-1, :2]
        if self.conf.gd != -1:
            nrepeats = self.conf.gd // g.shape[-1]
            g = np.tile(g, (nrepeats, ))

        return g

    def _get_action(self, state, goal):
        return get_action(state, goal, self.agent)

class EvaluatorReacher(EvaluatorReacherSawyer):

    def _get_goal(self, demo_state, demo_action):
        g = demo_state[-1, -3:]
        if self.conf.gd != -1:
            nrepeats = self.conf.gd // g.shape[-1]
            g = np.tile(g, (nrepeats, ))

        return g

    def _get_action(self, state, goal):
        return get_action(state, goal, self.agent)


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
    exp_name = f'gcbcv2_{pargs.env_name}'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    data_path = pargs.dataset_path
    train_dataset = GCBCDataset(data_path=data_path, mode='train', nshots_per_task=pargs.num_shots, env_name=pargs.env_name)
    valid_dataset = GCBCDataset(data_path=data_path, mode='valid', env_name=pargs.env_name)
    test_dataset  = GCBCDataset(data_path=data_path, mode='test', env_name=pargs.env_name)

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
    

    if pargs.env_name.startswith('maze2d'):
        evaluator_cls = EvaluatorPM
    elif pargs.env_name.startswith('reacher'):
        evaluator_cls = EvaluatorReacher
    evaluator = evaluator_cls(pargs, agent, eval_output_dir, test_dataset)
    evaluator.eval()

if __name__ == '__main__':
    main(_parse_args())
