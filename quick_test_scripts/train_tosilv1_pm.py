
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
from osil.eval import evaluate_osil_pm

from osil.debug import register_pdb_hook
register_pdb_hook()

from torch.utils.data import Dataset
import torch
import d4rl; import gym

from osil_gen_data.data_collector import OsilDataCollector

# set the spin colors of an axes
def set_spine_color(ax, color):
    for dir in ['top', 'bottom', 'left', 'right']:
        ax.spines[dir].set_color(color)
        ax.spines[dir].set_linewidth(4)

class PointmassBCDataset(Dataset):

    def __init__(
        self,
        data_path,
        mode='train', # valid / test are also posssible
        seed=0,
    ):
        SPLITS = {'train': (0, 0.8), 'valid': (0.8, 0.9), 'test': (0.9, 1)}

        self.data_path = Path(data_path)
        collector = OsilDataCollector.load(data_path)
        self.raw_data = collector.data

        task_name_list = []
        for task_id in self.raw_data:
            for var_id in self.raw_data[task_id]:
                task_name_list.append((task_id, var_id))
        np.random.seed(seed)
        inds = np.random.permutation(np.arange(len(task_name_list)))

        sratio, eratio = SPLITS[mode]
        s = int(sratio * len(inds))
        e = int(eratio * len(inds))
        self.allowed_ids = set(task_name_list[i] for i in inds[s:e].tolist())
        
        # states, actions, targets = [], [], []
        # for task_id, task in self.raw_data.items():
        #     for var_id, var in task.items():
        #         if (task_id, var_id) not in self.allowed_ids:
        #             continue
        #         for ep in var:
        #             states.append(ep['state'])
        #             actions.append(ep['action'])
        #             targets.append(ep['target'])

        # self.states = np.concatenate(states, 0)
        # self.actions = np.concatenate(actions, 0)
        # self.targets = np.concatenate(targets, 0)
        
    def __len__(self):
        return len(self.allowed_ids)

    def __getitem__(self, idx):
        task_id, var_id = self.allowed_ids[idx]
        episodes = self.raw_data[task_id][var_id]

        c_idx, t_idx = np.random.randint(len(episodes), size=(2,))

        return dict(
            context_s=torch.as_tensor(episodes[c_idx]['state'], dtype=torch.float),
            context_a=torch.as_tensor(episodes[c_idx]['action'], dtype=torch.float),
            target_s=torch.as_tensor(episodes[t_idx]['state'], dtype=torch.float),
            target_a=torch.as_tensor(episodes[t_idx]['action'], dtype=torch.float)
        )

def custom_collate(batch):
    # Note: we will have dynamic batch size for training the MLP part which should be ok? not sure!
    elem = batch[0]
    ret = {}
    for key in elem:
        if key.startswith('context'):
            ret[key] = torch.stack([e[key] for e in batch], 0)
        elif key.startswith('target'):
            ret[key] = torch.cat([e[key].view(-1, elem[key].shape[-1]) for e in batch])
            if 'ptr' not in ret:
                ret['ptr'] = torch.tensor([len(e[key]) for e in batch])

    return ret
    


def get_test_cases(pargs):
    test_dataset = PointmassBCDataset(data_path=pargs.dataset_path, mode='test')
    test_task_ids = test_dataset.allowed_ids
    
    # (states, actions, rst_state)
    for task_id, var_id in test_task_ids:
        episodes = test_dataset.raw_data[task_id][var_id]
        
        test_cases = []
        for ep_id, ep in enumerate(episodes):
            # random design choice: use the next episode to obtain reset
            rst_idx = (ep_id + 1) % len(episodes)
            test_cases.append((ep['state'], ep['action'], episodes[rst_idx]['state'][0]))

    return test_cases

def eval(pargs, agent, output_dir):
    print('Evaluating the agent ...')
    output_dir = Path(output_dir)
    test_cases = get_test_cases(pargs)

    successes = []
    device = agent.device

    example_trajs = []
    for demo_state, demo_action, new_rst_state in test_cases:
        env = gym.make(pargs.env_name)
        
        # set the reset
        pos = new_rst_state[:2]
        vel = new_rst_state[2:]
        env.reset()
        env.set_state(pos, vel)

        # set the target
        goal = demo_state[-1][:2]
        s = new_rst_state
        done = False
        step = 0

        visited_xys = []
        while not done and step < 128:
            # step through the policy
            state_tens = torch.as_tensor(s[None], dtype=torch.float, device=device)
            goal_tens = torch.as_tensor(goal[None], dtype=torch.float, device=device)
            pred_ac = agent(state_tens, goal_tens)
            a = pred_ac.squeeze(0).detach().cpu().numpy()
            
            visited_xys.append(s[:2])
            ns, _, _, _ = env.step(a)

            if np.linalg.norm(ns[:2] - goal) < 0.1:
                done = True
            else:
                s = ns
                step += 1

        example_trajs.append(dict(
            visited_xys=np.stack(visited_xys, 0),
            demo_xy=demo_state[:, :2],
        ))
        successes.append(done)

    write_yaml(output_dir / 'summary.yaml', dict(success_rate=float(np.mean(successes))))
    write_pickle(output_dir / 'example_trajs.pkl', example_trajs)

    print('Plotting examples ...')
    T = 16
    nrows = int(T ** 0.5)
    ncols = -(-T // nrows) # cieling
    plot_path = output_dir / f'examples_{T}.png'

    _, axes = plt.subplots(nrows, ncols, figsize=(15, 8), squeeze=False)
    axes = axes.flatten()
    for idx, traj in enumerate(example_trajs[:T]):
        policy_xy = traj['visited_xys']
        demo_xy = traj['demo_xy']
        axes[idx].plot(policy_xy[:, 0], policy_xy[:, 1], linestyle='-', c='orange', linewidth=5, label='policy')
        axes[idx].plot(demo_xy[:, 0], demo_xy[:, 1], linestyle='-', c='red', linewidth=5, label='demo')
        axes[idx].scatter([demo_xy[-1, 0]], [demo_xy[-1, 1]], s=320, marker='*', c='green', label='goal')
        axes[idx].scatter([policy_xy[-1, 0]], [policy_xy[-1, 1]], s=320, marker='*', c='red', label='policy_end')
        set_spine_color(axes[idx], 'green' if successes[idx] else 'red')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=250)

    # plot failed ones too
    print('Plotting failed examples ...')
    T = max(16, len(successes) - sum(successes))
    nrows = int(T ** 0.5)
    ncols = -(-T // nrows) # cieling
    plot_path = output_dir / f'examples_{T}_failed.png'

    plt.close()
    _, axes = plt.subplots(nrows, ncols, figsize=(15, 8), squeeze=False)
    axes = axes.flatten()

    count = 0
    for idx, traj in enumerate(example_trajs):
        if successes[idx]:
            continue
        elif count == T:
            break
        policy_xy = traj['visited_xys']
        demo_xy = traj['demo_xy']
        axes[count].plot(policy_xy[:, 0], policy_xy[:, 1], linestyle='-', c='orange', linewidth=5, label='policy')
        axes[count].plot(demo_xy[:, 0], demo_xy[:, 1], linestyle='-', c='red', linewidth=5, label='demo')
        axes[count].scatter([demo_xy[-1, 0]], [demo_xy[-1, 1]], s=320, marker='*', c='green', label='goal')
        axes[count].scatter([policy_xy[-1, 0]], [policy_xy[-1, 1]], s=320, marker='*', c='red', label='policy_end')
        count += 1

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=250)

    
    print(f'Evaluating the agent is done, success rate: {float(np.mean(successes))}')

def _parse_args():

    parser = argparse.ArgumentParser()
    # basic common params
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight_decay', '-wc', default=1e-4, type=float)
    parser.add_argument('--hidden_dim', '-hd', default=256, type=int)
    parser.add_argument('--batch_size', '-bs', default=1024, type=int)
    parser.add_argument('--lr', '-lr', default=3e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--env_name', type=str)
    # other params
    parser.add_argument('--frac', '-fr', default=1.0, type=float)
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
    
    # assert pargs.trg_size == 1, 'For BC trg_size should be 1'
    # dset = GCPM(ctx_size=pargs.ctx_size, trg_size=pargs.trg_size, goal_type=pargs.goal_type)
    # train_dataset = Subset(dset, indices=np.arange(len(dset) - pargs.val_dsize))
    # valid_dataset = Subset(dset, indices=np.arange(len(dset) - pargs.val_dsize, len(dset)))

    data_path = pargs.dataset_path
    dset = PointmassBCDataset(data_path=data_path, mode='train')
    train_dataset = Subset(dset, indices=np.arange(int(len(dset)*pargs.frac)))
    valid_dataset = PointmassBCDataset(data_path=data_path, mode='valid')
    
    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=40)
    vloader = DataLoader(valid_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=16)
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
        resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
        # val_check_interval=500, # check val set every x steps
        # val_check_interval=1000,
        callbacks=[ckpt_callback],
    )

    eval_output_dir = ''
    if train:
        trainer.fit(agent, train_dataloaders=[tloader], val_dataloaders=[vloader])
        eval_output_dir = Path(trainer.checkpoint_callback.best_model_path).parent
    else:
        eval_output_dir = pargs.eval_path
        if ckpt and not pargs.eval_path:
            eval_output_dir = str(Path(ckpt).parent.resolve())
            warnings.warn(f'Checkpoint is given for evaluation, but evaluation path is not determined. Using {eval_output_dir} by default')
    

    eval(pargs, agent, eval_output_dir)



if __name__ == '__main__':
    main(_parse_args())
