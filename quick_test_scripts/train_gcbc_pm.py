
from gc import callbacks
import warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from osil.data import GCPM, TestOsilPM
from osil.nets import GCBC
from osil.utils import ParamDict
from osil.eval import evaluate_osil_pm

from osil.debug import register_pdb_hook
register_pdb_hook()

from torch.utils.data import Dataset
import torch
import d4rl; import gym

class PointmassBCDataset(Dataset):

    def __init__(self, name='maze2d-open-v0', block_size=64, goal_type='only_last'):
        
        self.block_size = block_size
        
        env = gym.make(name)
        self.data = env.get_dataset()
        self.obses = self.data['observations']
  
        self.acs = self.data['actions']
        self.goal_type = goal_type
    
    
    def __len__(self):
        return len(self.obses) - 2*self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        obses = self.obses[idx:idx + 2*self.block_size]
        acs = self.acs[idx:idx + 2*self.block_size]
        
        # update 1: x, y are subset of larger obses but goal is still defined based on x
        start_idx = torch.randint(high=self.block_size, size=(1, ))
        x = torch.tensor(obses[start_idx: start_idx + self.block_size], dtype=torch.float)
        y = torch.tensor(acs[start_idx: start_idx + self.block_size], dtype=torch.float)

        # update 2: x, y are subset of larger but goal is defined based on that large trajectory
        if self.goal_type == 'only_last':
            # x, y location of the last step
            # goal = x[-1:, :2]
            goal = torch.as_tensor(obses[-1:, :2], dtype=torch.float)
        elif self.goal_type == 'zero':
            goal = torch.zeros(1, 2).to(x)
        elif self.goal_type == 'last+start':
            # goal = torch.cat([x[0, :2], x[-1, :2]], -1)[None]
            goal_np = np.concatenate([obses[0, :2], obses[-1, :2]], -1)[None]
            goal = torch.as_tensor(goal_np, dtype=torch.float)
        else:
            raise ValueError('unknown goal type')


        return x, goal, y
    

def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--weight_decay', '-wc', default=1e-4, type=float)
    parser.add_argument('--hidden_dim', '-hd', default=256, type=int)
    parser.add_argument('--batch_size', '-bs', default=8192, type=int)
    # parser.add_argument('--val_dsize', '-vs', default=1000000, type=int)
    parser.add_argument('--val_dsize', '-vs', default=1000, type=int)
    # parser.add_argument('--lr', '-lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--ctx_size', default=64, type=int) # how far in the future do u want the goal to be at?
    parser.add_argument('--trg_size', default=1, type=int) # decoding context
    parser.add_argument('--goal_type', default='only_last', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_path', type=str)
    
    parser.add_argument('--use_wandb', '-wb', action='store_true')
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')
    # parser.add_argument('--num_eval_episodes', default=10, type=int)
    # parser.add_argument('--num_trajs', '-ntraj', default=100, type=int)

    return parser.parse_args()


def main(pargs):
    exp_name = f'gcbc_pm'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    # assert pargs.trg_size == 1, 'For BC trg_size should be 1'
    # dset = GCPM(ctx_size=pargs.ctx_size, trg_size=pargs.trg_size, goal_type=pargs.goal_type)
    # train_dataset = Subset(dset, indices=np.arange(len(dset) - pargs.val_dsize))
    # valid_dataset = Subset(dset, indices=np.arange(len(dset) - pargs.val_dsize, len(dset)))


    dset = PointmassBCDataset(block_size=64, goal_type=pargs.goal_type)
    train_dataset = Subset(dset, indices=np.arange(len(dset) - pargs.val_dsize))
    valid_dataset = Subset(dset, indices=np.arange(len(dset) - pargs.val_dsize, len(dset)))
    
    tloader = DataLoader(train_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=40)
    vloader = DataLoader(valid_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=16)
    obs, goal, act = train_dataset[0]
    # obs, act = train_dataset[0]
    # goal_dim = 8

    # # plot some example datasets
    # _, axes = plt.subplots(1,1, squeeze=False)
    # axes = axes.flatten()
    
    # states, goals, acts = next(iter(tloader))
    # for i in range(1):
    #     axes[i].plot(states[i, :, 0], states[i, :, 1], color='b', linestyle='--')
    #     axes[i].scatter([goals[i, 0].item()], [goals[i, 1].item()], marker='*', color='green')
    #     axes[i].scatter(states[i, 0, 0], states[i, 0, 1], marker='.', color='r')
    #     axes[i].scatter(states[i, -1, 0], states[i, -1, 1], marker='.', color='g')
    
    # plt.tight_layout()
    # plt.savefig('gcdt_data.png')
    # breakpoint()

    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        obs_shape=(obs.shape[-1],),
        # obs_shape=obs.shape[1:],
        ac_dim=act.shape[-1],
        goal_dim=goal.shape[-1],
        # goal_dim=goal_dim,
        lr=1e-4,
        goal_type=pargs.goal_type,
        wd=pargs.weight_decay,
    )

    ######## check model's input to output dependency
    # import torch
    # input_tokens = torch.randn(1, 10, pargs.hidden_dim)
    # outputs = agent.transformer(inputs_embeds=input_tokens, output_attentions=True)
    # attn = outputs['attentions'][0][0,0].detach().numpy()
    # plt.imshow(attn)
    # plt.show()
    args_var = vars(pargs)
    ckpt = args_var.get('ckpt', '')
    resume = args_var.get('resume', False)
    
    agent = GCBC.load_from_checkpoint(ckpt) if ckpt else GCBC(config)
    agent = agent.to(device=pargs.device)

    if pargs.use_wandb:
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
    train = (ckpt and resume) or not ckpt
    if train:
        trainer.fit(agent, train_dataloaders=[tloader], val_dataloaders=[vloader])
        eval_output_dir = Path(trainer.checkpoint_callback.best_model_path).parent
    else:
        eval_output_dir = pargs.eval_path
        if ckpt and not pargs.eval_path:
            eval_output_dir = str(Path(ckpt).parent.resolve())
            warnings.warn(f'Checkpoint is given for evaluation, but evaluation path is not determined. Using {eval_output_dir} by default')
    
    print('Evaluating the agent ...')
    pl.seed_everything(0)
    # test dataset uses twice the block size at the moment to not introduce 
    # any confounding factor fro early timesteps in the epiode that we want to imitate. 
    test_dataset = TestOsilPM(traj_size=128)
    evaluate_osil_pm(agent, test_dataset, eval_output_dir=eval_output_dir, render_examples=True)
    print('Evaluating the agent is done.')


if __name__ == '__main__':
    main(_parse_args())
