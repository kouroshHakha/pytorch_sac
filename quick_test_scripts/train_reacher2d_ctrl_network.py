
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.nn as nn

from utils import stack_frames, write_pickle, write_yaml, gif_to_tensor_image

print(f'Workspace: {Path.cwd()}')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from osil.nets import  BaseLightningModule, MLP, EncoderWithBnorm
from osil.utils import ParamDict
from osil.eval import EvaluationCallback
from osil.evaluators.reacher2d import EvaluatorReacher2D_GCBC_Img

from osil.datasets.reacher2d.gcbc import Reacher2DGCBCDataset

from osil.debug import register_pdb_hook
register_pdb_hook()

import torch
import d4rl; import gym
import envs

import tqdm
import einops


class Custom_EvaluatorReacher2D_GCBC_Img(EvaluatorReacher2D_GCBC_Img):

    def _update_test_cases(self, test_dataset, test_cases):
        for idx in range(len(test_cases)):
            target_eef = test_cases[idx]['target_s'][-1, 4:6]
            test_cases[idx].update(goal=target_eef)
        return test_cases

class PredNN(BaseLightningModule):
    
    def _build_network(self):
        self.hidden_dim = 256
        self.img_enc = 32
        self.n_stack = 4
        self.conf.obs_shape = (3, 64, 64)
        self.enc = EncoderWithBnorm(obs_shape=self.conf.obs_shape, h_dim=self.img_enc)
        self.eef_network = MLP(
            in_channel=2 + self.n_stack * self.img_enc, 
            out_channel=2,
            hidden_dim=self.hidden_dim , 
            n_layers=3
        )


    def ff(self, batch, compute_loss=True):
        
        obs = batch['obs']
        B = obs.shape[0]
        y = [obs[:, i*3:i*3+3] for i in range(self.n_stack)]
        x = einops.rearrange(y, 'i B C H W -> (B i) C H W').contiguous()
        obs_emb = self.enc(x).view(B, self.n_stack, -1) # B, N, D

        eef_in = torch.cat([batch['target_eef'], obs_emb.view(B, -1)], -1)
        pred_ac = self.eef_network(eef_in)
        ret = dict(pred_ac=pred_ac.detach())
        if compute_loss:
            loss = nn.MSELoss()(pred_ac, batch['action'])
            ret['loss'] = loss
        return ret

    def get_action(self, obs, goal):
        batch = {'target_eef': goal, 'obs': obs}
        ret = self.ff(batch, compute_loss=False)
        return dict(action=ret['pred_ac'])
    

def _parse_args():

    parser = argparse.ArgumentParser()
    # basic common params
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight_decay', '-wc', default=1e-4, type=float)
    parser.add_argument('--hidden_dim', '-hd', default=256, type=int)
    parser.add_argument('--batch_size', '-bs', default=1024, type=int)
    parser.add_argument('--lr', '-lr', default=1e-3, type=float)
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
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--stack_frames', default=1, type=int)
    parser.add_argument('--version', default=3, type=int)

    parser.add_argument('--gd', '-gd', default=-1, type=int)
    # checkpoint resuming and testing
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--eval_every_nsteps', default=1000, type=int)
    parser.add_argument('--start_eval_after', default=-1, type=int)

    # wandb
    parser.add_argument('--use_wandb', '-wb', action='store_true')
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')
    parser.add_argument('--num_workers', default=0, type=int)

    return parser.parse_args()

def main(pargs):
    exp_name = f'reacher2d_ctrl_net'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    num_workers = pargs.num_workers

    train_path = f'reacher_2d_train_v{pargs.version}'
    test_path = f'reacher_2d_test_v{pargs.version}'

    train_dataset = Reacher2DGCBCDataset(data_path=train_path, image_based=True, n_stack_frames=4)
    test_dataset = Reacher2DGCBCDataset(data_path=test_path, image_based=True, n_stack_frames=4)

    print('training set size:', len(train_dataset))
    print('test set size:',     len(test_dataset))

    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=num_workers, pin_memory=True)
    vloader = DataLoader(test_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=num_workers, pin_memory=True)
    batch = next(iter(tloader))


    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        lr=pargs.lr,
        wd=pargs.weight_decay,
    )

    ######## check model's input to output dependency
    args_var = vars(pargs)
    ckpt = args_var.get('ckpt', '')
    resume = args_var.get('resume', False)

    
    agent = PredNN.load_from_checkpoint(ckpt) if ckpt else PredNN(config)
    agent = agent.to(device=pargs.device)

    # agent.train()
    # losses = []
    # for batch in tloader:
    #     batch = {k: v.to(agent.device) for k, v in batch.items()}
    #     loss = agent.training_step([batch])
    #     losses.append(loss.item())
    
    # print(f'initialization loss: {np.mean(losses)}')
    # breakpoint()

    train = (ckpt and resume) or not ckpt
    if pargs.use_wandb and train:
        import wandb
        run_name = exp_name if not pargs.run_name else f'{exp_name}_{pargs.run_name}'
        wandb_run = wandb.init(
            project='reacher2d_ctrl_net',
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

    trainer = pl.Trainer(
        max_epochs=pargs.max_epochs,
        max_steps=pargs.max_steps,
        resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
        callbacks=[ckpt_callback_valid, ckpt_callback_train],
    )

    eval_output_dir = ''
    if train:
        trainer.fit(agent, train_dataloaders=[tloader], val_dataloaders=[vloader])
        eval_output_dir = Path(ckpt_callback_valid.best_model_path).parent
        print(f'Evaluating with {ckpt_callback_valid.best_model_path} ...')
        agent = agent.load_from_checkpoint(ckpt_callback_valid.best_model_path)
    else:
        eval_output_dir = str(Path(ckpt).parent.resolve())


    conf = ParamDict(max_eval_episodes=100, max_render=10, verbose=True, env_name='Reacher2DTest-v1', is_image=True)
    evaluator = Custom_EvaluatorReacher2D_GCBC_Img(conf, test_dataset, eval_output_dir, mode='test')
    evaluator.eval(agent)


if __name__ == '__main__':
    main(_parse_args())
