
from functools import partial
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

from osil.nets import TOsilv1
from osil.utils import ParamDict
from osil.eval import OsilEvaluatorPM, OsilEvaluatorReacher, EvaluatorReacherSawyerDT
from osil.data import collate_fn_for_supervised_osil, OsilPairedDataset

from osil.debug import register_pdb_hook
register_pdb_hook()

from torch.utils.data import Dataset
import torch
import d4rl; import gym


def _parse_args():

    parser = argparse.ArgumentParser()
    # basic common params
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight_decay', '-wc', default=1e-4, type=float)
    parser.add_argument('--hidden_dim', '-hd', default=256, type=int)
    parser.add_argument('--batch_size', '-bs', default=64, type=int)
    parser.add_argument('--goal_dim', '-gd', default=256, type=int)
    parser.add_argument('--max_padding', default=128, type=int)
    parser.add_argument('--lr', '-lr', default=3e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--env_name', type=str)
    # other params
    parser.add_argument('--use_gpt_decoder', action='store_true')
    parser.add_argument('--use_contrastive', action='store_true')
    parser.add_argument('--num_shots', '-ns', default=-1, type=int, 
                        help='number of shots per each task variation \
                            (-1 means max number of shots available in the dataset)')
    # checkpoint resuming and testing
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--save_ckpt_every_steps', type=int, default=-1)
    
    # wandb
    parser.add_argument('--use_wandb', '-wb', action='store_true')
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')

    return parser.parse_args()


def main(pargs):
    exp_name = f'tosilv1_{pargs.env_name}'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    data_path = pargs.dataset_path
    train_dataset = OsilPairedDataset(data_path=data_path, mode='train', nshots_per_task=pargs.num_shots, env_name=pargs.env_name)
    valid_dataset = OsilPairedDataset(data_path=data_path, mode='valid', nshots_per_task=100, env_name=pargs.env_name)
    test_dataset = OsilPairedDataset(data_path=data_path, mode='test',   nshots_per_task=100, env_name=pargs.env_name)

    collate_fn = partial(collate_fn_for_supervised_osil, padding=pargs.max_padding, pad_targets=pargs.use_gpt_decoder)
    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    vloader = DataLoader(valid_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    batch_elem = train_dataset[0]

    ### visualize data
    # K = 16
    # nrows = int(K ** 0.5)
    # ncols = -(-K // nrows) # cieling

    # plt.close()
    # _, axes = plt.subplots(nrows, ncols, figsize=(15, 8), squeeze=False)
    # axes = axes.flatten()

    # for i in range(K):
    #     sample = valid_dataset[i]
    #     policy_xy = sample['target_s'][:, :2].numpy()
    #     demo_xy = sample['context_s'][:, :2].numpy()
    #     axes[i].plot(policy_xy[:, 0], policy_xy[:, 1], linestyle='-', c='orange', linewidth=5, label='target')
    #     axes[i].plot(demo_xy[:, 0], demo_xy[:, 1], linestyle='-', c='red', linewidth=5, label='context')
    #     axes[i].scatter([demo_xy[-1, 0]], [demo_xy[-1, 1]], s=320, marker='*', c='green', label='context_g')
    #     axes[i].scatter([policy_xy[-1, 0]], [policy_xy[-1, 1]], s=320, marker='*', c='red', label='target_g')
    #     axes[i].set_xlim(0, 4)
    #     axes[i].set_ylim(0, 6)

    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('tosil_data.png', dpi=250)

    # plt.close()
    # _, axes = plt.subplots(1, 1, figsize=(15, 8), squeeze=False)
    # axes = axes.flatten()
    # ax = axes[0]

    # for i in range(len(train_dataset)):
    #     sample = train_dataset[i]
    #     demo_xy = sample['context_s'][:, :2].numpy()
    #     ax.plot(demo_xy[:, 0], demo_xy[:, 1], linestyle='-', c='blue', linewidth=1, label='train')
    #     ax.scatter([demo_xy[-1, 0]], [demo_xy[-1, 1]], s=320, marker='*', c='green')

    # for i in range(len(valid_dataset)):
    #     sample = valid_dataset[i]
    #     demo_xy = sample['context_s'][:, :2].numpy()
    #     ax.plot(demo_xy[:, 0], demo_xy[:, 1], linestyle='-', c='orange', linewidth=1, label='valid')
    #     ax.scatter([demo_xy[-1, 0]], [demo_xy[-1, 1]], s=320, marker='*', c='red')

    # ax.set_xlim(0, 4)
    # ax.set_ylim(0, 6)

    # plt.savefig('debug_tosilv2_train_valid.png', dpi=250)
    # breakpoint()

    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        obs_dim=batch_elem['context_s'].shape[-1],
        max_ep_len=pargs.max_padding,
        ac_dim=batch_elem['context_a'].shape[-1],
        goal_dim=pargs.goal_dim,
        lr=pargs.lr,
        wd=pargs.weight_decay,
        use_gpt_decoder=pargs.use_gpt_decoder,
        use_contrastive=pargs.use_contrastive,
    )

    args_var = vars(pargs)
    ckpt = args_var.get('ckpt', '')
    resume = args_var.get('resume', False)
    
    agent = TOsilv1.load_from_checkpoint(ckpt) if ckpt else TOsilv1(config)
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
                # save_on_train_epoch_end=True,
                mode='min',
                save_top_k=1 if pargs.save_ckpt_every_steps == -1 else -1,
                every_n_train_steps=None if pargs.save_ckpt_every_steps == -1 else pargs.save_ckpt_every_steps,
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
        ckpt_callback.to_yaml(eval_output_dir / 'model_ckpts.yaml')
        agent = agent.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        eval_output_dir = pargs.eval_path
        if ckpt and not pargs.eval_path:
            eval_output_dir = str(Path(ckpt).parent.resolve())
            warnings.warn(f'Checkpoint is given for evaluation, but evaluation path is not determined. Using {eval_output_dir} by default')

    if pargs.env_name.startswith('maze2d'):
        evaluator_cls = OsilEvaluatorPM
    elif pargs.env_name.startswith('reacher'):
        evaluator_cls = OsilEvaluatorReacher
        # evaluator_cls = EvaluatorReacherSawyerDT
    evaluator_cls(pargs, agent, eval_output_dir, test_dataset, mode='test').eval()
    evaluator_cls(pargs, agent, eval_output_dir, valid_dataset, mode='valid').eval()



if __name__ == '__main__':
    main(_parse_args())
