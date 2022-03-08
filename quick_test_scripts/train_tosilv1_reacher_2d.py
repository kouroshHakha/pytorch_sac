
import dataclasses
from functools import partial
from gc import callbacks
from typing import Optional
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

from osil.nets import TOsilv1, TOsilv1DebugReacher, get_goal_color
from osil.utils import ParamDict
from osil.eval import EvaluatorReacher2D_TOSIL, EvaluationCallback
from osil.data import collate_fn_for_supervised_osil, OsilPairedDataset, OsilDataCollector

from osil.debug import register_pdb_hook
register_pdb_hook()

from torch.utils.data import Dataset
import torch
import d4rl; import gym

import tqdm


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
    parser.add_argument('--num_shots', '-ns', default=-1, type=int, 
                        help='number of shots per each task variation \
                            (-1 means max number of shots available in the dataset)')
    parser.add_argument('--task_size', default=-1, type=int)

    # checkpoint resuming and testing
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--save_ckpt_every_steps', type=int, default=-1)
    parser.add_argument('--eval_every_nsteps', default=1000, type=int)
    parser.add_argument('--start_eval_after', default=-1, type=int)
    
    # wandb
    parser.add_argument('--use_wandb', '-wb', action='store_true')
    parser.add_argument('--wandb_id', type=str, help='Wandb id to allow for resuming')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name, if not provided the deafult of wandb is used')

    return parser.parse_args()



class OsilPairedDatasetReacher2D(Dataset):

    def __init__(
        self,
        data_path,
        nshots_per_task=-1, # sets the number of examples per task variation, -1 means to use the max
        task_size=-1,
    ):

        self.data_path = Path(data_path)
        collector = OsilDataCollector.load(data_path)
        self.raw_data = collector.data
        self.nshots_per_task = nshots_per_task

        class_to_task_map = {}
        class_id = 0
        for task_id in self.raw_data:
            task_size = len(self.raw_data[task_id]) if task_size == -1 else task_size
            for var_id in sorted(self.raw_data[task_id].keys()):
                if var_id < task_size:
                    class_to_task_map[class_id] = (task_id, var_id)
                    class_id += 1
        self.allowed_ids = list(class_to_task_map.values())


        self.ep_lens = []
        for task_id, var_id in self.allowed_ids:
            episodes = self.raw_data[task_id][var_id]
            max_ep_idx = len(episodes) if nshots_per_task == -1 else nshots_per_task
            if max_ep_idx < 2:
                print(f'You need at least two examples per task to make an osil statement, using two instead.')
                max_ep_idx = 2
            if max_ep_idx > len(episodes):
                print(f'Using fewer than {max_ep_idx}, since there are not too many samples for task {task_id}_{var_id}.')
                max_ep_idx = len(episodes)

            self.ep_lens.append(max_ep_idx)
        
        self.n_episodes = sum(self.ep_lens)
        
    def __len__(self):
        return self.n_episodes

    def __getitem__(self, index):
        idx = index % len(self.allowed_ids)
        task_id, var_id = self.allowed_ids[idx]
        episodes = self.raw_data[task_id][var_id]

        c_idx, t_idx = np.random.randint(self.ep_lens[idx], size=(2,))

        data = dict(
            context_s=torch.as_tensor(episodes[c_idx]['state'], dtype=torch.float),
            context_a=torch.as_tensor(episodes[c_idx]['action'], dtype=torch.float),
            target_s=torch.as_tensor(episodes[t_idx]['state'], dtype=torch.float),
            target_a=torch.as_tensor(episodes[t_idx]['action'], dtype=torch.float),
        )

        data.update(
            target_s_enc=data['target_s'].clone(), 
            target_a_enc=data['target_a'].clone(),    
        )

        return data

def main(pargs):
    exp_name = f'tosilv1_reacher2d'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    train_path = 'reacher_2d_train_v2'
    test_path = 'reacher_2d_test_v2'
    train_dataset = OsilPairedDatasetReacher2D(data_path=train_path, nshots_per_task=pargs.num_shots, task_size=pargs.task_size)
    test_dataset = OsilPairedDatasetReacher2D(data_path=test_path)

    collate_fn = partial(collate_fn_for_supervised_osil, padding=pargs.max_padding, pad_targets=pargs.use_gpt_decoder)
    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    vloader = DataLoader(test_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    batch_elem = train_dataset[0]

    # ## debug dataset
    # for batch_idx, batch in tqdm.tqdm(enumerate(tloader)):
    #     context_s = batch['context_s']
    #     target_s = batch['target_s']
    #     ptr = batch['ptr']

    #     # 1. get the goal color based on where the eef of context ends up at
    #     context_goal, c_info = get_goal_color(context_s[:, -1])

    #     # 2. get the final dest. of the target based on its final eef position
    #     target_inds = ptr.cumsum(0) - 1
    #     target_last_states = target_s[target_inds]
    #     target_goal, t_info = get_goal_color(target_last_states)

    #     # 3. see if both colors match
    #     if torch.any(target_goal != context_goal):
    #         print(f'found unmatched context and target @batch_idx = {batch_idx}')
    #         breakpoint()


    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        obs_dim=batch_elem['context_s'].shape[-1],
        max_ep_len=pargs.max_padding,
        ac_dim=batch_elem['context_a'].shape[-1],
        goal_dim=pargs.goal_dim,
        lr=pargs.lr,
        wd=pargs.weight_decay,
        use_gpt_decoder=pargs.use_gpt_decoder,
    )

    args_var = vars(pargs)
    ckpt = args_var.get('ckpt', '')
    resume = args_var.get('resume', False)
    evaluator_cls = EvaluatorReacher2D_TOSIL
    
    # agent = TOsilv1.load_from_checkpoint(ckpt) if ckpt else TOsilv1(config)
    agent = TOsilv1DebugReacher.load_from_checkpoint(ckpt) if ckpt else TOsilv1DebugReacher(config)
    agent = agent.to(device=pargs.device)

    # debug contrastive
    agent.eval()
    losses = []
    for batch in tloader:
        cuda_batch = {k: v.to(agent.device) for k, v in batch.items()}
        ret = agent.ff(cuda_batch, compute_loss=True)
        losses.append(ret['loss'])
    
    print(f'train_loss = {torch.stack(losses).mean().item()}')
    breakpoint()
    agent.train()

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
                save_top_k=1 if pargs.save_ckpt_every_steps == -1 else -1,
                every_n_train_steps=None if pargs.save_ckpt_every_steps == -1 else pargs.save_ckpt_every_steps,
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
        ParamDict(max_eval_episodes=100, env_name='Reacher2D-v1', 
        max_padding=pargs.max_padding, use_gpt_decoder=pargs.use_gpt_decoder),
        train_dataset, mode='train'
    )
    eval_ckpt_train = EvaluationCallback(
        train_evaluator, 
        eval_every_n_updates=pargs.eval_every_nsteps, 
        dirpath=ckpt_callback_valid.dirpath,
        start_evaluating_after=pargs.start_eval_after,
    )

    test_evaluator = evaluator_cls(
        ParamDict(max_eval_episodes=100, env_name='Reacher2DTest-v1',
        max_padding=pargs.max_padding, use_gpt_decoder=pargs.use_gpt_decoder),
        test_dataset, mode='test',
    )
    eval_ckpt_test = EvaluationCallback(test_evaluator, 
        eval_every_n_updates=pargs.eval_every_nsteps,
        dirpath=ckpt_callback_valid.dirpath,
        start_evaluating_after=pargs.start_eval_after,
    )


    trainer = pl.Trainer(
        max_epochs=pargs.max_epochs,
        max_steps=pargs.max_steps,
        resume_from_checkpoint=ckpt if resume else None,
        logger=logger,
        gpus=1 if pargs.device == 'cuda' else 0,
        callbacks=[ckpt_callback_valid, ckpt_callback_train, eval_ckpt_train, eval_ckpt_test],
        terminate_on_nan=True,
    )

    eval_output_dir = ''
    if train:
        trainer.fit(agent, train_dataloaders=[tloader], val_dataloaders=[vloader])
        eval_output_dir = Path(eval_ckpt_test.best_model_path).parent
        print(f'Evaluating with {eval_ckpt_test.best_model_path} ...')
        agent = agent.load_from_checkpoint(eval_ckpt_test.best_model_path)
    else:
        eval_output_dir = pargs.eval_path
        if ckpt and not pargs.eval_path:
            eval_output_dir = str(Path(ckpt).parent.resolve())
            warnings.warn(f'Checkpoint is given for evaluation, but evaluation path is not determined. Using {eval_output_dir} by default')

    conf = ParamDict(max_eval_episodes=100, max_render=10, verbose=True, env_name='Reacher2D-v1',
        max_padding=pargs.max_padding, use_gpt_decoder=pargs.use_gpt_decoder)
    evaluator_cls(conf, train_dataset, eval_output_dir, mode='train').eval(agent)
    conf = ParamDict(max_eval_episodes=100, max_render=10, verbose=True, env_name='Reacher2DTest-v1',
        max_padding=pargs.max_padding, use_gpt_decoder=pargs.use_gpt_decoder)
    evaluator_cls(conf, test_dataset, eval_output_dir, mode='test').eval(agent)



if __name__ == '__main__':
    main(_parse_args())
