
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



class OsilPairedDatasetReacher2D(Dataset):

    def __init__(
        self,
        data_path,
        mode='train', # valid / test are also posssible
        seed=0,
        nshots_per_task=-1, # sets the number of examples per task variation, -1 means to use the max
        env_name='',
    ):
        # SPLITS = {'train': (0, 0.8), 'valid': (0.8, 0.9), 'test': (0.9, 1)}
        self.splits = SPLITS[env_name]


        self.data_path = Path(data_path)
        collector = OsilDataCollector.load(data_path)
        self.raw_data = collector.data

        # task_name_list = []
        class_to_task_map = {}
        class_id = 0
        for task_id in self.raw_data:
            for var_id in sorted(self.raw_data[task_id].keys()):
                class_to_task_map[class_id] = (task_id, var_id)
                class_id += 1
        self.allowed_ids = [class_to_task_map[i] for i in self.splits[mode]]


        self.ep_lens = []
        for task_id, var_id in self.allowed_ids:
            episodes = self.raw_data[task_id][var_id]
            max_ep_idx = len(episodes) if nshots_per_task == -1 else nshots_per_task
            if max_ep_idx < 2:
                print(f'You need at least two examples per task to make an osil statement, using two instead.')
                max_ep_idx = 2
            if max_ep_idx > len(episodes):
                print(f'Using fewer than {max_ep_idx}, since there are not too many samples for task {task_id}_{var_id}.')
            self.ep_lens.append(max_ep_idx)
            # for var_id in self.raw_data[task_id]:
                # task_name_list.append((task_id, var_id))
                # episodes = self.raw_data[task_id][var_id]
                # max_ep_idx = len(episodes) if nshots_per_task == -1 else nshots_per_task
                # if max_ep_idx < 2:
                #     print(f'You need at least two examples per task to make an osil statement, using two instead.')
                #     max_ep_idx = 2
                # if max_ep_idx > len(episodes):
                #     print(f'Using fewer than {max_ep_idx}, since there are not too many samples for task {task_id}_{var_id}.')
                # ep_len_list.append(max_ep_idx)
        # np.random.seed(seed+10)
        # inds = np.random.permutation(np.arange(len(task_name_list)))
        
        # sratio, eratio = SPLITS[mode]
        # s = int(sratio * len(inds))
        # e = int(eratio * len(inds))
        # assert e > s, 'Not enough data is present for proper split'
        # self.allowed_ids = [task_name_list[int(i)] for i in inds[s:e]]
        # self.ep_lens = [ep_len_list[int(i)] for i in inds[s:e]]
        self.n_episodes = sum(self.ep_lens)
        
    def __len__(self):
        return self.n_episodes

    def __getitem__(self, index):
        idx = index % len(self.allowed_ids)
        task_id, var_id = self.allowed_ids[idx]
        episodes = self.raw_data[task_id][var_id]

        c_idx, t_idx = np.random.randint(self.ep_lens[idx], size=(2,))

        return dict(
            context_s=torch.as_tensor(episodes[c_idx]['state'], dtype=torch.float),
            context_a=torch.as_tensor(episodes[c_idx]['action'], dtype=torch.float),
            target_s=torch.as_tensor(episodes[t_idx]['state'], dtype=torch.float),
            target_a=torch.as_tensor(episodes[t_idx]['action'], dtype=torch.float)
        )

def main(pargs):
    exp_name = f'tosilv1_{pargs.env_name}'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    data_path = pargs.dataset_path
    train_dataset = OsilPairedDataset(data_path=data_path, mode='train', nshots_per_task=pargs.num_shots, env_name=pargs.env_name)
    test_dataset = OsilPairedDataset(data_path=data_path, mode='test',   nshots_per_task=100, env_name=pargs.env_name)

    collate_fn = partial(collate_fn_for_supervised_osil, padding=pargs.max_padding, pad_targets=pargs.use_gpt_decoder)
    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    vloader = DataLoader(test_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    batch_elem = train_dataset[0]


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
