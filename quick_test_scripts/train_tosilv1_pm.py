
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
    ):
        SPLITS = {'train': (0, 0.8), 'valid': (0.8, 0.9), 'test': (0.9, 1)}

        self.data_path = Path(data_path)
        collector = OsilDataCollector.load(data_path)
        self.raw_data = collector.data

        task_name_list = []
        ep_len_list = []
        for task_id in self.raw_data:
            for var_id in self.raw_data[task_id]:
                task_name_list.append((task_id, var_id))
                ep_len_list.append(len(self.raw_data[task_id][var_id]))
        np.random.seed(seed)
        inds = np.random.permutation(np.arange(len(task_name_list)))
        
        sratio, eratio = SPLITS[mode]
        s = int(sratio * len(inds))
        e = int(eratio * len(inds))
        assert e > s, 'Not enough data is present for proper split'
        self.allowed_ids = [task_name_list[int(i)] for i in inds[s:e]]
        self.n_episodes = sum([ep_len_list[int(i)] for i in inds[s:e]])
        
    def __len__(self):
        return self.n_episodes

    def __getitem__(self, index):
        idx = index % len(self.allowed_ids)
        task_id, var_id = self.allowed_ids[idx]
        episodes = self.raw_data[task_id][var_id]

        c_idx, t_idx = np.random.randint(len(episodes), size=(2,))

        return dict(
            context_s=torch.as_tensor(episodes[c_idx]['state'], dtype=torch.float),
            context_a=torch.as_tensor(episodes[c_idx]['action'], dtype=torch.float),
            target_s=torch.as_tensor(episodes[t_idx]['state'], dtype=torch.float),
            target_a=torch.as_tensor(episodes[t_idx]['action'], dtype=torch.float)
        )

def custom_collate(batch, padding=128):
    # Note: we will have dynamic batch size for training the MLP part which should be ok? not sure!
    # you should use torch.repeat_interleave(x, ptr, dim=0) to repeat 
    # x with a frequency of values in ptr across dim=0 (this is needed in the decoder training)
    elem = batch[0]
    ret = {}
    attn_mask = None
    for key in elem:
        if key.startswith('context'):
            # zero pad and then stack + create attention mask
            token_shape = (len(batch), padding) + elem[key].shape[1:]
            ret[key] = torch.zeros(token_shape, dtype=elem[key].dtype, device=elem[key].device) 
            if attn_mask is None:
                attn_mask = torch.zeros(token_shape[:2], dtype=torch.long, device=elem[key].device)
            for e_idx, e in enumerate(batch):
                # left padding
                seq_len = min(len(e[key]), padding)
                if seq_len != len(e[key]):
                    print(f'Cutting off a trajectory (length = {len(e[key])}) because it is too long!')
                ret[key][e_idx, :seq_len] = e[key][:seq_len]
                attn_mask[e_idx, :seq_len] = 1
            
            if 'attention_mask' not in ret and attn_mask is not None:
                # makes sure we only create attn_mask once based on c_s or c_a
                ret['attention_mask'] = attn_mask

        elif key.startswith('target'):
            ret[key] = torch.cat([e[key].view(-1, elem[key].shape[-1]) for e in batch], 0)
            if 'ptr' not in ret:
                ret['ptr'] = torch.tensor([len(e[key]) for e in batch])

    return ret
    

class Evaluator(EvaluatorPointMazeBase):

    def _get_goal(self, demo_state, demo_action):
        device = self.agent.device
        batch = dict(
            context_s=torch.as_tensor(demo_state).float().to(device),
            context_a=torch.as_tensor(demo_action).float().to(device),
        )
        batch = custom_collate([batch], padding=self.conf.max_padding)
        with torch.no_grad():
            goal = self.agent.get_task_emb(batch['context_s'], batch['context_a'], batch['attention_mask'])
            goal = goal.squeeze(0)

        return goal.detach().cpu().numpy()

    def _get_action(self, state, goal):
        device = self.agent.device
        state_tens = torch.as_tensor(state[None], dtype=torch.float, device=device)
        goal_tens = torch.as_tensor(goal[None], dtype=torch.float, device=device)
        pred_ac = self.agent.decoder(state_tens, goal_tens)
        a = pred_ac.squeeze(0).detach().cpu().numpy()
        return a


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
    exp_name = f'tosilv1_pm'
    print(f'Running {exp_name} ...')
    pl.seed_everything(pargs.seed)
    
    data_path = pargs.dataset_path
    dset = PointmassBCDataset(data_path=data_path, mode='train')
    train_dataset = Subset(dset, indices=np.arange(int(len(dset)*pargs.frac)))
    valid_dataset = PointmassBCDataset(data_path=data_path, mode='valid')
    
    collate_fn = partial(custom_collate, padding=pargs.max_padding)
    tloader = DataLoader(train_dataset, shuffle=True, batch_size=pargs.batch_size, num_workers=40, collate_fn=collate_fn)
    vloader = DataLoader(valid_dataset, shuffle=False, batch_size=pargs.batch_size, num_workers=16, collate_fn=collate_fn)
    batch_elem = train_dataset[0]

    # ### visualize data
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

    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('tosil_data.png', dpi=250)
    # breakpoint()

    config = ParamDict(
        hidden_dim=pargs.hidden_dim,
        obs_dim=batch_elem['context_s'].shape[-1],
        max_ep_len=pargs.max_padding,
        ac_dim=batch_elem['context_a'].shape[-1],
        goal_dim=pargs.goal_dim,
        lr=pargs.lr,
        wd=pargs.weight_decay,
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
    else:
        eval_output_dir = pargs.eval_path
        if ckpt and not pargs.eval_path:
            eval_output_dir = str(Path(ckpt).parent.resolve())
            warnings.warn(f'Checkpoint is given for evaluation, but evaluation path is not determined. Using {eval_output_dir} by default')

    test_dataset = PointmassBCDataset(data_path=data_path, mode='test')
    evaluator = Evaluator(pargs, agent, eval_output_dir, test_dataset)
    evaluator.eval()


if __name__ == '__main__':
    main(_parse_args())
