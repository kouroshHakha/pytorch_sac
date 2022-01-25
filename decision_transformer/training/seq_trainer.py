import time
import torch
import numpy as np
from tqdm import tqdm
from decision_transformer.training.trainer import Trainer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset


import gym
import d4rl


class PointmassBCDataset(Dataset):

    def __init__(self, name='maze2d-open-v0', block_size=64):
        
        self.block_size = block_size
        
        env = gym.make(name)
        self.data = env.get_dataset()
        self.obses = self.data['observations']
        self.state_mean = self.obses.mean(0)
        self.state_std = self.obses.std(0)
        
        self.data_size = self.obses.shape[0] - self.block_size - 10
  
        self.acs = self.data['actions']
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def __len__(self):
        return len(self.obses) - self.block_size

    def __getitem__(self, idx):
   
        idx = idx % self.data_size
        
        max_len = self.block_size
        traj = self.data
        si = idx

        s = (traj['observations'][si:si + max_len] - self.state_mean ) / self.state_std
        a = traj['actions'][si:si + max_len] 
        r = traj['rewards'][si:si + max_len][...,None]
        if 'terminals' in traj:
            d = traj['terminals'][si:si + max_len]
        else:
            d = traj['dones'][si:si + max_len]
            
        timesteps = np.arange(s.shape[0])
 
        rtg = None
  
        mask = np.ones(max_len)
        s = torch.from_numpy(s)
        a = torch.from_numpy(a)
        r = torch.from_numpy(r)
        d = torch.from_numpy(d)
        rtg = torch.zeros(r.shape)
        timesteps = torch.from_numpy(timesteps)
        mask = torch.from_numpy(mask)
        
        return s, a, r, d, rtg, timesteps, mask
    

class SequenceTrainer(Trainer):

    def train_step(self):
        if self.loader is None:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        else:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = next(self.loader)
      
        

        if states.device != self.model.transformer.device:
            device = self.model.transformer.device
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            rtg = rtg.to(device)
            timesteps = timesteps.to(device)
            attention_mask = attention_mask.to(device)
        
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )
     

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

    def train_only_iteration(self, num_steps, iter_num=0, print_logs=False):
        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        # use loader
        if self.dataset is not None:
            
            data = PointmassBCDataset(name=self.dataset, block_size=self.block_size)

            self.loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=self.batch_size,
                                num_workers=4)
        
            self.loader = iter(self.loader)
            
            for _ in tqdm(range(num_steps)):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
        # custom loader
        else:
            for _ in tqdm(range(num_steps)):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs


class CategoricalSequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, _, rtg, timesteps, attention_mask, dists = self.get_batch(self.batch_size)
        
        
        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, dists, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()

        # use loader
        if self.loader_fn is not None:
            self.loader = self.loader_fn()
            for _ in self.loader:
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
        # custom loader
        else:
            for _ in range(num_steps):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model, iter_num-1)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_only_iteration(self, num_steps, iter_num=0, print_logs=False):
        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs
