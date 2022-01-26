
from pathlib import Path
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

class DMC(Dataset):

    def __init__(self, task_name='walker_stand', obs_type='state', block_size=64, num_trajs=100):

        assert obs_type in ['states', 'pixels']
        
        self.block_size = block_size
        
        root = Path('agent_runs')
        file_path =  root / f'{task_name}_{num_trajs}.pickle'
        if not file_path.exists():
            file_path =  root / f'{task_name}.pickle'

        with file_path.open('rb') as f:
            data = pickle.load(f)
        
        self.obs = data['obs'] if obs_type == 'pixels' else data['state']
        self.acs = data['action']
        self.reward = data['reward']

        self.num_trajs = num_trajs if num_trajs else len(self.obs)
        self.num_cols = len(self.obs[0]) - self.block_size
    
    def __len__(self):
        return self.num_trajs * self.num_cols

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data

        traj_idx = idx // self.num_cols
        start_idx = idx % self.num_cols
        obses = self.obs[traj_idx][start_idx:start_idx + self.block_size]
        acs = self.acs[traj_idx][start_idx:start_idx + self.block_size]
        
        x = torch.tensor(obses, dtype=torch.float)
        y = torch.tensor(acs, dtype=torch.float)
        return x, y

    def normalize_returns(self, returns):
        max_return = self.reward.sum((-1, -2))[:self.num_trajs].max()
        return returns / max_return

class Concat(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length


class GCDMC(Dataset):

    def __init__(self, task_name='walker_stand', obs_type='state', block_size=64, num_trajs=100):

        assert obs_type in ['states', 'pixels']
        
        root = Path('agent_runs')
        file_path =  root / f'{task_name}_{num_trajs}.pickle'
        if not file_path.exists():
            file_path =  root / f'{task_name}.pickle'

        with file_path.open('rb') as f:
            data = pickle.load(f)
        
        self.obs = data['obs'] if obs_type == 'pixels' else data['state']
        self.acs = data['action']
        self.reward = data['reward']

        self.num_trajs = len(self.obs)
        self.num_cols = len(self.obs[0])

    def __len__(self):
        return self.num_trajs * self.num_cols

    def __getitem__(self, idx):

        traj_idx = idx // self.num_cols
        start_idx = idx % self.num_cols
        obses = self.obs[traj_idx][start_idx:start_idx+1]
        acs = self.acs[traj_idx][start_idx:start_idx+1]
        goal = self.obs[traj_idx][-1:]
        x = torch.tensor(obses, dtype=torch.float)
        g = torch.tensor(goal, dtype=torch.float)
        y = torch.tensor(acs, dtype=torch.float)
        return x, g, y

    def normalize_returns(self, returns):
        max_return = self.reward.sum((-1, -2))[:self.num_trajs].max()
        return returns / max_return

class GCPM(Dataset):

    def __init__(self, name='maze2d-open-v0', block_size=64):
        import d4rl; import gym
            
        self.block_size = block_size
        
        env = gym.make(name)
        self.data = env.get_dataset()
        self.obses = self.data['observations']

        self.acs = self.data['actions']

    def __len__(self):
        return len(self.obses) - self.block_size*2 - 1000
        
    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        obses = self.obses[idx:idx + self.block_size]
        acs = self.acs[idx:idx + self.block_size]
        
        # x, y location of some future state that is at least 24 and at most 64 steps away from the last state in the current trajectory
        # rand_id = np.random.randint(low=24, high=64)
        rand_id = 0
        goal = self.obses[idx + self.block_size + rand_id: idx + self.block_size + rand_id + 1, :2]
        
        
        x = torch.tensor(obses, dtype=torch.float)
        goal = torch.tensor(goal, dtype=torch.float).squeeze(0)
        y = torch.tensor(acs, dtype=torch.float)

        # output: x (B, T, s_dim), goal (B, g_dim), y (B, T, a_dim)
        return x, goal, y


class OsilPM(Dataset):

    def __init__(self, name='maze2d-open-v0', ctx_size=256, trg_size=64):
        import d4rl; import gym

        self.context_size = ctx_size
        self.target_size = trg_size

        env = gym.make(name)
        self.data = env.get_dataset()
        self.obses = self.data['observations']
        self.acs = self.data['actions']

    def __len__(self):
        return len(self.obses) - self.context_size * 2

    def __getitem__(self, idx):
        # c_idx = slice(idx, idx + self.context_size)
        c_s = torch.tensor(self.obses[idx: idx + self.context_size], dtype=torch.float)
        c_a = torch.tensor(self.acs[idx: idx + self.context_size], dtype=torch.float)

        t_start = np.random.randint(low=idx, high=idx + self.context_size - self.target_size)
        t_s = torch.tensor(self.obses[t_start: t_start + self.target_size], dtype=torch.float)
        t_a = torch.tensor(self.acs[t_start: t_start + self.target_size], dtype=torch.float)

        return c_s, c_a, t_s, t_a