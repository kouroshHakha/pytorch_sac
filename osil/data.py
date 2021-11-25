
from pathlib import Path
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

class DMC(Dataset):

    def __init__(self, task_name='walker_stand', obs_type='state', block_size=64, num_trajs=100):

        assert obs_type in ['states', 'pixels']
        
        self.block_size = block_size

        with open(Path('agent_runs') / f'{task_name}.pickle', 'rb') as f:
            data = pickle.load(f)
        
        self.obs = data['obs'] if obs_type == 'pixels' else data['state']
        self.acs = data['action']
        self.terminal = data['terminal']
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