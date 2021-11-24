
from pathlib import Path
import pickle

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