
from logging import warning
from pathlib import Path
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

import d4rl; import gym

from osil_gen_data.data_collector import OsilDataCollector



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

    def __init__(self, name='maze2d-open-v0', ctx_size=256, trg_size=64, goal_type='only_last'):
        self.context_size = ctx_size
        self.target_size = trg_size
        self.goal_type = goal_type

        self.name = name
        env = gym.make(name)
        self.data = env.get_dataset()
        self.obses = self.data['observations']

        self.acs = self.data['actions']

        self.n = len(self.obses) - self.context_size
        self.m = self.context_size - self.target_size

    def __len__(self):
        return self.n * self.m 
        
    def __getitem__(self, idx):
        
        ctx_idx = idx // self.m
        c_s = torch.tensor(self.obses[ctx_idx: ctx_idx + self.context_size], dtype=torch.float)
        c_a = torch.tensor(self.acs[ctx_idx: ctx_idx + self.context_size], dtype=torch.float)
        
        if self.goal_type == 'only_last':
            # x, y location of the last step
            goal = c_s[-1:, :2]
        elif self.goal_type == 'zero':
            goal = torch.zeros(1, 2).to(c_s)
        elif self.goal_type == 'last+start':
            goal = torch.cat([c_s[0, :2], c_s[-1, :2]], -1)[None]
        else:
            raise ValueError('unknown goal type')

        # t_start = np.random.randint(low=idx, high=idx + self.context_size - self.target_size)
        t_start = idx % self.m 
        t_s = torch.tensor(self.obses[t_start: t_start + self.target_size], dtype=torch.float)
        t_a = torch.tensor(self.acs[t_start: t_start + self.target_size], dtype=torch.float)

        # output: x (B, T, s_dim), goal (B, g_dim), y (B, T, a_dim)
        return t_s, goal, t_a


class OsilPM(Dataset):

    def __init__(self, name='maze2d-open-v0', ctx_size=256, trg_size=64):
        self.context_size = ctx_size
        self.target_size = trg_size

        self.name = name
        env = gym.make(name)
        self.data = env.get_dataset()
        self.obses = self.data['observations']
        self.acs = self.data['actions']

    def __len__(self):
        return len(self.obses) - self.context_size

    def __getitem__(self, idx):
        c_s = torch.tensor(self.obses[idx: idx + self.context_size], dtype=torch.float)
        c_a = torch.tensor(self.acs[idx: idx + self.context_size], dtype=torch.float)

        t_start = np.random.randint(low=idx, high=idx + self.context_size - self.target_size)
        t_s = torch.tensor(self.obses[t_start: t_start + self.target_size], dtype=torch.float)
        t_a = torch.tensor(self.acs[t_start: t_start + self.target_size], dtype=torch.float)

        return c_s, c_a, t_s, t_a


class TestOsilPM(Dataset):

    def __init__(self, name='maze2d-open-v0', traj_size=128):
        self.traj_size = traj_size

        self.name = name
        env = gym.make(name)
        self.data = env.get_dataset()
        self.obses = self.data['observations']
        self.acs = self.data['actions']

    def __len__(self):
        return len(self.obses) - self.traj_size

    def __getitem__(self, idx):
        s = torch.tensor(self.obses[idx: idx + self.traj_size], dtype=torch.float)
        a = torch.tensor(self.acs[idx: idx + self.traj_size], dtype=torch.float)
        return s, a

    def get_new_env(self):
        return gym.make(self.name)

###################################################
# to enable backward compatible comparision with the other experiment

SPLITS = {
    'reacher_7dof-v1': {
        # 'valid': [0, 1, 2, 3, 16, 17, 18, 19],
        'valid': [8, 25, 44, 41, 45, 32, 57, 34],
        'test':  [29, 22, 2, 4, 0, 37, 17, 35],
        # 'test': [32, 33, 34, 35, 48, 49, 50, 51],
    }, 
    'maze2d-open-v0': {
        'valid': [3], 
        'test': [6, 7],
    },
    "robosuite_pick_place": {
        "valid": [5, 8],
        "test": [1, 13],
    },
    "robosuite_task0": {
        "train": [0],
        "valid": [],
        "test": [],
    },
}
SPLITS['reacher_7dof-v1']['train'] = [i for i in np.arange(64) if i not in SPLITS['reacher_7dof-v1']['valid'] + SPLITS['reacher_7dof-v1']['test']]
SPLITS['maze2d-open-v0']['train'] = [i for i in np.arange(15) if i not in SPLITS['maze2d-open-v0']['valid'] + SPLITS['maze2d-open-v0']['test']]
SPLITS['robosuite_pick_place']['train'] = [i for i in np.arange(16) if i not in SPLITS['robosuite_pick_place']['valid'] + SPLITS['robosuite_pick_place']['test']]

class OsilPairedDataset(Dataset):

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

def pad_tokens(batch, key, padding):
    elem = batch[0]
    token_shape = (len(batch), padding) + elem[key].shape[1:]
    token = torch.zeros(token_shape, dtype=elem[key].dtype, device=elem[key].device) 
    attn = torch.zeros(token_shape[:2], dtype=torch.long, device=elem[key].device)

    for e_idx, e in enumerate(batch):
        # left padding
        seq_len = min(len(e[key]), padding)
        if seq_len != len(e[key]):
            print(f'Cutting off a trajectory (length = {len(e[key])}) because it is too long!')
        token[e_idx, :seq_len] = e[key][:seq_len]
        attn[e_idx, :seq_len] = 1

    return token, attn

def collate_fn_for_supervised_osil(batch, padding=128, ignore_keys=None, pad_targets=False):
    # Note: we will have dynamic batch size for training the MLP part which should be ok? not sure!
    # you should use torch.repeat_interleave(x, ptr, dim=0) to repeat 
    # x with a frequency of values in ptr across dim=0 (this is needed in the decoder training)
    elem = batch[0]
    ret = {}
    attn_mask = None
    used_keys = set()

    if ignore_keys is None:
        ignore_keys = []
        
    for key in elem:
        if key.startswith(('context_s', 'context_a')):
            # zero pad and then stack + create attention mask
            # token_shape = (len(batch), padding) + elem[key].shape[1:]
            # ret[key] = torch.zeros(token_shape, dtype=elem[key].dtype, device=elem[key].device) 
            # if attn_mask is None:
            #     attn_mask = torch.zeros(token_shape[:2], dtype=torch.long, device=elem[key].device)
            # for e_idx, e in enumerate(batch):
            #     # left padding
            #     seq_len = min(len(e[key]), padding)
            #     if seq_len != len(e[key]):
            #         print(f'Cutting off a trajectory (length = {len(e[key])}) because it is too long!')
            #     ret[key][e_idx, :seq_len] = e[key][:seq_len]
            #     attn_mask[e_idx, :seq_len] = 1
            ret[key], attn_mask = pad_tokens(batch, key, padding)
            if 'attention_mask' not in ret and attn_mask is not None:
                # makes sure we only create attn_mask once based on c_s or c_a
                ret['attention_mask'] = attn_mask


            used_keys.add(key)

        # added for testing the contrastive idea
        elif key.startswith(('target_s_enc', 'target_a_enc')):
            ret[key], attn_mask = pad_tokens(batch, key, padding)
            if 'target_mask_enc' not in ret and attn_mask is not None:
                # makes sure we only create attn_mask once based on c_s or c_a
                ret['target_mask_enc'] = attn_mask
            used_keys.add(key)

        elif key.startswith(('target_s', 'target_a')):
            if pad_targets:
                ret[key], attn = pad_tokens(batch, key, padding)
                if 'target_mask' not in ret and attn is not None:
                    ret['target_mask'] = attn
            else:
                ret[key] = torch.cat([e[key].view(-1, elem[key].shape[-1]) for e in batch], 0)
                if 'ptr' not in ret:
                    ret['ptr'] = torch.tensor([len(e[key]) for e in batch])
            used_keys.add(key)

    for key in elem:
        if key not in used_keys and key not in ignore_keys:
            # try:
                ret[key] = torch.stack([b[key] for b in batch], 0)
            # except:
            #     warning.warn(f'attribute {key} is not batchable, consider removing it from the dataset?')
            #     ret[key] = [b[key] for b in batch]

    return ret
    