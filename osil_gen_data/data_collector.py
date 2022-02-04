from collections import defaultdict
import numpy as np
from pathlib import Path
from utils import read_pickle, write_pickle, write_yaml

class OsilDataCollector:

    def __init__(self, path='./osil_dataset', base_name='data'):
        self.data = defaultdict(dict)

        self.path = Path(path)
        self.base_name = base_name
        self.ep_lens = []
        self.num_total_steps = 0
        self.num_total_episodes = 0

    def append(self, ep, task_idx, variation_idx):
        if variation_idx not in self.data[task_idx]:
            self.data[task_idx][variation_idx] = []
        episodes = self.data[task_idx][variation_idx]
        episodes.append(ep)

        self.ep_lens.append(len(ep['state']))
        self.num_total_steps += len(ep['state'])
        self.num_total_episodes += 1

    @property
    def meta_data(self):
        return {
            'num_total_timesteps': self.num_total_steps,
            'num_total_episodes': self.num_total_episodes,
            'avg_ep_len': float(np.mean(self.ep_lens)),
            'max_ep_len': int(np.max(self.ep_lens)),
            'min_ep_len': int(np.min(self.ep_lens)),
        }

    def npify(self, container):

        if isinstance(container, np.ndarray):
            return container
        if isinstance(container, (list, tuple)):
            if isinstance(container[0], np.ndarray):
                return np.stack(container, 0)
            else:
                return container.__class__([self.npify(item) for item in container])
        elif isinstance(container, dict):
            ret = {}
            for k, v in container.items():
                ret[k] = self.npify(v)
            return ret

        else:
            raise ValueError(f'unsupported container {type(container)} for npify function')

    def get_name(self, task_id, var_id):
        return f'{self.base_name}_{task_id}_{var_id}'

    def get_task_data(self, task_id, var_id=None):
        """
        params:
            task_id -- interpreted as {task_id}_{var_id} if var_id is None, othewise just task_id
            var_id -- variation id if not Nont
        """
        if var_id is None:
            task_id, var_id = map(int, task_id.split('_'))
        
        try:
            return self.npify(self.data[task_id][var_id])
        except KeyError:
            if task_id not in self.data:
                raise ValueError(f'Data object does not contain task_id = {task_id}')
            raise ValueError(f'Data object -- task_id = {task_id} does not contain var_id = {var_id}')

    def save_var(self, task_id, var_id):
        # saves the variation in a sep file
        tname = self.get_name(task_id, var_id)
        fpath = self.path / f'{tname}.pkl'
        dobj = self.get_task_data(task_id, var_id)
        write_pickle(fpath, dobj)

    def save_meta(self):
        # saves a human readable format of meta data
        fpath = self.path / f'{self.base_name}_meta.yaml'
        write_yaml(fpath, self.meta_data)

    @classmethod
    def load(cls, path):
        collector = OsilDataCollector(path=path)
        for fpath in collector.path.iterdir():
            if fpath.is_file() and fpath.suffix == '.pkl':
                split = fpath.stem.split('_')
                task_id, var_id = map(int, split[-2:])
                if not collector.base_name:
                    collector.base_name = '_'.join(split[:-2])
                
                episodes = read_pickle(fpath)
                for ep in episodes:
                    collector.append(ep, task_id, var_id)
        return collector    