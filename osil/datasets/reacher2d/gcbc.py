
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from osil_gen_data.data_collector import OsilDataCollector
from osil.nets import get_goal_color
from utils import read_hdf5, stack_frames

import time

class Reacher2DGCBCDataset(Dataset):

    def __init__(
        self,
        data_path,
        seed=0,
        nshots_per_task=-1, # sets the number of examples per task variation, -1 means to use the max
        task_size=-1,
        image_based=False,
        n_stack_frames=1,
    ):

        self.data_path = Path(data_path)
        collector = OsilDataCollector.load(data_path)
        self.raw_data = collector.data
        self.nshots_per_task = nshots_per_task
        self.image_based = image_based
        self.transform = T.Resize(64)
        self.n_stack_frames = n_stack_frames

        self.total_num_states = 0

        # create this allowed ids to make it compatible with previously implemented evaluation functions
        class_to_task_map = {}
        class_id = 0
        s = time.time()
        for task_id in self.raw_data:
            task_size = len(self.raw_data[task_id]) if task_size == -1 else task_size
            for var_id in sorted(self.raw_data[task_id].keys()):
                if var_id < task_size:
                    class_to_task_map[class_id] = (task_id, var_id)
                    class_id += 1
        self.allowed_ids = list(class_to_task_map.values())

        obses, states, actions, targets = [], [], [], []
        conds = []
        print('Loading the dataset into memory ...')
        for task_id, var_id in self.allowed_ids:
            # if var_id > 150: # smaller training set
            #     continue
            episodes = self.raw_data[task_id][var_id]

            task_obses, task_states, task_actions, task_targets = [], [], [], []
            task_conds = []
            if len(episodes) < 2:
                print(f'Skipping task {task_id}-{var_id}, because of insufficient examples')
                continue

            max_ep_idx = len(episodes) if nshots_per_task == -1 else nshots_per_task
            if max_ep_idx < 2:
                print(f'You need at least two examples per task to make an osil statement, using two instead.')
                max_ep_idx = 2
            if max_ep_idx > len(episodes):
                print(f'Using fewer than {max_ep_idx}, since there are not too many samples for task {task_id}_{var_id}.')
            for ep in episodes[:max_ep_idx]:
                
                cond_id = ep['cond_id'].item()
                if self.image_based:

                    original_path = self.data_path / 'torch_imgs' / f'{var_id}_{cond_id}.torch'
                    cached_path = self.data_path / f'torch_imgs_processed_stack_{n_stack_frames}' / f'{var_id}_{cond_id}.torch'

                    if cached_path.exists():
                        stacked_imgs = torch.load(cached_path)
                        # this already has C, H, W order
                    else:
                        cached_path.parent.mkdir(exist_ok=True, parents=True)
                        img = torch.load(original_path).permute(0, 3, 1, 2) #channel first
                        stacked_imgs = stack_frames(img, n_stack_frames)
                        # # unit test
                        # assert all(torch.all(stacked_imgs[i] == img[i-n_stack_frames+1:i+1].reshape(n_stack_frames*3, 64, 64)) for i in range(n_stack_frames, len(img)))
                        torch.save(stacked_imgs, cached_path)
                    
                    
                    task_obses.append(self.transform(stacked_imgs).numpy())

                # for some reason images are one time step short from the actual state representation
                ep_len = len(ep['state']) - 1 if self.image_based else len(ep['state'])
                self.total_num_states += ep_len

                task_states.append(ep['state'][:ep_len])
                task_actions.append(ep['action'][:ep_len])
                task_conds.append(cond_id)

            # at the end of this task variation choose the correponding target
            
            # last state of other episode in this sub task (last max_ep_idx episodes)
            # task_targets += [state[-1] for state in task_states]

            # the target color
            task_targets = []
            for state in task_states:
                color, _ = get_goal_color(torch.as_tensor(state[-1:]).float())
                task_targets.append(color[0].numpy())

            if task_obses:
                obses.append(np.stack(task_obses))
            states.append(np.stack(task_states, 0))
            actions.append(np.stack(task_actions, 0))
            targets.append(np.stack(task_targets, 0))
            conds.append(np.array(task_conds))

        print(f'Dataset loading done in {time.time() - s:.4} seconds.')
        # self.states = np.concatenate(states, 0)
        # self.actions = np.concatenate(actions, 0)
        # self.targets = np.concatenate(targets, 0)
        # assert self.states.shape[0] == self.actions.shape[0] == self.targets.shape[0]

        # preserve the trajectory structure
        self.obses = obses
        self.states = states
        self.actions = actions
        self.targets = targets
        self.conds = conds
        assert len(states) == len(self.actions) == len(self.targets) == len(self.conds)

        # hard-code size for now
        
    def __len__(self):
        # return len(self.states)
        return self.total_num_states

    def __getitem__(self, idx):
        # randomly sample the task first
        task_idx = np.random.randint(len(self.states), size=())

        # sample a trajectory idx for the state / action but sample a different index for the target
        state_action_idx, target_idx = np.random.randint(len(self.states[task_idx]), size=(2,))
        
        # get the actions
        acs = self.actions[task_idx][state_action_idx]
        # within a trajectory sample a random time-step
        rand_tstep = np.random.randint(len(acs), size=())
        ac = torch.as_tensor(acs[rand_tstep], dtype=torch.float)
        

        # get the cur and goal state
        if self.image_based:
            s = torch.as_tensor(self.obses[task_idx][state_action_idx][rand_tstep])
            # target_idx = (state_action_idx + 1) % len(self.states[task_idx])
            # g = torch.as_tensor(self.obses[task_idx][target_idx][-1])
            g = torch.as_tensor(self.obses[task_idx][state_action_idx][-1, -3:])
        else:
            s = torch.as_tensor(self.states[task_idx][state_action_idx][rand_tstep], dtype=torch.float)
            g = torch.as_tensor(self.targets[task_idx][target_idx], dtype=torch.float)

        return s, g, ac
