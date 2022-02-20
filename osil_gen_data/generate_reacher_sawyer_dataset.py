

from tqdm import tqdm
import numpy as np
import imageio
import argparse
import itertools
from pathlib import Path
from functools import partial

import torch

import envs; import gym
from osil_gen_data.data_collector import OsilDataCollector

import utils
from snapshot import load_snapshot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    # parser.add_argument('--env_name', type=str, default='maze2d-large-v1', help='Maze type')
    parser.add_argument('--output_suffix', type=str, default='osil_dataset', help='the suffix for output dir')
    # parser.add_argument('--num_tasks', type=int, default=10, help='Max number of goals')
    # parser.add_argument('--num_variation_per_task', type=int, default=10, help='How many times to generate new sequence of goals within a task')
    parser.add_argument('--num_demos_per_variation', type=int, default=100, help='How many times to reset the agent in diff locations')
    parser.add_argument('--agent_snapshot', '-ckpt', type=str)

    return parser.parse_args()

def main():

    args = parse_args()
    env_name = 'reacher_7dof-v1'
    env = gym.make(env_name)

    max_n_render_eps = 10

    # load agent
    agent = load_snapshot(args.agent_snapshot)

    s = env.reset()
    act = env.action_space.sample()
    done = False

    # generate a list of targets from and 6x6x6 grid on target_space
    target_space = env.target_space
    lo = target_space.low
    hi = target_space.high
    ngrid = 4
    range_list = [np.linspace(lo[i], hi[i], ngrid).tolist() for i in range(len(lo))]

    target_list = []
    for element in itertools.product(*range_list):
        target_list.append(list(map(partial(round, ndigits=2), element)))
    target_list = np.stack(target_list, 0)

    path_name = f'./{env_name}_{args.output_suffix}'
    data = OsilDataCollector(path=path_name)
    for task_idx in [0]: # only goal reaching task
        for var_idx in range(len(target_list)):
            ep_count = 0

            imgs = []
            with tqdm(total=args.num_demos_per_variation, desc=f'[var_id={var_idx}]') as tbar:                
                while ep_count < args.num_demos_per_variation:

                    # start a new episode 
                    env.reset()
                    env.set_target(target_list[var_idx])
                    s = env.get_obs()
                    # s = env.reset_model(reset_target=False)
                    
                    done = False
                    ep = {'state': [], 'action': [], 'target': []}
                    ep_len = 0
                    while not done:
                        # act = env.action_space.sample()
                        with torch.no_grad(), utils.eval_mode(agent):
                            # step is a dummy value in eval_mode=True it should not be used
                            act = agent.act(s, step=0, eval_mode=True)

                        if args.noisy:
                            act = act + np.random.randn(*act.shape) * 0.5

                        act = np.clip(act, -1.0, 1.0)
                        # here we have s, act, target
                        obs = env.get_obs(remove_target=True)
                        ep['state'].append(obs)
                        ep['action'].append(act)
                        ep['target'].append(env.get_target())

                        s, _, done, _ = env.step(act)
                        ep_len += 1

                        if args.render and ep_count < max_n_render_eps:
                            img = env.unwrapped.sim.render(128, 128, mode='offscreen')
                            imgs.append(img[::-1])

                    ep_count += 1
                    tbar.update(1)
                    data.append(ep, task_idx, var_idx)

                data.save_var(task_idx, var_idx)

                if imgs:
                    imageio.mimsave(Path(path_name) / f'data_{task_idx}_{var_idx}.gif', imgs, fps=25)

    data.save_meta()
    foo = data.npify(data.data)

    # select the train / valid / test splits based on gif visualization
    # logic: use targets more leaning towards right as train, and the left most ones for test and valid
    # based on the visualization early var_ids are more towards left
    test_valid_cands = np.random.permutation(np.arange(0, 12)).tolist()
    test_var_ids = test_valid_cands[:6]
    valid_var_ids = test_valid_cands[6:]
    train_var_ids = np.arange(12, 64).tolist()

    print('-'*30)
    print('test ids:')
    print(test_var_ids)
    print('valid ids:')
    print(valid_var_ids)
    print('train ids:')
    print(train_var_ids)

    """
    v0
    ---
    freezing these numbers:
    test ids:
    [4, 10, 3, 9, 7, 2]
    valid ids:
    [6, 1, 5, 8, 0, 11]
    train ids: np.arange(12, 64)
    [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    """

    breakpoint()


if __name__ == '__main__':
    main()