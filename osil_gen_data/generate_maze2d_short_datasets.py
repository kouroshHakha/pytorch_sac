from collections import defaultdict
import gym
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import argparse
from pathlib import Path
from utils import read_hdf5, read_pickle, write_hdf5, write_pickle, write_yaml
from tqdm import tqdm
from time import sleep

from osil_gen_data.data_collector import OsilDataCollector

def get_random_xy():
    r = np.random.rand(2)
    # TODO: These numbers are for maze2d-open
    high = np.array([4, 6])
    low = np.array([0, 0])
    return r * (high - low) + low

dist_threshold = 4.0
def get_rst(env):
    s = env.reset_to_location(get_random_xy())
    while np.linalg.norm(s[:2] - env._target, 2) > dist_threshold:
        # print('trying again ... ', s[:2], env._target)
        s = env.reset_to_location(get_random_xy())
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-large-v1', help='Maze type')
    parser.add_argument('--output_suffix', type=str, default='osil_dataset', help='the suffix for output dir')
    parser.add_argument('--num_tasks', type=int, default=10, help='Max number of goals')
    parser.add_argument('--num_variation_per_task', type=int, default=10, help='How many times to generate new sequence of goals within a task')
    parser.add_argument('--num_demos_per_variation', type=int, default=100, help='How many times to reset the agent in diff locations')

    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps
    MAX_EP_LEN = 128

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze)
    target_list = np.stack([np.array(loc) for loc in env.empty_and_goal_locations], 0).astype(env.observation_space.dtype) 

    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    done = False

    path_name = f'./{args.env_name}_{args.output_suffix}' if not args.noisy else f'./{args.env_name}_{args.output_suffix}_noisy'
    data = OsilDataCollector(path=path_name)
    for task_idx in [0]: #tqdm(range(args.num_tasks)):
        for var_idx in range(len(target_list)):
            ep_count = 0
            # target_list = []
            # for _ in range(task_idx + 1):
            #     env.set_target()
            #     target_list.append(env.get_target())

            with tqdm(total=args.num_demos_per_variation, desc=f'[var_id={var_idx}]') as tbar:                
                while ep_count < args.num_demos_per_variation:

                    # start a new episode 
                    # tgt = np.array([2.94510641, 0.9043638])
                    env.set_target(target_list[var_idx])
                    s = get_rst(env)
                    target_cnt = 0
                    # s = np.array([3.19845818e+00, 7.98728805e+00, -1.37212445e-03, -2.61658096e-01])
                    # env.set_state(s[:2], s[2:])
                    act = env.action_space.sample()
                    done = False
                    ep = {'state': [], 'action': [], 'target': []}
                    ep_len = 0
                    while target_cnt < task_idx + 1 and ep_len < MAX_EP_LEN :
                        
                        position = s[0:2]
                        velocity = s[2:4]
                        env.set_target(target_list[var_idx])
                        act, done = controller.get_action(position, velocity, env._target)
                        if args.noisy:
                            act = act + np.random.randn(*act.shape)*0.5

                        act = np.clip(act, -1.0, 1.0)
                        ns, _, _, _ = env.step(act)
                        # here we have s, act, target
                        ep['state'].append(s)
                        ep['action'].append(act)
                        ep['target'].append(env._target)
                        ep_len += 1

                        if done:
                            target_cnt += 1
                            # env.set_target(target_list[target_cnt])
                            done = False
                        else:
                            s = ns

                        if args.render:
                            env.render()

                    if ep_len >= MAX_EP_LEN:
                        # print(f'remaining goals {remaining_goals} but episode reached a max len of {ep_len}')
                        continue

                    ep_count += 1
                    tbar.update(1)
                    data.append(ep, task_idx, var_idx)

                data.save_var(task_idx, var_idx)

    data.save_meta()
    foo = data.npify(data.data)
    breakpoint()

if __name__ == "__main__":
    main()
