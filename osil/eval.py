
import matplotlib.pyplot as plt
import os
import tqdm 
from time import sleep
import numpy as np
from pathlib import Path

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import torch
torch.backends.cudnn.benchmark = True


import dmc, utils
from video import VideoRecorder

from utils import write_yaml, write_pickle
import d4rl; import gym
import yaml
import time

class Evaluator:

    def __init__(self, agent, cfg, output_dir='') -> None:
        self.env = dmc.make(cfg.task_name, cfg.seed, obs_type=cfg.obs_type)
        self.agent = agent
        self.cfg = cfg
   
        # TODO
        self.output_dir = output_dir = Path(output_dir) if output_dir else Path(cfg.ckpt).parent
        if not output_dir:
            raise ValueError('Output dir is empty.')
        self.video = VideoRecorder(output_dir, camera_id=2 if cfg.task_name.startswith('quadruped') else 0)


    def eval_step(self, time_step):
        agent = self.agent
        env = self.env

        with torch.no_grad(), utils.eval_mode(agent):
            # s = time.time()
            obs = torch.from_numpy(time_step.observation).float().to(agent.device)
            obs = obs.unsqueeze(0)
            # move_time = time.time() - s
            # s = time.time()
            action = agent(obs)[0]
            # ff_time = time.time() - s
            # s = time.time()
            time_step = env.step(action.detach().cpu().numpy())
            # env_time = time.time() - s
            # print(f'move_time = {move_time:10.4f}, ff_time = {ff_time:10.4f}, env_time = {env_time:10.4f}')
            return time_step

    def run(self):
        cfg = self.cfg
        env = self.env
        video = self.video

        returns = []
        trange = tqdm.trange(cfg.num_eval_episodes)
        for ep in trange:
            time_step = env.reset()
            ep_rews = []
            video.init(env, enabled=True)
            while not time_step.last():
                video.record(env)
                time_step = self.eval_step(time_step)
                ep_rews.append(time_step.reward)

            returns.append(np.sum(ep_rews))
            video.save(f'ep_{ep}.mp4')

            # update prog bar
            trange.set_description(f'[Ep {ep}] return = {float(np.sum(ep_rews)):.2f}')
            trange.refresh()
            sleep(0.01)

        returns = np.array(returns)
        norm_returns = cfg.normalizer(returns)
        summary_dict = dict(
            returns = returns.tolist(),
            norm_returns = norm_returns.tolist(),
            ret_avg = float(np.mean(returns)),
            ret_std = float(np.std(returns)),
            norm_ret_avg = float(np.mean(norm_returns)),
            norm_ret_std = float(np.std(norm_returns)),
        )

        with open(self.output_dir / 'summary.yaml', 'w') as f:
            yaml.dump(summary_dict, f)

class EvaluatorGC(Evaluator):

    def __init__(self, agent, cfg, output_dir='') -> None:
        super().__init__(agent, cfg, output_dir=output_dir)
        sample_ft_demo = self.cfg['ft_dset'][0]
        self.goal = sample_ft_demo[1]

    def eval_step(self, time_step):
        agent = self.agent
        env = self.env

        with torch.no_grad(), utils.eval_mode(agent):
            # s = time.time()
            obs = torch.from_numpy(time_step.observation).float().to(agent.device)
            obs = obs.unsqueeze(0)
            goal = self.goal
            # move_time = time.time() - s
            # s = time.time()
            action = agent(obs, goal)[0]
            # ff_time = time.time() - s
            # s = time.time()
            time_step = env.step(action.detach().cpu().numpy())
            # env_time = time.time() - s
            # print(f'move_time = {move_time:10.4f}, ff_time = {ff_time:10.4f}, env_time = {env_time:10.4f}')
            return time_step



def evaluate_osil_pm(agent, test_dataset, eval_output_dir='', render_examples=True):
    T_exps = 100
    T_render = 4
    plot_path = Path(eval_output_dir) / f'examples_{T_render}.png'
    output_yaml_path = Path(eval_output_dir) / f'eval.yaml'
    examples = []
    for _ in range(T_exps):
        env = test_dataset.get_new_env()
        idx = np.random.randint(len(test_dataset))
        batch_item = test_dataset[idx]
        output = agent.imitate(env, batch_item)
        examples.append({'demo': batch_item, 'output': output})

    if render_examples:
        nrows = int(len(examples) ** 0.5)
        ncols = -(-len(examples) // nrows) # cieling

        _, axes = plt.subplots(nrows, ncols, figsize=(15, 8), squeeze=False)
        axes = axes.flatten()

        for idx, example in enumerate(examples[:T_render]):
            # demo 
            demo_xys = example['demo'][0].detach().cpu().numpy()[:, :2]

            # # random policy (10 trajectories)
            # rand_policy_trajs = example['output']['rand_trajs']
            # rand_policy_xys = np.stack([np.stack(traj['states'], 0)[:, :2] for traj in rand_policy_trajs], 0) 
            # rand_policy_xys = rand_policy_xys.reshape(-1, 2)
            # axes[idx].scatter(rand_policy_xys[:,0], rand_policy_xys[:,1], c='g', alpha=0.1)

            # TODO: 64 is assumed to be in the middle for now
            # policy output
            output = example['output']
            if 'policy_traj_mask=True' in output:
                policy_xy_masked = np.stack(output['policy_traj_mask=True']['states'], 0)[:, :2]
                axes[idx].plot(policy_xy_masked[:, 0], policy_xy_masked[:, 1], linestyle='-', c='orange', linewidth=5, label='mask=True')
                policy_xy_unmask = np.stack(output['policy_traj_mask=False']['states'], 0)[:, :2]
                axes[idx].plot(policy_xy_unmask[:, 0], policy_xy_unmask[:, 1], linestyle='-', c='red', linewidth=5, label='mask=False')
            else:
                policy_xy_0 = np.stack(output['policy_traj_0']['states'], 0)[:, :2]
                axes[idx].plot(policy_xy_0[:, 0], policy_xy_0[:, 1], linestyle='-', c='orange', alpha=0.5, linewidth=5, label='start=0')
                policy_xy_64 = np.stack(output['policy_traj_64']['states'], 0)[:, :2]
                axes[idx].plot(policy_xy_64[:, 0], policy_xy_64[:, 1], linestyle='-', c='red', alpha=0.5, linewidth=5, label='start=64')

            # mark start and end (goal) of imitation
            axes[idx].scatter([demo_xys[0, 0]], [demo_xys[0, 1]], s=320, marker='*', c='red', label='start')
            axes[idx].scatter([demo_xys[-1, 0]], [demo_xys[-1, 1]], s=320, marker='*', c='green', label='goal')
            mid = len(demo_xys)//2
            axes[idx].scatter([demo_xys[mid, 0]], [demo_xys[mid, 1]], s=320, c='orange', label='policy_start')

            # plot the demonstration
            axes[idx].plot(demo_xys[:, 0], demo_xys[:, 1], linestyle='--', c='b')

            # axes[idx].set_xlim([0, 4])
            # axes[idx].set_ylim([0, 4])

        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=250)


    for example in examples:
        demo = example['demo']
        output=  example['output']

        demo_xys = demo[0].detach().cpu().numpy()[:, :2]
        if 'policy_traj_mask=True' in output:
            policy_xy_masked = np.stack(output['policy_traj_mask=True']['states'], 0)[:, :2]
            policy_xy_unmask = np.stack(output['policy_traj_mask=False']['states'], 0)[:, :2]
        else:
            policy_xy_0 = np.stack(output['policy_traj_0']['states'], 0)[:, :2]
            policy_xy_64 = np.stack(output['policy_traj_64']['states'], 0)[:, :2]


# set the spin colors of an axes
def set_spine_color(ax, color):
    for dir in ['top', 'bottom', 'left', 'right']:
        ax.spines[dir].set_color(color)
        ax.spines[dir].set_linewidth(4)

class EvaluatorPointMazeBase:


    def __init__(self, conf, agent, output_dir, test_dset):
        self.conf = conf
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.test_cases = self.get_test_cases(test_dset)

    @classmethod
    def get_test_cases(cls, test_dataset):
        print('Preparing test cases from test dataset ...')
        test_task_ids = test_dataset.allowed_ids
        
        # (states, actions, rst_state)
        test_cases = []
        for task_id, var_id in test_task_ids:
            episodes = test_dataset.raw_data[task_id][var_id]
            
            for ep_id, ep in enumerate(episodes):
                # random design choice: use the next episode to obtain reset
                rst_idx = (ep_id + 1) % len(episodes)
                test_dict = dict(
                    context_s=ep['state'],
                    context_a=ep['action'],
                    target_s=episodes[rst_idx]['state'],
                    target_a=episodes[rst_idx]['action'],
                    rst=episodes[rst_idx]['state'][0],
                )
                test_cases.append(test_dict)

        print('Preparation done.')
        return test_cases


    def _get_goal(self, demo_state, demo_action):
        raise NotImplementedError

    def _get_action(self, state, goal):
        raise NotImplementedError

    def eval(self):
        successes = []
        example_trajs = []
        print(f'Running evaluation on {len(self.test_cases)} test cases ...')
        for test_case in self.test_cases:

            demo_state      = test_case['context_s']
            demo_action     = test_case['context_a']
            new_rst_state   = test_case['rst']

            env = gym.make(self.conf.env_name)
            
            # set the reset
            pos = new_rst_state[:2]
            vel = new_rst_state[2:]
            env.reset()
            env.set_state(pos, vel)

            # set the target
            goal = self._get_goal(demo_state, demo_action)
            s = new_rst_state
            done = False
            step = 0

            visited_xys = []
            while not done and step < 128:
                # step through the policy
                a = self._get_action(s, goal)
                # a = test_case['target_a'][a_idx]

                visited_xys.append(s[:2])
                ns, _, _, _ = env.step(a)

                if np.linalg.norm(ns[:2] - demo_state[-1][:2]) < 0.1:
                    done = True
                else:
                    s = ns
                    step += 1

            example_trajs.append(dict(
                visited_xys=np.stack(visited_xys, 0),
                demo_xy=demo_state[:, :2],
                gt_xy=test_case['target_s'][:, :2],
            ))
            successes.append(done)

        write_yaml(self.output_dir / 'summary.yaml', dict(success_rate=float(np.mean(successes))))
        write_pickle(self.output_dir / 'example_trajs.pkl', example_trajs)

        print('Plotting examples ...')
        T = 16
        nrows = int(T ** 0.5)
        ncols = -(-T // nrows) # cieling
        plot_path = self.output_dir / f'examples_{T}.png'

        _, axes = plt.subplots(nrows, ncols, figsize=(15, 8), squeeze=False)
        axes = axes.flatten()
        for idx, traj in enumerate(example_trajs[:T]):
            policy_xy = traj['visited_xys']
            demo_xy = traj['demo_xy']
            gt_xy = traj['gt_xy']
            axes[idx].plot(policy_xy[:, 0], policy_xy[:, 1], linestyle='-', c='orange', linewidth=5, label='policy', alpha=0.5)
            axes[idx].plot(demo_xy[:, 0], demo_xy[:, 1], linestyle='-', c='red', linewidth=5, label='demo', alpha=0.5)
            axes[idx].plot(gt_xy[:, 0], gt_xy[:, 1], linestyle='--', c='blue', linewidth=1, label='gt')
            axes[idx].scatter([demo_xy[-1, 0]], [demo_xy[-1, 1]], s=320, marker='*', c='green', label='goal')
            axes[idx].scatter([policy_xy[-1, 0]], [policy_xy[-1, 1]], s=320, marker='*', c='red', label='policy_end')
            set_spine_color(axes[idx], 'green' if successes[idx] else 'red')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=250)

        # plot failed ones too
        num_fails = len(successes) - sum(successes)
        if num_fails > 0:
            print('Plotting failed examples ...')
            T = min(16, len(successes) - sum(successes))
            nrows = int(T ** 0.5)
            ncols = -(-T // nrows) # cieling
            plot_path = self.output_dir / f'examples_{T}_failed.png'

            plt.close()
            _, axes = plt.subplots(nrows, ncols, figsize=(15, 8), squeeze=False)
            axes = axes.flatten()

            count = 0
            for idx, traj in enumerate(example_trajs):
                if successes[idx]:
                    continue
                elif count == T:
                    break
                policy_xy = traj['visited_xys']
                demo_xy = traj['demo_xy']

                axes[count].plot(policy_xy[:, 0], policy_xy[:, 1], linestyle='-', c='orange', linewidth=5, label='policy', alpha=0.5)
                axes[count].plot(demo_xy[:, 0], demo_xy[:, 1], linestyle='-', c='red', linewidth=5, label='demo', alpha=0.5)
                axes[count].plot(gt_xy[:, 0], gt_xy[:, 1], linestyle='--', c='blue', linewidth=1, label='gt')
                axes[count].scatter([demo_xy[-1, 0]], [demo_xy[-1, 1]], s=320, marker='*', c='green', label='goal')
                axes[count].scatter([policy_xy[-1, 0]], [policy_xy[-1, 1]], s=320, marker='*', c='red', label='policy_end')
                count += 1

            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=250)
        print(f'Evaluating the agent is done, success rate: {float(np.mean(successes))}')


