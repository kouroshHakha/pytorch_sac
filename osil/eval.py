
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

from tempfile import mkdtemp, mkstemp
import dmc, utils
from video import VideoRecorder

from osil.data import collate_fn_for_supervised_osil

from utils import save_as_gif, write_yaml, write_pickle
import envs
import d4rl; import gym
import yaml
import time
import imageio

# robo suite
from robosuite_env.osil_utils.env_utils import create_env, env_obs_to_vector
from robosuite_env.custom_obs_setting import PICK_PLACE_OBSERVATION_KEYS

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

class EvaluatorBase:


    def __init__(self, conf, agent, output_dir, test_dset, mode='test'):
        self.conf = conf
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.test_cases = self.get_test_cases(test_dset)


    @classmethod
    def get_test_cases(cls, test_dataset):
        print('Preparing test cases from test dataset ...')
        test_task_ids = test_dataset.allowed_ids
        
        # (states, actions, rst_state)
        test_cases = []
        for task_id, var_id in test_task_ids:
            # TODO: grab the first 100
            episodes = test_dataset.raw_data[task_id][var_id][:100]
            
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
        raise NotImplementedError

class EvaluatorPointMazeBase(EvaluatorBase):

    def eval(self):
        successes = []
        example_trajs = []
        print(f'Running evaluation on {len(self.test_cases)} {self.mode} cases ...')
        for test_case in tqdm.tqdm(self.test_cases):

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
                # a = env.action_space.sample()

                visited_xys.append(s[:2])
                ns, _, _, _ = env.step(a)

                if np.linalg.norm(ns[:2] - demo_state[-1][:2]) < 0.15:
                    done = True
                # else:
                if not done:
                    s = ns
                    step += 1

            example_trajs.append(dict(
                visited_xys=np.stack(visited_xys, 0),
                demo_xy=demo_state[:, :2],
                gt_xy=test_case['target_s'][:, :2],
            ))

            successes.append(done)

        write_yaml(self.output_dir / f'summary_{self.mode}.yaml', dict(success_rate=float(np.mean(successes))))
        write_pickle(self.output_dir / f'example_trajs_{self.mode}.pkl', example_trajs)

        print('Plotting examples ...')
        T = 16
        nrows = int(T ** 0.5)
        ncols = -(-T // nrows) # cieling
        plot_path = self.output_dir / f'examples_{self.mode}_{T}.png'

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
            axes[idx].set_xlim([0, 4])
            axes[idx].set_ylim([0, 6])
            set_spine_color(axes[idx], 'green' if successes[idx] else 'red')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=250)

        # plot failed ones too
        num_fails = len(successes) - sum(successes)
        if num_fails > 0:
            print('Plotting failed examples ...')
            T = min(16, num_fails)
            nrows = int(T ** 0.5)
            ncols = -(-T // nrows) # cieling
            plot_path = self.output_dir / f'examples_{self.mode}_{T}_failed.png'

            plt.close()
            _, axes = plt.subplots(nrows, ncols, figsize=(15, 8), squeeze=False)
            axes = axes.flatten()

            count = 0
            for idx, traj in enumerate(example_trajs):
                if successes[idx]:
                    continue
                if count == T:
                    break
                policy_xy = traj['visited_xys']
                demo_xy = traj['demo_xy']
                gt_xy = traj['gt_xy']

                axes[count].plot(policy_xy[:, 0], policy_xy[:, 1], linestyle='-', c='orange', linewidth=5, label='policy', alpha=0.5)
                axes[count].plot(demo_xy[:, 0], demo_xy[:, 1], linestyle='-', c='red', linewidth=5, label='demo', alpha=0.5)
                axes[count].plot(gt_xy[:, 0], gt_xy[:, 1], linestyle='--', c='blue', linewidth=1, label='gt')
                axes[count].scatter([demo_xy[-1, 0]], [demo_xy[-1, 1]], s=320, marker='*', c='green', label='goal')
                axes[count].scatter([policy_xy[-1, 0]], [policy_xy[-1, 1]], s=320, marker='*', c='red', label='policy_end')
                axes[count].set_xlim([0, 4])
                axes[count].set_ylim([0, 6])
                count += 1

            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=250)
        print(f'Evaluating the agent is done, success rate: {float(np.mean(successes))}')


class EvaluatorReacherSawyer(EvaluatorBase):

    def render(self, env):
        return env.unwrapped.sim.render(256, 256, mode='offscreen')[::-1]

    def eval(self):
        successes, rewards = [], []
        print(f'Running evaluation on {len(self.test_cases)} {self.mode} cases ...')

        # TODO
        max_render = 10
        policy_imgs = []
        demo_imgs = []

        demo_failed_imgs, policy_failed_imgs = [], []
        failed_case_counter = 0

        shuffled_inds = np.random.permutation(len(self.test_cases))
        for test_counter, test_idx in tqdm.tqdm(enumerate(shuffled_inds)):
            test_case = self.test_cases[test_idx]
            
            demo_state      = test_case['context_s']
            demo_action     = test_case['context_a']
            new_rst_state   = test_case['rst']
            demo_target     = demo_state[-1, -3:] # eef of the last step

            env = gym.make(self.conf.env_name)

            # render demo policy
            if test_counter < max_render:
                env.reset()
                env.set_target(demo_target)
                s = env.robot_reset_to_qpos(demo_state[0, :7])
                for a in demo_action:
                    demo_imgs.append(self.render(env))
                    env.step(a)

            # set the reset
            env.reset()
            env.set_target(demo_target)
            env.robot_reset_to_qpos(new_rst_state[:7]) # reset qpos
            s = env.get_obs(remove_target=True)

            # set the target
            goal = self._get_goal(demo_state, demo_action)
            done = False
            step = 0

            success = False
            total_dist = 0
            policy_a, policy_s = [], []
            consecutive_steps = 0
            for _ in demo_action: # since the ep_len is always n it makes sense to do this
                # step through the policy
                a = self._get_action(s, goal)
                # a = env.action_space.sample()
                if test_counter < max_render:
                    policy_imgs.append(self.render(env))
                _, _, done, _ = env.step(a)

                # log
                policy_a.append(a)
                policy_s.append(s)

                s = env.get_obs(remove_target=True)
                step_dist = np.linalg.norm(s[-3:] - demo_state[-1, -3:], ord=1)
                total_dist += step_dist
                
                if step_dist < 0.1:
                    if consecutive_steps > 5:
                        success = True
                    consecutive_steps += 1
                else:
                    consecutive_steps = 0
                
                if done:
                    break
                step += 1

            rewards.append(-total_dist)
            successes.append(success)

            if not success and failed_case_counter < max_render:
                # replay both the demo and policy for this index
                env.reset()
                env.set_target(demo_target)
                s = env.robot_reset_to_qpos(demo_state[0, :7])
                for a in demo_action:
                    demo_failed_imgs.append(self.render(env))
                    env.step(a)

                env.reset()
                env.set_target(demo_target)
                s = env.robot_reset_to_qpos(policy_s[0][:7])
                for a in policy_a:
                    policy_failed_imgs.append(self.render(env))
                    env.step(a)

                failed_case_counter += 1
        

        summary = dict(
            success_rate=float(np.mean(successes)),
            total_dist=-float(np.mean(rewards))
        )
        write_yaml(self.output_dir / f'summary_{self.mode}.yaml', summary)
        write_pickle(self.output_dir / f'example_trajs_{self.mode}.pkl', dict(demo=demo_imgs, policy=policy_imgs))

        print(f'success rate: {float(np.mean(successes))}, total_dist: {-float(np.mean(rewards))}')

        if policy_imgs:
            policy_imgs = np.stack(policy_imgs, 0)
            demo_imgs   = np.stack(demo_imgs, 0)

            assert demo_imgs.shape == policy_imgs.shape

            print('Plotting examples ...')
            plot_path = self.output_dir / f'examples_{self.mode}.gif'

            pngs = []
            tmp_dir = mkdtemp()
            for demo_traj, policy_traj in zip(demo_imgs, policy_imgs):
                plt.close()
                fig = plt.figure()
                plt.subplot(121)
                plt.imshow(demo_traj)
                plt.title('demo')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(122)
                plt.imshow(policy_traj)
                plt.title('policy')
                plt.xticks([])
                plt.yticks([])

                _, filename = mkstemp(dir=tmp_dir)
                filename += '.png'

                fig.savefig(filename)
                plt.clf() # clear figure
                pngs.append(filename)

            plot_images = [imageio.imread(png) for png in pngs]
            imageio.mimsave(plot_path, plot_images, fps=25)
            print('Plotting done.')

        if policy_failed_imgs:
            policy_failed_imgs = np.stack(policy_failed_imgs, 0)
            demo_failed_imgs   = np.stack(demo_failed_imgs, 0)
            assert demo_failed_imgs.shape == policy_failed_imgs.shape

            print('Plotting failed examples ...')
            plot_path = self.output_dir / f'examples_{self.mode}_failed.gif'

            pngs = []
            tmp_dir = mkdtemp()
            for demo_traj, policy_traj in zip(demo_failed_imgs, policy_failed_imgs):
                plt.close()
                fig = plt.figure()
                plt.subplot(121)
                plt.imshow(demo_traj)
                plt.title('demo')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(122)
                plt.imshow(policy_traj)
                plt.title('policy')
                plt.xticks([])
                plt.yticks([])

                _, filename = mkstemp(dir=tmp_dir)
                filename += '.png'

                fig.savefig(filename)
                plt.clf() # clear figure
                pngs.append(filename)

            plot_images = [imageio.imread(png) for png in pngs]
            imageio.mimsave(plot_path, plot_images, fps=25)
            print('Plotting done.')

class EvaluatorReacherSawyerDT(EvaluatorReacherSawyer):

    def render(self, env):
        return env.unwrapped.sim.render(256, 256, mode='offscreen')[::-1]

    def _get_goal(self, demo_state, demo_action):
        device = self.agent.device
        batch = dict(
            context_s=torch.as_tensor(demo_state).float().to(device),
            context_a=torch.as_tensor(demo_action).float().to(device),
        )
        batch = collate_fn_for_supervised_osil([batch], padding=self.conf.max_padding, pad_targets=self.conf.use_gpt_decoder)
        with torch.no_grad():
            goal = self.agent.get_task_emb(batch['context_s'], batch['context_a'], batch['attention_mask'])
            goal = goal.squeeze(0)

        return goal.detach().cpu().numpy()

    def _get_action(self, states, actions, goal):
        device = self.agent.device
        state_tens = torch.stack([torch.as_tensor(s, device=device, dtype=torch.float) for s in states], 0)
        goal_tens = torch.as_tensor(goal, device=device, dtype=torch.float)
        if actions:
            action_tens = torch.stack([torch.as_tensor(a, device=device, dtype=torch.float) for a in actions], 0)
        else:
            action_tens = None
        action = self.agent.decoder.get_action(state_tens, goal_tens, past_actions=action_tens)
        return action.detach().cpu().numpy()

    def eval(self):
        successes, rewards = [], []
        print(f'Running evaluation on {len(self.test_cases)} {self.mode} cases ...')

        # TODO
        max_render = 10
        policy_imgs = []
        demo_imgs = []

        demo_failed_imgs, policy_failed_imgs = [], []
        failed_case_counter = 0

        shuffled_inds = np.random.permutation(len(self.test_cases))
        for test_counter, test_idx in tqdm.tqdm(enumerate(shuffled_inds)):
            test_case = self.test_cases[test_idx]
            
            demo_state      = test_case['context_s']
            demo_action     = test_case['context_a']
            new_rst_state   = test_case['rst']
            demo_target     = demo_state[-1, -3:] # eef of the last step

            env = gym.make(self.conf.env_name)

            # render demo policy
            if test_counter < max_render:
                env.reset()
                env.set_target(demo_target)
                s = env.robot_reset_to_qpos(demo_state[0, :7])
                for a in demo_action:
                    demo_imgs.append(self.render(env))
                    env.step(a)

            # set the reset
            env.reset()
            env.set_target(demo_target)
            env.robot_reset_to_qpos(new_rst_state[:7]) # reset qpos
            s = env.get_obs(remove_target=True)

            states = [s]
            actions = []

            # set the target
            goal = self._get_goal(demo_state, demo_action)
            done = False
            step = 0

            success = False
            total_dist = 0
            policy_a, policy_s = [], []
            for _ in demo_action: # since the ep_len is always n it makes sense to do this
                # step through the policy

                a = self._get_action(states, actions, goal)
                # a = env.action_space.sample()
                if test_counter < max_render:
                    policy_imgs.append(self.render(env))
                _, _, done, _ = env.step(a)

                # log
                policy_a.append(a)
                policy_s.append(s)

                s = env.get_obs(remove_target=True)
                step_dist = np.linalg.norm(s[-3:] - demo_state[-1, -3:], ord=1)
                total_dist += step_dist
                
                states.append(s)
                actions.append(a)

                if not success:
                    # TODO: is this the right threshold?
                    success = step_dist < 0.05
                if done:
                    break
                step += 1

            rewards.append(-total_dist)
            successes.append(success)

            if not success and failed_case_counter < max_render:
                # replay both the demo and policy for this index
                env.reset()
                env.set_target(demo_target)
                s = env.robot_reset_to_qpos(demo_state[0, :7])
                for a in demo_action:
                    demo_failed_imgs.append(self.render(env))
                    env.step(a)

                env.reset()
                env.set_target(demo_target)
                s = env.robot_reset_to_qpos(policy_s[0, :7])
                for a in policy_a:
                    policy_failed_imgs.append(self.render(env))
                    env.step(a)

                failed_case_counter += 1
        
        policy_imgs = np.stack(policy_imgs, 0)
        demo_imgs   = np.stack(demo_imgs, 0)

        assert demo_imgs.shape == policy_imgs.shape

        policy_failed_imgs = np.stack(policy_failed_imgs, 0)
        demo_failed_imgs   = np.stack(demo_failed_imgs, 0)
        assert demo_failed_imgs.shape == policy_failed_imgs.shape

        summary = dict(
            success_rate=float(np.mean(successes)),
            total_dist=-float(np.mean(rewards))
        )
        write_yaml(self.output_dir / f'summary_{self.mode}.yaml', summary)
        write_pickle(self.output_dir / f'example_trajs_{self.mode}.pkl', dict(demo=demo_imgs, policy=policy_imgs))

        print(f'success rate: {float(np.mean(successes))}, total_dist: {-float(np.mean(rewards))}')

        print('Plotting all examples ...')
        plot_path = self.output_dir / f'examples_{self.mode}.gif'

        pngs = []
        tmp_dir = mkdtemp()
        for demo_traj, policy_traj in zip(demo_imgs, policy_imgs):
            plt.close()
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(demo_traj)
            plt.title('demo')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(122)
            plt.imshow(policy_traj)
            plt.title('policy')
            plt.xticks([])
            plt.yticks([])

            _, filename = mkstemp(dir=tmp_dir)
            filename += '.png'

            fig.savefig(filename)
            plt.clf() # clear figure
            pngs.append(filename)

        plot_images = [imageio.imread(png) for png in pngs]
        imageio.mimsave(plot_path, plot_images, fps=25)
        print('Plotting done.')

        print('Plotting failed examples ...')
        plot_path = self.output_dir / f'examples_{self.mode}_failed.gif'

        pngs = []
        tmp_dir = mkdtemp()
        for demo_traj, policy_traj in zip(demo_failed_imgs, policy_failed_imgs):
            plt.close()
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(demo_traj)
            plt.title('demo')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(122)
            plt.imshow(policy_traj)
            plt.title('policy')
            plt.xticks([])
            plt.yticks([])

            _, filename = mkstemp(dir=tmp_dir)
            filename += '.png'

            fig.savefig(filename)
            plt.clf() # clear figure
            pngs.append(filename)

        plot_images = [imageio.imread(png) for png in pngs]
        imageio.mimsave(plot_path, plot_images, fps=25)
        print('Plotting done.')


class EvaluatorPickPlaceSawyer(EvaluatorBase):

    @classmethod
    def get_test_cases(cls, test_dataset):
        print('Preparing test cases from test dataset ...')
        test_task_ids = test_dataset.allowed_ids

        # (states, actions, rst_state)
        test_cases = []
        for task_id, var_id in test_task_ids:
            # TODO: grab the first 100
            episodes = test_dataset.raw_data[task_id][var_id][:100]

            for ep_id, ep in enumerate(episodes):
                # random design choice: use the next episode to obtain reset
                rst_idx = (ep_id + 1) % len(episodes)
                test_dict = dict(
                    context_s=ep['state'],
                    context_a=ep['action'],
                    context_init_raw_state=None if "init_raw_state" not in ep else ep["init_raw_state"],
                    target_s=episodes[rst_idx]['state'],
                    target_a=episodes[rst_idx]['action'],
                    target_init_raw_state=None if "init_raw_state" not in episodes[rst_idx] else episodes[rst_idx]["init_raw_state"],
                    rst=episodes[rst_idx]['state'][0],
                    task_id=task_id,
                )
                test_cases.append(test_dict)
        print('Preparation done.')
        return test_cases

    def render(self, env):
        raise NotImplementedError
        return env.unwrapped.sim.render(256, 256, mode="offscreen")[::-1]

    def eval(self):
        successes, episode_lens = [], []
        print(f"Running evaluation on {len(self.test_cases)} {self.mode} cases ...")

        # TODO
        max_render = 2
        max_eval = 2
        policy_imgs = []
        demo_imgs = []
        expert_imgs = []
        env_map = {}

        shuffled_inds = np.random.permutation(len(self.test_cases))
        for test_counter, test_idx in tqdm.tqdm(enumerate(shuffled_inds)):
            if test_counter >= max_eval:
                break
            test_case = self.test_cases[test_idx]

            demo_state = test_case["context_s"]
            demo_action = test_case["context_a"]
            demo_rst_state = test_case["context_init_raw_state"]
            target_rst_state = test_case["target_init_raw_state"]
            target_state = test_case["target_s"]
            target_action = test_case["target_a"]
            task_id = test_case["task_id"]

            if task_id in env_map:
                env = env_map[task_id]
            else:
                env = create_env(task_id, render=False, camera_obs=test_counter < max_render)
                env_map[task_id] = env

            # render demo policy
            if test_counter < max_render:
            #     env.reset()
            #     env.sim.reset()
            #     env.sim.set_state_from_flattened(demo_rst_state) # show demonstration
            #     env.sim.forward()
            #     for a in demo_action:
            #         # obs, reward, done, info = env.step(unstandardize_action(a))
            #         obs, reward, done, info = env.step(a)
            #         demo_imgs.append(obs["image"])

                env.reset()
                env.sim.reset()
                env.sim.set_state_from_flattened(target_rst_state) # show policy
                env.sim.forward()
                for a in target_action:
                    obs, reward, done, info = env.step(a)
                    expert_imgs.append(obs["image"])


            # set the reset
            env.reset()
            env.sim.reset()
            env.sim.set_state_from_flattened(target_rst_state) # reset to target inital state
            env.sim.forward()
            s = target_state[0]

            # set the target
            goal = self._get_goal(demo_state, demo_action, target_state, target_action)
            done = False
            step = 0

            success = False
            # max_steps = 200
            max_steps = 100 # len(target_state)

            # for idx in range(max_steps):  # since the ep_len is always n it makes sense to do this
            for idx in range(max_steps):  # since the ep_len is always n it makes sense to do this
                # step through the policy
                a = self._get_action(s, goal)
                # a = self._get_action(target_state[idx], goal)
                # print('-'*30)
                # print('action delta: ', np.linalg.norm(a[:3] - target_action[idx, :3]))
                # a = env.action_space.sample()
                # action = unstandardize_action(a)
                # HACK (kourosh): fix the rotation of the eef and binarize gripper action based on its sign
                action = a
                action[3:6] = [0.546875, -0.296875,  0.]
                action[-1] = 1 if (action[-1] > 0) and (idx < max_steps - 20) else -1 
                obs, reward, done, info = env.step(action)
                # obs, reward, done, info = env.step(target_action[idx])
                if test_counter < max_render:
                    policy_imgs.append(obs["image"])
                s = env_obs_to_vector(obs, PICK_PLACE_OBSERVATION_KEYS)
                print(f'[{idx}], state: {s[-12:]}')

                # if idx < len(target_state) - 1:
                #     print('state delta: ', np.linalg.norm(s - target_state[idx+1]))
                # foo = [len(obs[k]) for k in PICK_PLACE_OBSERVATION_KEYS]
                # breakpoint()

                success = reward > 0
                if success:
                    break
                step += 1

            # if test_counter < max_render:
            #     diff = step - len(demo_action)
            #     last_image_gray = np.stack((np.mean(demo_imgs[-1], -1),) * 3, axis=-1).astype(int)
            #     for d in range(diff):
            #         demo_imgs.append(last_image_gray)

            #     diff = step - len(target_action)
            #     last_image_gray = np.stack((np.mean(expert_imgs[-1], -1),) * 3, axis=-1).astype(int)
            #     for d in range(diff):
            #         expert_imgs.append(last_image_gray)

            episode_lens.append(idx)
            successes.append(success)

        policy_imgs = np.stack(policy_imgs, 0)
        # demo_imgs   = np.stack(demo_imgs, 0)
        # expert_imgs = np.stack(expert_imgs, 0)
        # assert demo_imgs.shape == policy_imgs.shape

        summary = dict(
            success_rate=float(np.mean(successes)), total_dist=-float(np.mean(episode_lens))
        )

        summary_yaml = self.output_dir / f"summary_{self.mode}.yaml"
        write_yaml(summary_yaml, summary)

        images_dict = dict(demo=demo_imgs, policy=policy_imgs, expert=expert_imgs)
        write_pickle(
            self.output_dir / f"example_trajs_{self.mode}.pkl",
            images_dict,
        )
        for key, imgs in images_dict.items():
            save_path = os.path.join(self.output_dir,f"vis_{key}_{self.mode}.gif")
            if len(imgs) > 0:
                save_as_gif(imgs, save_path)

        print(
            f"success rate: {float(np.mean(successes))}, total_dist: {-float(np.mean(episode_lens))}"
        )
        print(f"Saved to :{self.output_dir}")

class TOsilEvaluator(EvaluatorBase):

    def _get_goal(self, demo_state, demo_action):
        device = self.agent.device
        batch = dict(
            context_s=torch.as_tensor(demo_state).float().to(device),
            context_a=torch.as_tensor(demo_action).float().to(device),
        )
        batch = collate_fn_for_supervised_osil([batch], padding=self.conf.max_padding, pad_targets=self.conf.use_gpt_decoder)
        with torch.no_grad():
            goal = self.agent.get_task_emb(batch['context_s'], batch['context_a'], batch['attention_mask'])
            goal = goal.squeeze(0)

        return goal.detach().cpu().numpy()

    def _get_action(self, state, goal):
        device = self.agent.device
        state_tens = torch.as_tensor(state[None], dtype=torch.float, device=device)
        goal_tens = torch.as_tensor(goal[None], dtype=torch.float, device=device)

        pred_ac = self.agent.decoder(state_tens, goal_tens)
        a = pred_ac.squeeze(0).detach().cpu().numpy()
        return a

class OsilEvaluatorPM(EvaluatorPointMazeBase, TOsilEvaluator): pass
class OsilEvaluatorReacher(EvaluatorReacherSawyer, TOsilEvaluator): pass


