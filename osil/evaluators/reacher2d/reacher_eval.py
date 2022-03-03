
import numpy as np
import torch
from PIL import Image
import tqdm
from pathlib import Path

import envs; import gym
from utils import read_hdf5, write_pickle, write_yaml, save_as_gif
from osil.evaluators import EvaluatorBase


class Reacher2DEvalBase(EvaluatorBase):

    def __init__(self, conf, dset, output_dir='', mode='train'):
        super().__init__(conf, dset, output_dir, mode)
        self._update_test_cases(self.dataset, self.test_cases)
        self.is_image = self.conf.is_image
    
    # implemented below 
    def _update_test_cases(self, test_dataset, test_cases):
        raise NotImplementedError

    def _render(self, env, size=128, channel_first=False):
        image = env.render('rgb_array')
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((size, size), Image.ANTIALIAS)
        image = np.flipud(np.array(pil_image))
        if channel_first:
            image = image.transpose(2, 0, 1).copy()

        return image

    def _get_test_cases(self, test_dataset):
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
                    target_task_id = (var_id, episodes[rst_idx]['cond_id'].item()),
                    context_task_id = (var_id, ep['cond_id'].item())
                )
                test_cases.append(test_dict)
        print('Preparation done.')
        return test_cases

    def _save_outputs(self, summary, images_dict):

        if self.output_dir:
            write_yaml(Path(self.output_dir) / f'summary_{self.mode}.yaml', summary)
            write_pickle(
                Path(self.output_dir) / f"example_trajs_{self.mode}.pkl",
                images_dict,
            )
            for key, imgs in images_dict.items():
                save_path = Path(self.output_dir) / f"vis_{key}_{self.mode}.gif"
                if len(imgs) > 0:
                    save_as_gif(imgs, save_path)

            print(f'success rate: {summary["success_rate"]}, total_reward: {summary["total_reward"]}')
            print(f"Saved to :{self.output_dir}")

    # mandatory functions
    def eval(self, agent):
        successes, rewards = [], []
        max_render = self.conf.get('max_render', 0)
        max_eval_episodes = self.conf.get('max_eval_episodes', len(self.test_cases))
        seed = self.conf.get('seed', 0)
        verbose = self.conf.get('verbose', False)

        if verbose:
            print(f'Running evaluation on {len(self.test_cases)} {self.mode} cases ...')

        if not self.output_dir:
            # don't save anything 
            max_render = -1

        policy_imgs = []
        demo_imgs = []
        expert_imgs = []

        np.random.seed(seed)
        shuffled_inds = np.random.permutation(len(self.test_cases))
        env = gym.make(self.conf.env_name)

        test_iter = enumerate(shuffled_inds[:max_eval_episodes])
        test_iter = tqdm.tqdm(test_iter) if verbose else test_iter
        for test_counter, test_idx in test_iter:
            test_case = self.test_cases[test_idx]
            
            demo_state      = test_case['context_s']
            demo_action     = test_case['context_a']
            demo_task_id, demo_cond_id = test_case['context_task_id']
            
            new_rst_state   = test_case['rst']
            target_task_id, target_cond_id = test_case['target_task_id']

            target_action   = test_case['target_a']

            
            # render demo policy
            if test_counter < max_render:
                env.set_task(demo_task_id, demo_cond_id)
                env.reset()
                env.set_state(demo_state[0, :2], demo_state[0, 2:4])
                for a in demo_action:
                    demo_imgs.append(self._render(env))
                    env.step(a)

                env.set_task(target_task_id, target_cond_id)
                env.reset()
                env.set_state(new_rst_state[:2], new_rst_state[2:4])
                for a in target_action:
                    expert_imgs.append(self._render(env))
                    env.step(a)


            # set the reset
            env.set_task(target_task_id, target_cond_id)
            env.reset()
            s = env.set_state(new_rst_state[:2], new_rst_state[2:4])

            if self.is_image:
                s = self._render(env, agent.conf.obs_shape[-1], channel_first=True)
            
            # set the target
            goal = self._get_goal(agent, test_case)
            step = 0

            success = False
            total_reward = 0
            policy_a, policy_s = [], []
            for _ in demo_action: # since the ep_len is always n it makes sense to do this
                # step through the policy
                a = self._get_action(agent, s, goal)
                if test_counter < max_render:
                    policy_imgs.append(self._render(env))
                ns, reward, _, _ = env.step(a)

                # log
                policy_a.append(a)
                policy_s.append(s)

                s = ns

                if self.is_image:
                    s = self._render(env, agent.conf.obs_shape[-1], channel_first=True)

                total_reward += reward
                
                if bool(reward):
                    success = True
                step += 1

            rewards.append(total_reward)
            successes.append(success)

        summary = dict(
            success_rate=float(np.mean(successes)),
            total_reward=float(np.mean(rewards))
        )

        images_dict = dict(demo=demo_imgs, policy=policy_imgs, expert=expert_imgs)
        self._save_outputs(summary, images_dict)
        
        
        return summary

class EvaluatorReacher2DState(Reacher2DEvalBase):

    def _update_test_cases(self, test_dataset, test_cases):
        test_task_ids = test_dataset.allowed_ids
        test_case_idx = 0
        for task_id, var_id in test_task_ids:
            episodes = test_dataset.raw_data[task_id][var_id]
            for ep in episodes:
                test_cases[test_case_idx].update(goal=ep['target'])
                test_case_idx += 1
    
        return test_cases

class EvaluatorReacher2DGoalImg(Reacher2DEvalBase):

    def _update_test_cases(self, test_dataset, test_cases):
        data_path = test_dataset.data_path
        test_task_ids = test_dataset.allowed_ids
        transform = test_dataset.transform
        test_case_idx = 0
        for task_id, var_id in test_task_ids:
            episodes = test_dataset.raw_data[task_id][var_id]
            for ep_id, ep in enumerate(episodes):
                rst_idx = (ep_id + 1) % len(episodes)
                # goal_img = torch.load(data_path / 'torch_imgs' / f'{var_id}_{ep["cond_id"].item()}.torch')[-1]
                goal_img = torch.load(data_path / 'torch_imgs' / f'{var_id}_{episodes[rst_idx]["cond_id"].item()}.torch')[-1]
                goal_img = transform(goal_img.permute(2, 0, 1)).numpy()
                test_cases[test_case_idx].update(goal=goal_img)
                test_case_idx += 1

        return test_cases