from collections import defaultdict

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

