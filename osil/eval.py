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

            
def evaluate(agent, cfg, output_dir=''):

    env = dmc.make(cfg.task_name, cfg.seed, obs_type=cfg.obs_type)
    # TODO
    output_dir = Path(output_dir) if output_dir else Path(cfg.ckpt).parent
    if not output_dir:
        raise ValueError('Output dir is empty.')
    video = VideoRecorder(output_dir, camera_id=2 if cfg.task_name.startswith('quadruped') else 0)

    def eval_step(time_step):
        with torch.no_grad(), utils.eval_mode(agent):
            obs = torch.from_numpy(time_step.observation).float().to(agent.device)
            obs = obs.unsqueeze(0)
            action = agent(obs)[0]
            time_step = env.step(action.detach().cpu().numpy())
            return time_step

    returns = []
    trange = tqdm.trange(cfg.num_eval_episodes)
    for ep in trange:
        time_step = env.reset()
        ep_rews = []
        video.init(env, enabled=True)

        while not time_step.last():
            video.record(env)
            time_step = eval_step(time_step)
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

    with open(output_dir / 'summary.yaml', 'w') as f:
        yaml.dump(summary_dict, f)
