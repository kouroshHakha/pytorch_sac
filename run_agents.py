from collections import defaultdict
import time
import warnings
import numpy as np
import pickle

from argparse import ArgumentParser

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import torch
import tqdm

import dmc, utils
from video import VideoRecorder
from snapshot import load_snapshot

torch.backends.cudnn.benchmark = True


def _parse_args():

    parser = ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--snapshot', type=str)
    parser.add_argument('--num_episodes', default=1, type=int)
    parser.add_argument('--output_dir', default='agent_runs', type=str)
    parser.add_argument('--only_states', action='store_true')

    return parser.parse_args()

def main(cfg):
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.seed)
    eval_env = dmc.make(cfg.task, cfg.seed, obs_type='states' if cfg.only_states else 'both')

    video = VideoRecorder(Path(cfg.output_dir), camera_id=2 if cfg.task.startswith('quadruped') else 0)

    # load snapshot
    agent = load_snapshot(cfg.snapshot)

    def eval_step(time_step):
        with torch.no_grad(), utils.eval_mode(agent):
            # hard code step to num_expl_steps to never run exploration acting
            action = agent.act(time_step.observation if cfg.only_states else time_step.state, agent.num_expl_steps, eval_mode=True)
            time_step = eval_env.step(action)
            return time_step

    episodes = defaultdict(lambda: [])
    trange = tqdm.trange(cfg.num_episodes)
    for ep in trange:
        time_step = eval_env.reset()
        episode = defaultdict(lambda: [])
        # only store the video of the first episode as sanity check
        video.init(eval_env, enabled=False if cfg.only_states else (ep==0))
        while not time_step.last():
            video.record(eval_env)
            time_step = eval_step(time_step)

            episode['state'].append(time_step.observation if cfg.only_states else time_step.state)
            episode['action'].append(time_step.action)
            episode['reward'].append(np.array([time_step.reward]))
            if not cfg.only_states:
                episode['obs'].append(time_step.observation)
                episode['next_obs'].append(time_step.next_observation)
                episode['next_state'].append(time_step.next_state)
                episode['terminal'].append(np.array([time_step.last()]))

            # episode_reward += time_step.reward

        episode = {k: np.stack(v) for k, v in episode.items()}

        trange.set_description(f'[episode {ep+1}] ep_reward: {episode["reward"].sum():10.2f}')
        fpath = video.save_dir / f'{cfg.task}/ep_{ep}.mp4'
        fpath.parent.mkdir(parents=True, exist_ok=True)
        video.save(f'{cfg.task}/ep_{ep}.mp4')

        for k in episode:
            episodes[k].append(episode[k])
    
    # dict(obs, state, action, next_obs, next_state, terminal) each item is an array (episodes, tstep, *obj_shape)
    episodes = {k: np.stack(v) for k, v in episodes.items()}
    with open(Path(cfg.output_dir) / f'{cfg.task}_{cfg.num_episodes}.pickle', 'wb') as f:
        pickle.dump(episodes, f)

if __name__ == '__main__':
    main(_parse_args())
