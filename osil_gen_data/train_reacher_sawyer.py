


import warnings

from snapshot import load_snapshot

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import numpy as np
import torch
from dmc import ExtendedTimeStep, StepType

from dmc import specs # used to make the interface compatible with replay buffer

import envs; import gym
import utils
from logger import Logger
from replay_buffer3 import ReplayBuffer
from video import VideoRecorder
from snapshot import load_snapshot, save_snapshot

torch.backends.cudnn.benchmark = True

from osil.debug import register_pdb_hook
register_pdb_hook()

snapshot_steps = [1, 25000, 50000, 100000, 150000, 200000, 250000, 500000, 1000000, 1500000, 2000000]

def get_step_type(episode_step, done):
    if done:
        return StepType.LAST
    if episode_step == 0:
        return StepType.FIRST
    return StepType.MID


@hydra.main(config_path='..', config_name='config')
def main(cfg):
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb)

    # create envs
    train_env = gym.make(cfg.task)
    eval_env = gym.make(cfg.task)

    obs_space = train_env.observation_space
    act_space = train_env.action_space
    s = train_env.reset()
    observation_spec = specs.Array(obs_space.shape, np.float32, 'observation')
    action_spec = specs.Array(act_space.shape, np.float32, 'action')
    env_specs = (
        observation_spec, # obs
        action_spec, # act
        specs.Array((1,), act_space.dtype, 'reward'), # reward
        specs.Array((1,), act_space.dtype, 'discount'), # discount
        specs.Array(obs_space.shape, np.float32, 'next_observation'), # obs
    )

    # create replay buffer
    replay_buffer = ReplayBuffer(specs=env_specs,
                                 max_size=cfg.replay_buffer_size,
                                 batch_size=cfg.batch_size,
                                 nstep=cfg.nstep,
                                 discount=cfg.discount)

    #self.replay_loader = make_replay_loader(self.work_dir / 'buffer',
    #                                        cfg.replay_buffer_size,
    #                                        cfg.batch_size,
    #                                        cfg.replay_buffer_num_workers,
    #                                        cfg.save_snapshot, cfg.nstep,
    #                                        cfg.discount)
    replay_iter = None

    video = VideoRecorder(work_dir if cfg.save_video else None, camera_id=0, is_mujoco=True)

    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_dim=observation_spec.shape[0],
        action_dim=action_spec.shape[0]
    )

    timer = utils.Timer()

    def eval(step, episode):
        eval_return = 0
        for i in range(cfg.num_eval_episodes):
            obs = eval_env.reset()
            done = False
            video.init(eval_env, enabled=True)
            while not done:
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(obs,
                                       step,
                                       eval_mode=True)
                obs, reward, done, _ = eval_env.step(action)
                video.record(eval_env)
                eval_return += reward

            video.save(f'{step}_{i}.gif')

        with logger.log_and_dump_ctx(step, ty='eval') as log:
            log('episode_return', eval_return / cfg.num_eval_episodes)
            log('episode', episode)
            log('total_time', timer.total_time())

    episode, episode_step, episode_return = 0, 0, 0
    obs = train_env.reset()
    done = False
    metrics = None
    for step in range(cfg.num_train_steps + 1):
        if done:
            episode += 1
            if metrics is not None:
                elapsed_time, total_time = timer.reset()
                with logger.log_and_dump_ctx(step, ty='train') as log:
                    log('fps', episode_step / elapsed_time)
                    log('total_time', total_time)
                    log('episode_return', episode_return)
                    log('episode', episode)
                    log('buffer_size', len(replay_buffer))

            obs = train_env.reset()
            done = False
            episode_step, episode_return = 0, 0

        if step + 1 in snapshot_steps:
            save_snapshot(agent, step + 1, logger._log_dir)

        if step % cfg.eval_every_steps == 0:
            eval(step, episode)

        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, step, eval_mode=False)

        if step >= cfg.num_seed_steps:
            if replay_iter is None:
                replay_iter = iter(replay_buffer)
            metrics = agent.update(replay_iter, step)
            logger.log_metrics(metrics, step, ty='train')

        next_obs, reward, done, _ = train_env.step(action)
        episode_return += reward
        # make the interface compatible by defining a TimeStep object
        time_step = ExtendedTimeStep(
            observation=obs.astype('float32'),
            next_observation=next_obs.astype('float32'),
            state=obs.astype('float32'),
            next_state=next_obs.astype('float32'),
            step_type=get_step_type(episode_step, done),
            action=action.astype('float32'),
            reward=reward.astype('float32'),
            # timestep discount should be 1?? cuz replay buffer takes care of cfg.discount already
            discount=1.0, ## ??
        )
        obs = next_obs
        replay_buffer.add(time_step)
        episode_step += 1


if __name__ == '__main__':
    main()
