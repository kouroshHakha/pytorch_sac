from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    id='reacher_7dof-v1',
    entry_point='envs.reacher_sawyer:Reacher7DOFEnv',
    max_episode_steps=50,
)

register(
    id='Reacher2D-v1',
    entry_point='envs.reacher2d:ReacherMILEnv',
    kwargs={'train': True},
    max_episode_steps=50,
    reward_threshold=-0.05,
)

register(
    id='Reacher2DTest-v1',
    entry_point='envs.reacher2d:ReacherMILEnv',
    kwargs={'train': False},
    max_episode_steps=50,
    reward_threshold=-0.05,
)


from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from mjrl.envs.reacher_sawyer import Reacher7DOFEnv
