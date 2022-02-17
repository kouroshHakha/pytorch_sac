from pprint import pprint
import envs
import gym
import imageio

import numpy as np
import matplotlib.pyplot as plt


def plot_hand_poses(hand_poses, suffix='', prefix='hand_poses'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(hand_poses[:, 0], hand_poses[:, 1], hand_poses[:, 2])
    plt.savefig(f'{prefix}_3d_{suffix}.png')

    plt.close()
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(hand_poses[:, 0], density=True, bins=200)
    plt.savefig(f'{prefix}_x_{suffix}.png')

    plt.close()
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(hand_poses[:, 1], density=True, bins=200)
    plt.savefig(f'{prefix}_y_{suffix}.png')


    plt.close()
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(hand_poses[:, 2], density=True, bins=200)
    plt.savefig(f'{prefix}_z_{suffix}.png')




### test reacher sawyer env by rendering it 
env = gym.make('reacher_7dof-v1')

# imgs = []
# traj_rews = []
# for it in range(30):
#     s = env.reset()
#     rewards = []
#     for step in range(50):
#         img = env.unwrapped.sim.render(256, 256, mode='offscreen')
#         imgs.append(img[::-1])
#         a = env.action_space.sample()
#         s, r, d, _ = env.step(a)
#         hand_pos = env.get_env_infos()['state']['hand_pos']
#         rewards.append(r)
#         if d:
#             break
    
#     traj_rews.append(sum(rewards))

# print(traj_rews)
# imageio.mimsave('img.gif', imgs, fps=25)

# # plot distribution of resets and goals of the env upon reset
# hand_poses = []
# target_poses = []
# for it in range(10000):
#     s = env.reset()
#     state = env.get_env_infos()['state']

#     hand_poses.append(state['hand_pos'])
#     target_poses.append(state['target_pos'])


# hand_poses = np.stack(hand_poses, 0)
# target_poses = np.stack(target_poses, 0)

# # plot_hand_poses(hand_poses)
# # plot_hand_poses(target_poses, prefix='target_poses')

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(hand_poses[:, 0], hand_poses[:, 1], hand_poses[:, 2], alpha=0.1, label='hand')
# ax.scatter3D(target_poses[:, 0], target_poses[:, 1], target_poses[:, 2], alpha=0.1, label='target')
# plt.legend()
# plt.savefig(f'hand_and_target_poses_3d.png')


# # plot distribution of ee_pos by randomly taking actions to see if there is an unsupported region with random init
# hand_poses = []
# for t in range(10):
#     env.reset()
#     for _ in range(1000):
#         hand_pos = env.get_env_infos()['state']['hand_pos']
#         hand_poses.append(hand_pos)
#         env.step(env.action_space.sample())

# hand_poses = np.stack(hand_poses, 0)

# plot_hand_poses(hand_poses, 'trajs')


# env.reset()

# for _ in range(10):

#     env.step(env.action_space.sample())


possible_qposes = [env.init_qpos]
for _ in range(10000):
    env.reset()
    # idx = np.random.randint(len(possible_qposes))
    # s = env.robot_reset_to_qpos(possible_qposes[idx])
    s = env.robot_reset_to_qpos(possible_qposes[-1])
    for _ in range(10):
        s, _, _, _ = env.step(env.action_space.sample())
    possible_qposes.append(s[:7])

possible_qposes = np.stack(possible_qposes, 0)


hand_poses = []
for qpos in possible_qposes:
    s = env.robot_reset_to_qpos(qpos)
    hand_pos = env.get_env_infos()['state']['hand_pos']
    hand_poses.append(hand_pos)

hand_poses = np.stack(hand_poses, 0)
plot_hand_poses(hand_poses, prefix='hand_poses', suffix='v2')

for i in range(7):
    plt.close()
    plt.hist(possible_qposes[:, i], bins=200)
    plt.savefig(f'qpos_{i}.png')
breakpoint()