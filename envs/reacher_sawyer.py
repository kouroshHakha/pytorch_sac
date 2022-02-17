from mimetypes import init
import numpy as np
from gym import utils, spaces
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer

# this is a modified reacher to take out target_pos from the state 
class Reacher7DOFEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.hand_sid = -2
        self.target_sid = -1
        mujoco_env.MujocoEnv.__init__(self, 'sawyer.xml', 4)
        utils.EzPickle.__init__(self)
        self.hand_sid = self.model.site_name2id("finger")
        self.target_sid = self.model.site_name2id("target")


        # high = np.array([0.3, 0.2, 0.25])
        high = np.array([0.5, 0.2, 0.4])
        low = np.array([-0.5, -0.25, -0.4]) # plotted some random trajectories to get these numbers for hand_pos visistation distribution
        self.target_space = spaces.Box(low, high, dtype=np.float32)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self.get_obs()
        reward = self.get_reward(obs, a)
        return obs, reward, False, self.get_env_infos()

    def get_obs(self, remove_target=False, remove_joint_info=False):
        # remove_target and remove_joint_info will only remove the corresponding extra infos 
        obs_list = []

        if not remove_joint_info:
            obs_list += [
                self.data.qpos.flat,
                self.data.qvel.ravel() * self.dt,       # delta_x instead of velocity
            ]
        obs_list.append(self.data.site_xpos[self.hand_sid])
        if not remove_target:
            obs_list.append(self.data.site_xpos[self.target_sid])
        return np.concatenate(obs_list)

    def get_target(self):
        return self.data.site_xpos[self.target_sid]

    def get_reward(self, obs, act=None):
        # compatible with get_obs()
        obs = np.clip(obs, -10.0, 10.0)
        if len(obs.shape) == 1:
            # vector obs, called when stepping the env
            hand_pos = obs[-6:-3]
            target_pos = obs[-3:]
            target_pos = self.get_target()
            l1_dist = np.sum(np.abs(hand_pos - target_pos))
            l2_dist = np.linalg.norm(hand_pos - target_pos)
        else:
            # raise ValueError('We have removed target from state representation')
            obs = np.expand_dims(obs, axis=0) if len(obs.shape) == 2 else obs
            hand_pos = obs[:, :, -6:-3]
            target_pos = obs[:, :, -3:]
            l1_dist = np.sum(np.abs(hand_pos - target_pos), axis=-1)
            l2_dist = np.linalg.norm(hand_pos - target_pos, axis=-1)
        reward = - l1_dist - 5.0 * l2_dist
        return reward

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        rewards = self.get_reward(obs)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()

    # --------------------------------
    # resets and randomization
    # --------------------------------
    def robot_reset(self):
        # init_state = self.observation_space.sample()
        # qpos = init_state[:7]
        # qvel = init_state[7:14]
        # qpos = self.init_qpos + np.random.uniform(-0.2, 0.2, size=self.init_qpos.shape)
        low_qpos  = np.array([-1.5, -0.5, -1.5, -2, -1.5, -1, -1])
        high_qpos = np.array([ 1.5,  0.5,  1.5, -1,  1.5,  0,  1])
        qpos = np.random.uniform(0, 1, size=low_qpos.shape) * (high_qpos - low_qpos) + low_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    def target_reset(self):
        # target_pos = np.array([0.1, 0.1, 0.1])
        # target_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
        # target_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
        # target_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        # self.model.site_pos[self.target_sid] = target_pos
        self.model.site_pos[self.target_sid] = self.target_space.sample()
        self.sim.forward()

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        self.target_reset()
        return self.get_obs()

    #### new functions for reseting the env to the correct state
    def robot_reset_to_qpos(self, qpos):
        # reset to the given qpos with zero velocity, return the updated state 
        # (end effector position is derived)
        self.set_state(qpos, self.init_qvel)
        state = self.get_obs()
        return state

    def set_target(self, target_pos):
        self.model.site_pos[self.target_sid] = target_pos
        self.sim.forward()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.data.site_xpos[self.target_sid].copy()
        hand_pos = self.data.site_xpos[self.hand_sid].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos, hand_pos=hand_pos)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        self.model.site_pos[self.target_sid] = target_pos
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 2.0
