import numpy as np
from scipy.spatial.distance import cdist
from gym import utils
from PIL import Image
import gc
import glob
import os
from gym.envs.mujoco import mujoco_env
import mujoco_py
# from mujoco_py.mjlib import mjlib
# try:
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))
from pathlib import Path

from mujoco_py import MjViewer

class ReacherMILEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, train=True):
        # gc.enable() # automatic garbage collection
        if not train:
            xml_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "assets/sim_vision_reach_test_xmls/*")))
        else:
            xml_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "assets/sim_vision_reach_train_xmls/*")))
        
        self.n_distractors = 2

        self.xml_paths = {}
        for path in xml_paths:
            xml_fname = Path(path).stem
            res = xml_fname.split('_')
            task_id = int(res[1])
            cond_id = int(res[3])
            self.xml_paths[(task_id, cond_id)] = path

        self.xml_idx = (0, 0)
        self.consecutive_steps = 0
        self.set_task(*self.xml_idx)
        utils.EzPickle.__init__(self)


    def set_task(self, task_id, cond_id, reset_cube_order=True):
        # hard coding ids since their name is not available in xml
        # this will work as long as the xml files are consistent in order
        target_geom_id = 4
        cube0_geom_id = 5
        cube1_geom_id = 6

        self.xml_idx = (task_id, cond_id)
        self._setup_mj_from_xml(self.xml_paths[self.xml_idx])

        self.eept_vel = np.zeros_like(self.get_body_com("fingertip"))
        self.sim.forward()

        # upon reset we shuffle the cubes and update the target_idx
        cubes = [
            self.get_body_com('cube_0')[:2].copy(),
            self.get_body_com('cube_1')[:2].copy(),
            self.get_body_com('target')[:2].copy(),
        ]

        colors = [
            self.model.geom_rgba[cube0_geom_id][:3].copy(),
            self.model.geom_rgba[cube1_geom_id][:3].copy(),
            self.model.geom_rgba[target_geom_id][:3].copy(),
        ]

        if reset_cube_order:
            self.cube_order = np.random.permutation(3)
        
        try:
            self.cubes = [cubes[i] for i in self.cube_order]
            self.cube_colors = [colors[i] for i in self.cube_order]
        except AttributeError:
            raise ValueError('Reset Cube order should be True for the first time call.')

        mujoco_env.MujocoEnv.__init__(self, self.xml_paths[self.xml_idx], 5)

        self.target_idx = np.where(self.cube_order == 2)[0][0]

    def get_xml(self):
        return self.xml_paths[self.xml_idx]

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

        return self._get_obs()

    def step(self, a):
        prev_eept = self.get_body_com("fingertip")
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # reward = reward_dist# + reward_ctrl

        if np.linalg.norm(vec) < 0.05:
            self.consecutive_steps += 1
        else:
            self.consecutive_steps = 0
        
        reward = int(self.consecutive_steps >= 10)
        self.do_simulation(a, self.frame_skip)
        # self.model.forward()
        curr_eept = self.get_body_com("fingertip")
        self.eept_vel = (curr_eept - prev_eept) / self.dt
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
    
    def get_obs_dict(self):
        return dict(
            qpos=self.data.qpos.flat[:2],
            qvel=self.data.qvel.flat[:2],
            eef_pos=self.get_body_com("fingertip")[:2],
            eef_vel=self.eept_vel[:2],
            cube_0_pos=self.cubes[0],
            cube_0_rgb=self.cube_colors[0],
            cube_1_pos=self.cubes[1],
            cube_1_rgb=self.cube_colors[1],
            cube_2_pos=self.cubes[2],
            cube_2_rgb=self.cube_colors[2],
            target_color=self.cube_colors[self.target_idx], # target
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.
        self.viewer.cam.lookat[1] = 0.
        self.viewer.cam.lookat[2] = 0.
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = 90 #-90
        self.viewer.cam.azimuth = 90

    def reset_model(self):
        self.set_task(*self.xml_idx, reset_cube_order=False)
        self.consecutive_steps = 0
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[:2],
            self.data.qvel.flat[:2],
            self.get_body_com("fingertip")[:2],
            self.eept_vel[:2],
            self.cubes[0],
            self.cube_colors[0],
            self.cubes[1],
            self.cube_colors[1],
            self.cubes[2],
            self.cube_colors[2],
        ])
    
    def _setup_mj_from_xml(self, model_path):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not Path(fullpath).exists():
            raise IOError("File %s does not exist" % fullpath)

        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
