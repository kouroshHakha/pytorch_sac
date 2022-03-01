
import envs
import gym

from osil.debug import register_pdb_hook
register_pdb_hook()

import pprint

if __name__ == '__main__':

    env = gym.make('Reacher2D-v1')
    obs = env.reset()
    obs_dict = env.get_obs_dict()
    pprint.pprint(obs_dict)
    breakpoint()