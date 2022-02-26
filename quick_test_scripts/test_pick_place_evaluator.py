
from osil.eval import EvaluatorPickPlaceSawyer
from argparse import Namespace
import numpy as np
from pathlib import Path

from osil.debug import register_pdb_hook
register_pdb_hook()

from quick_test_scripts.train_gcbcv2 import GCBCDataset

class DummyEval(EvaluatorPickPlaceSawyer):

    def _get_action(self, state, goal):
        # rand between -1, 1
        return np.random.rand(7) * 2 - 1

    def _get_goal(self, demo_state, demo_action):
        return None


dataset_path = 'robosuite_pick_place_osil_data'
env_name = 'robosuite_pick_place'
pargs = Namespace(
    dataset_path=dataset_path,
    env_name=env_name,
    gd=-1,
)

test_dset = GCBCDataset(data_path=pargs.dataset_path, mode='test', nshots_per_task=100, env_name=pargs.env_name)

output_path = Path('./test_p&p')
output_path.mkdir(parents=True, exist_ok=True)
evaluator = DummyEval(pargs, agent=None, output_dir=output_path, test_dset=test_dset, mode='test')
evaluator.eval()