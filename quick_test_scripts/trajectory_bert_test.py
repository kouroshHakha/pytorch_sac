import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from osil.nets import TrajBERT
from osil.utils import ParamDict
from osil.debug import register_pdb_hook
register_pdb_hook()

hidden_dim = 128
seq_len = 32
seed = 10

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

config = ParamDict(
    hidden_dim=128,
    obs_shape=(1,),
    ac_dim=1,
    max_ep_len=1024,
)

model = TrajBERT(config)

states = (1 + torch.arange(2*5*1)).reshape(2, 5, 1).float()
actions = (torch.arange(2*5*1) * 1000 + 1).reshape(2, 5, 1).float()
masks = torch.tensor([[1, 0, 0, 0, 1], [1, 0, 1, 0, 1]]).long()
traj_emb = model(states, actions, masks)

breakpoint()