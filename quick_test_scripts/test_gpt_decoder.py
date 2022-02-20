
import torch
from osil.nets import GCTrajGPT
from osil.debug import register_pdb_hook
register_pdb_hook()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

B = 2
T = 3
obs_dim = 3
ac_dim = 2
goal_dim = 2

config = dict(
    hidden_dim=128,
    obs_shape=(obs_dim, ),
    ac_dim=ac_dim,
    goal_dim=goal_dim,
    max_ep_len=64,

)
model = GCTrajGPT(config)
model.to(device)
model.eval()

states = (1 + torch.arange(B * T * obs_dim)).reshape(B, T, obs_dim).to(device).float()
actions = ((1 + torch.arange(B * T * ac_dim)) * 10).reshape(B, T, ac_dim).to(device).float()
goals = ((1 + torch.arange(B * goal_dim)) * (-10)).reshape(B, goal_dim).to(device).float()
attn_mask = torch.ones(B, T).to(device).long()

output = model(states, actions, attn_mask, goals)
state_embs, action_embs = model.get_state_action_embs(states, actions, attn_mask, goals)
ret = model.ff(states, actions, attn_mask, goals, compute_loss=True)

# testing next action prediction
act = model.get_action(states[0], actions[0, :-1], goal=goals[0])
breakpoint()