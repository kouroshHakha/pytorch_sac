
import torch
from osil.evaluators import EvaluatorBase
from osil.data import collate_fn_for_supervised_osil
import numpy as np

class TOsilEvaluator(EvaluatorBase):

    def _get_goal(self, agent, test_cases):
        demo_state = test_cases['context_s']
        demo_action = test_cases['context_a']

        device = agent.device
        batch = dict(
            context_s=torch.as_tensor(demo_state).float().to(device),
            context_a=torch.as_tensor(demo_action).float().to(device),
        )
        batch = collate_fn_for_supervised_osil([batch], padding=self.conf.max_padding, pad_targets=self.conf.use_gpt_decoder)
        with torch.no_grad():
            goal = agent.get_task_emb(batch['context_s'], batch['context_a'], batch['attention_mask'])
            goal = goal.squeeze(0)

        return goal.detach().cpu().numpy()

    def _get_action(self, agent, state, goal):
        device = agent.device
        state_tens = torch.as_tensor(state[None], dtype=torch.float, device=device)
        goal_tens = torch.as_tensor(goal[None], dtype=torch.float, device=device)

        pred_ac = agent.decoder(state_tens, goal_tens)
        a = pred_ac.squeeze(0).detach().cpu().numpy()
        return a
