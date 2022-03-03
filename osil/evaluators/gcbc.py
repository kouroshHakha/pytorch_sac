
from osil.evaluators import EvaluatorBase
import torch

class GCBCEvaluator(EvaluatorBase):

    def _get_goal(self, agent, test_case):
        g = test_case['goal']
        # g = test_case['context_s'][-1]
        # g = test_case['target_s'][-1, 4:6]
        return g

    def _get_action(self, agent, state, goal):
        agent.eval()
        device = agent.device
        state_tens = torch.as_tensor(state[None], dtype=torch.float, device=device)
        goal_tens = torch.as_tensor(goal[None], dtype=torch.float, device=device)
        pred_ac = agent(state_tens, goal_tens)
        a = pred_ac.squeeze(0).detach().cpu().numpy()
        return a
