
from osil.evaluators import EvaluatorBase
import torch
import numpy as np

class GCBCEvaluator(EvaluatorBase):

    def _get_goal(self, agent, test_case):
        g = test_case['goal']
        # g = test_case['context_s'][-1]
        # g = test_case['target_s'][-1, 4:6]
        return g

    def _get_action(self, agent, state_list, goal):

        n_stack_frames = self.dataset.n_stack_frames
        
        # stack last n frames if needed (and zero pad with zeros)
        if n_stack_frames == 1:
            state = state_list[-1]
        elif n_stack_frames > 1:
            if len(state_list) < n_stack_frames:
                padding_shape = state_list[-1].shape
                cat_list = [np.zeros(padding_shape, dtype=np.uint8) for _ in range(n_stack_frames - len(state_list))]
                cat_list += state_list
            else:
                cat_list = state_list[-n_stack_frames:]
            state = np.concatenate(cat_list, 0)


        agent.eval()
        device = agent.device
        state_tens = torch.as_tensor(state[None], dtype=torch.float, device=device)
        goal_tens = torch.as_tensor(goal[None], dtype=torch.float, device=device)
        # pred_ac = agent(state_tens, goal_tens)
        pred_ac = agent.get_action(state_tens, goal_tens)
        a = pred_ac.squeeze(0).detach().cpu().numpy()
        return a
