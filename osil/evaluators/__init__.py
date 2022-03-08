
import abc

class EvaluatorBase(abc.ABC):

    def __init__(self, conf, dset, output_dir='', mode='train'):
        self.conf = conf
        self.output_dir = output_dir
        self.dataset = dset
        self.mode = mode
        self.test_cases = self._get_test_cases(dset)

    @abc.abstractmethod
    def _get_test_cases(self):
        pass
    
    @abc.abstractmethod
    def eval(self, agent):
        pass

    # optional methods to implement
    def _get_goal(self, agent, test_cases):
        raise NotImplementedError

    def _get_action(self, agent, state, goal):
        raise NotImplementedError
    