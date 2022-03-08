
from osil.evaluators.tosil import TOsilEvaluator
from osil.evaluators.gcbc import GCBCEvaluator
from osil.evaluators.reacher2d.reacher_eval import (
    EvaluatorReacher2DState, EvaluatorReacher2DGoalImg
)
        
# state
class EvaluatorReacher2D_GCBC_State(GCBCEvaluator, EvaluatorReacher2DState): pass
class EvaluatorReacher2D_TOSIL_State(TOsilEvaluator, EvaluatorReacher2DState): pass

# image
class EvaluatorReacher2D_GCBC_Img(GCBCEvaluator, EvaluatorReacher2DGoalImg): pass
