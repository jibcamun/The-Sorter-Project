try:
    from ..models import SorterState
except:
    from models import SorterState

def compute_reward(state: SorterState, numeric: float, feedback:str):
    state.reward[0].append(numeric)
    state.reward[1].append(feedback)