from mas.mas import MetaMAS
from .goal_gcn import GoalGCNMAS

MAS = {
    # Goal-conditioned MAS with GCN communication (recommended)
    'goal-gcn': GoalGCNMAS,
    'goalrl': GoalGCNMAS,
    'gcn': GoalGCNMAS,
}

def get_mas(mas_type: str) -> MetaMAS:

    if MAS.get(mas_type) is None:
        raise ValueError(f'Unsupported mas type: {mas_type}. Available: {list(MAS.keys())}')
    return MAS.get(mas_type)() 