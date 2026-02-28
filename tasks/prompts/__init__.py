from .alfworld_prompt import alfworld_solver_system_prompt, alfworld_few_shots
from .sciworld_prompt import sciworld_solver_system_prompt, sciworld_few_shots

def get_dataset_system_prompt(task: str, task_config: dict) -> str:
    prompt_map: dict = {
        'alfworld': alfworld_solver_system_prompt,
        'sciworld': sciworld_solver_system_prompt,
    }

    if prompt_map.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return prompt_map.get(task)


def get_task_few_shots(dataset: str, task_config: dict, few_shots_num: int) -> list[str]:
    
    if dataset == 'alfworld':
        task_type = task_config.get('task_type')
        if task_type is None:
            raise ValueError('The task config must have the `task_type` attribute.')
        return [alfworld_few_shots[f'react_{task_type}_2'], alfworld_few_shots[f'react_{task_type}_0']][:few_shots_num]
    
    elif dataset == 'sciworld':
        all_shots = list(sciworld_few_shots.values())
        return all_shots[:few_shots_num]
    
    else:
        raise ValueError(f'Unsupported dataset type: {dataset}')

