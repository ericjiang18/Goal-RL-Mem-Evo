import json

from .base_env import BaseEnv, BaseRecorder
from .alfworld_env import AlfworldEnv, AlfworldRecorder, get_env_name_from_gamefile, prefixes
from .fever_env import FeverEnv, FeverRecorder
from .pddl_env.pddl_env import PDDLEnv, PDDLRecorder, get_all_environment_configs

TASKS_PATH = {
    'alfworld': 'data/alfworld/alfworld_tasks_suffix.json',
    'fever': 'data/fever/fever_dev.jsonl',
    'pddl': 'data/pddl/test.jsonl'  
}

# The task data is now loaded on-demand inside the get_task function
# to prevent import-time errors when running a specific task.

def _get_envs():
    """Lazy env registry to avoid importing SciWorld when not needed."""
    envs = {
        'alfworld': AlfworldEnv,
        'fever': FeverEnv,
        'pddl': PDDLEnv,
    }
    try:
        from .sciworld_env import SciworldEnv
        envs['sciworld'] = SciworldEnv
    except ImportError:
        pass
    return envs

def _get_recorders():
    """Lazy recorder registry to avoid importing SciWorld when not needed."""
    recorders = {
        'alfworld': AlfworldRecorder,
        'fever': FeverRecorder,
        'pddl': PDDLRecorder,
    }
    try:
        from .sciworld_env import SciworldRecorder
        recorders['sciworld'] = SciworldRecorder
    except ImportError:
        pass
    return recorders


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    envs = _get_envs()
    if envs.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return envs.get(task)(env_config, max_trials)

def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    recorders = _get_recorders()
    if recorders.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return recorders.get(task)(working_dir=working_dir, namespace=namespace)

def get_task(task: str) -> list[dict]:
    
    if task == 'alfworld':
        import os
        import glob
        
        # Get ALFWorld data path from environment or use default
        alfworld_data = os.environ.get('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
        
        # Use the eval_out_of_distribution split (valid_unseen)
        games_dir = os.path.join(alfworld_data, 'json_2.1.1', 'valid_unseen')
        
        # Find all game.tw-pddl files (they are in subdirectories like task_type/trial_xxx/game.tw-pddl)
        game_files = glob.glob(os.path.join(games_dir, '*', '*', 'game.tw-pddl'))
        
        tasks = []
        for gamefile in sorted(game_files):
            # Extract task type from parent directory name (e.g., "look_at_obj_in_light-Bowl-None-DeskLamp-308")
            task_dir = os.path.basename(os.path.dirname(os.path.dirname(gamefile)))
            env_name = get_env_name_from_gamefile(task_dir)
            
            if env_name is None:
                continue  # Skip if we can't identify the task type
            
            # Use task directory as the goal description
            goal = task_dir
            
            tasks.append({
                'task': goal,
                'env_kwargs': {
                    'config': 'alfworld',
                    'gamefile': gamefile,
                },
                'task_type': prefixes.get(env_name, env_name),
                'env_name': env_name
            })
        
        print(f"Loaded {len(tasks)} ALFWorld tasks from {games_dir}")
        return tasks
    elif task == 'fever':
        with open(TASKS_PATH['fever'], 'r') as f:
            return [
                {
                    'task': row['claim'],
                    'answer': row['label'],
                    'env_name': 'fever',
                }
                for row in (json.loads(line) for line in f) 
            ][:100]
    elif task == 'pddl':
        TASK_NAMES = ["barman", "blockworld", "gripper", "tyreworld"]
        return get_all_environment_configs(TASK_NAMES, TASKS_PATH['pddl'])

    elif task == 'sciworld':
        from scienceworld import ScienceWorldEnv
        
        # Create a temporary env to dynamically query task names and variations
        tmp_env = ScienceWorldEnv("", envStepLimit=100)
        all_task_names = tmp_env.get_task_names()
        print(f"ScienceWorld supported tasks ({len(all_task_names)}): {all_task_names}")
        
        tasks = []
        for task_name in all_task_names:
            tmp_env.load(task_name, variationIdx=0)
            
            # Get test variations
            test_variations = tmp_env.get_variations_test()
            
            # Limit to max_variations from config (default 5)
            max_vars = 5
            variations = test_variations[:max_vars] if len(test_variations) > max_vars else test_variations
            
            for var_idx in variations:
                tasks.append({
                    'task': f'{task_name}_var{var_idx}',
                    'task_name': task_name,
                    'variation_idx': var_idx,
                    'simplification': 'teleportAction,openDoors,openContainers',
                    'env_name': task_name,
                })
        
        tmp_env.close()
        print(f"Loaded {len(tasks)} ScienceWorld tasks ({len(all_task_names)} task types)")
        return tasks

    raise ValueError(f'Unsupported task type: {task}')