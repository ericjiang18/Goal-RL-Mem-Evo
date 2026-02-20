from dataclasses import dataclass
from typing import Any
import re

from .base_env import BaseEnv, BaseRecorder


# ScienceWorld task names — queried dynamically at task-loading time.
# This constant is only a fallback; get_task() uses env.get_task_names() instead.
SCIWORLD_TASK_NAMES = None  # populated lazily


class SciworldEnv(BaseEnv):
    """ScienceWorld environment wrapper following the same pattern as AlfworldEnv."""

    def __init__(
        self,
        env_config: dict[str, Any],
        max_trials: int = 30,
    ):
        from scienceworld import ScienceWorldEnv
        self.env_config = env_config
        self.max_trials: int = max_trials
        self.env = ScienceWorldEnv("", envStepLimit=max_trials)
        self.done = False
        self.score = 0.0
        self.task_name = ""
        self.variation_idx = 0

    # Default simplifications that match the prompt:
    # - teleportAction: allows "teleport to LOC" for instant room navigation
    # - openDoors: all doors start open
    # - openContainers: all containers start open
    DEFAULT_SIMPLIFICATIONS = "teleportAction,openDoors,openContainers"

    def set_env(self, configs: dict) -> tuple[str, str]:
        self.task_name = configs['task_name']
        self.variation_idx = configs.get('variation_idx', 0)
        simplification = configs.get('simplification', self.DEFAULT_SIMPLIFICATIONS)

        self.env.load(
            self.task_name,
            variationIdx=self.variation_idx,
            simplificationStr=simplification,
        )
        obs, info = self.env.reset()

        self.done = False
        self.score = 0.0

        task_desc = info.get('taskDesc', self.task_name)
        task_main = f"{self.task_name}-{task_desc}"
        task_description = f"{obs}\n\nTask: {task_desc}"

        return task_main, task_description

    def step(self, action: str) -> tuple[str, float, bool]:
        action = self.process_action(action)

        # Intercept think/thought actions — they don't affect the environment
        if action.lower().startswith(('think:', 'think ', 'thought:')):
            return 'OK.', 0, self.done

        obs, reward, done, info = self.env.step(action)
        self.done = done
        self.score = info.get('score', 0.0)

        # Normalize reward: SciWorld reward is score delta
        # Map to: -1 for no progress, 0 for some progress, 1 for task completion
        if done and self.score >= 100:
            processed_reward = 1
        elif reward > 0:
            processed_reward = 0
        elif 'not a valid action' in obs.lower() or obs.strip() == '':
            processed_reward = -1
        else:
            processed_reward = 0

        return obs, processed_reward, self.done

    def feedback(self) -> tuple[float, bool, str]:
        success = self.score >= 100
        reward = self.score / 100.0
        if success:
            message = "You successfully finished this task!"
        else:
            message = f"You failed the task. Final score: {self.score}/100"

        return reward, success, message

    @staticmethod
    def process_action(action: str) -> str:
        """Normalize LLM-generated actions for ScienceWorld.
        
        Handles the "Thought: ...\nAction: ..." format from the prompt,
        extracting only the Action part to send to the environment.
        """
        action = action.strip()
        # Remove common LLM artifacts
        action = action.replace('<', '').replace('>', '')

        # Handle "Thought: ...\nAction: ..." format — extract the Action line
        if 'Action:' in action:
            # Find the last "Action:" line (in case Thought also mentions actions)
            for line in reversed(action.split('\n')):
                line = line.strip()
                if line.lower().startswith('action:'):
                    action = line.split(':', 1)[1].strip()
                    break

        # Take only the first line (in case there's still multi-line output)
        action = action.split('\n')[0].strip()

        # Remove square brackets that LLMs sometimes produce from the
        # "use OBJ [on OBJ]" documentation format.
        # e.g. "use brick [on inclined plane H]" → "use brick on inclined plane H"
        action = action.replace('[', '').replace(']', '')

        # Remove "OK." artifacts
        if action in ('OK.', 'OK'):
            action = 'look around'
        action = action.replace('OK.', '').replace('OK', '').strip()

        # Strip leading "> " prompt markers
        if action.startswith('> '):
            action = action[2:].strip()

        # Strip trailing periods that LLMs sometimes add
        if action.endswith('.'):
            action = action[:-1].strip()

        # Handle common LLM rewording mistakes
        # "go to X" → "teleport to X"
        if action.lower().startswith('go to '):
            action = 'teleport to ' + action[6:]

        return action


@dataclass
class SciworldRecorder(BaseRecorder):
    """Recorder for ScienceWorld tasks."""

    def __post_init__(self):
        super().__post_init__()
        self.task = 'sciworld'
        self.total_tasks = 0
        self.total_successes = 0
        self.total_score = 0.0
        self.task_scores: dict[str, list] = {}  # task_name -> [scores]

    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)
        message = f'---------- Task: {task_id} ({task_config.get("task_name", "")}) ----------'
        self.log(message)

    def task_end(self, reward: float, done: bool):
        self.total_tasks += 1
        score = reward * 100  # reward is score/100
        self.total_score += score

        task_name = self.current_task_config.get('task_name', 'unknown')
        if task_name not in self.task_scores:
            self.task_scores[task_name] = []
        self.task_scores[task_name].append(score)

        if done and score >= 100:
            self.total_successes += 1

        avg_score = self.total_score / self.total_tasks
        success_rate = self.total_successes / self.total_tasks

        message = (
            f'score: {score:.1f}, avg_score: {avg_score:.1f}, '
            f'success_rate: {success_rate:.3f} ({self.total_successes}/{self.total_tasks})'
        )
        self.log(message)
