from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

from mas.agents import Agent, Env
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase
from mas.llm import LLMCallable, Message

from mas.memory.mas_memory.goal_rl_integration import GoalRLMemory
from mas.memory.mas_memory.goal_module import GoalParser, StructuredGoal

from ..format import format_task_context, format_task_prompt_with_insights


EXECUTOR_PROMPT = """You are a smart agent designed to solve problems. You MUST strictly follow the output format of other agents' output."""

CRITIC_PROMPT = """You are a judge. Given a task and an agent's output for that task, your job is to evaluate the agent's output and give your suggestion.
NOTE:
- If you believe the agent's answer is correct, simply output `Support`.
- If you believe the agent's answer is incorrect, provide a concise and strong suggestion."""


class AgentWrapper:
    def __init__(self, agent_id: str, role: str, agent: Agent):
        self.agent_id = agent_id
        self.role = role
        self.agent = agent


@dataclass
class GoalGCNMAS(MetaMAS):
    """
    Goal-conditioned MAS with Goal RL, Skill Discovery, and Evolving Experience.

    Streamlined architecture:
    - Single Executor agent (with policy hints from Goal RL Q-values)
    - Optional Critic (only invoked when multiple executors disagree)
    """

    def __post_init__(self):
        self.observers = []
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=None)

        self._executors: List[AgentWrapper] = []
        self._critic: AgentWrapper = None

        self._current_goal: StructuredGoal = None
        self._goal_parser: GoalParser = None

    def build_system(
        self,
        reasoning: ReasoningBase,
        mas_memory: MASMemoryBase,
        env: Env,
        config: dict,
    ):
        num_executors = config.get('num_executors', 1)
        use_critic = config.get('use_critic', False)

        self._successful_topk = config.get('successful_topk', 2)
        self._failed_topk = config.get('failed_topk', 1)
        self._insights_topk = config.get('insights_topk', 5)
        self._threshold = config.get('threshold', 0.3)

        self.notify_observers("========================================")
        self.notify_observers("Goal-RL MAS Configuration")
        self.notify_observers("========================================")
        self.notify_observers(f"Num Executors     : {num_executors}")
        self.notify_observers(f"Use Critic        : {use_critic}")
        self.notify_observers(f"Successful TopK   : {self._successful_topk}")
        self.notify_observers(f"Insights TopK     : {self._insights_topk}")

        self._create_agents(reasoning, num_executors, use_critic)

        self.set_env(env)
        self.meta_memory = mas_memory

        if hasattr(mas_memory, 'goal_parser'):
            self._goal_parser = mas_memory.goal_parser
        else:
            self._goal_parser = GoalParser()

        self.notify_observers("Goal-RL MAS initialized successfully")

    def _create_agents(
        self,
        reasoning: ReasoningBase,
        num_executors: int,
        use_critic: bool,
    ):
        for i in range(num_executors):
            executor_agent = Agent(
                name=f'executor_{i}',
                role='executor',
                system_instruction=EXECUTOR_PROMPT,
                reasoning_module=reasoning,
            )
            self._executors.append(AgentWrapper(f'executor_{i}', 'executor', executor_agent))
            self.hire([executor_agent])

        if use_critic:
            critic_agent = Agent(
                name='critic',
                role='critic',
                system_instruction=CRITIC_PROMPT,
                reasoning_module=reasoning,
            )
            self._critic = AgentWrapper('critic', 'critic', critic_agent)
            self.hire([critic_agent])
    
    def schedule(self, task_config: dict) -> Tuple[float, bool]:
        task_main = task_config.get('task_main')
        task_description = task_config.get('task_description')
        few_shots = task_config.get('few_shots', [])

        if not task_main or not task_description:
            raise ValueError("task_main and task_description required")

        env = self.env

        self.meta_memory.init_task_context(task_main, task_description)
        self._current_goal = self._goal_parser.parse(task_main, task_description)

        # ---- Retrieve memory ----
        retrieval_result = self.meta_memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold,
        )

        if len(retrieval_result) == 4:
            successful_trajs, _, insights, skills = retrieval_result
        else:
            successful_trajs, _, insights = retrieval_result
            skills = []

        memory_shots = [
            format_task_context(t.task_description, t.task_trajectory, t.get_extra_field('key_steps'))
            for t in successful_trajs
        ]

        skills_text = ""
        if skills and hasattr(self.meta_memory, 'format_skills_for_prompt'):
            # Only inject 1 relevant skill to keep context clean and efficient
            skills_text = self.meta_memory.format_skills_for_prompt(skills, max_skills=1)
            #skills_text = self.meta_memory.format_skills_for_prompt(skills, max_skills=2)

        if skills_text:
            self.notify_observers(f"[Skills] Injecting {len(skills[:1])} discovered skill(s) into prompt")
            #self.notify_observers(f"[Skills] Injecting {len(skills)} discovered skill(s) into prompt")


        # ---- Main execution loop ----
        current_state = task_description
        recent_actions: list = []
        max_repeated_actions = 5
        real_steps = 0
        consecutive_thinks = 0
        max_free_thinks = 2
        step = 0

        while real_steps < env.max_trials and step < env.max_trials * 2:
            full_context = format_task_prompt_with_insights(
                few_shots=few_shots,
                memory_few_shots=memory_shots,
                insights=insights[:5] if insights else [],
                skills=skills_text,
                task_description=self.meta_memory.summarize(upstream_agent_ids=None),
            )

            # Append RL policy hint if available
            if isinstance(self.meta_memory, GoalRLMemory):
                suggested_action, q_value, _ = self.meta_memory.get_action_suggestion(current_state)
                if suggested_action and q_value > 0.3:
                    full_context += f"\n\n[Policy Hint] Based on past experience, action type \"{suggested_action}\" is recommended (confidence: {q_value:.2f}). You may follow or override this suggestion."

            # Collect executor proposals
            proposed_actions = []
            for executor in self._executors:
                response = executor.agent.response(full_context, self.reasoning_config)
                action = response.strip() if response else ""
                proposed_actions.append((executor.agent_id, action))

            best_action = proposed_actions[0][1] if proposed_actions else "look"

            # Lazy critic: only invoke when 2+ executors propose different actions
            if self._critic and len(proposed_actions) >= 2:
                unique_actions = list({a for _, a in proposed_actions if a})
                if len(unique_actions) > 1:
                    best_action = self._critic_select(proposed_actions)

            if not best_action:
                best_action = "look"

            action = env.process_action(best_action)

            # ---- Loop detection ----
            recent_actions.append(action)
            if len(recent_actions) > max_repeated_actions:
                recent_actions.pop(0)

            if len(recent_actions) >= max_repeated_actions:
                if len(set(recent_actions)) == 1:
                    self.notify_observers(f"[WARNING] Agent stuck in loop ({action}). Breaking out.")
                    break
                if all(a.lower().startswith(('think', 'thought')) for a in recent_actions):
                    contents = [a.split(':', 1)[-1].strip()[:50] for a in recent_actions]
                    if len(set(contents)) == 1:
                        self.notify_observers("[WARNING] Agent stuck in think loop. Breaking out.")
                        break

            observation, reward, done = env.step(action)

            if action.lower().startswith(('think', 'thought')):
                consecutive_thinks += 1
                if consecutive_thinks > max_free_thinks:
                    real_steps += 1
            else:
                consecutive_thinks = 0
                real_steps += 1

            step += 1

            self.notify_observers(f"Step {step}: {action}\nObs: {observation}")

            rl_info = self.meta_memory.move_memory_state(action, observation, reward=reward, done=done)

            if isinstance(rl_info, dict) and rl_info.get('guidance'):
                self.notify_observers(f"[Goal RL] {rl_info['guidance']}")

            current_state = observation

            if done:
                break

        final_reward, final_done, final_feedback = env.feedback()
        self.notify_observers(final_feedback)

        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
        self.meta_memory.backward(final_done)

        return final_reward, final_done

    def _critic_select(self, actions: List[Tuple[str, str]]) -> str:
        if not self._critic:
            return actions[0][1] if actions else ""

        unique_actions = list({a for _, a in actions if a})
        if len(unique_actions) <= 1:
            return unique_actions[0] if unique_actions else ""

        candidates = "\n".join(
            f"  Option {i+1}: {a}" for i, a in enumerate(unique_actions)
        )
        prompt = (
            f"You are a critic. Given the candidate actions below, pick the single "
            f"best action to execute. Reply with ONLY the chosen action text, nothing else.\n"
            f"{candidates}"
        )
        response = self._critic.agent.response(prompt, self.reasoning_config)
        chosen = response.strip() if response else ""

        for a in unique_actions:
            if a.lower() in chosen.lower() or chosen.lower() in a.lower():
                return a

        return unique_actions[0]

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)
