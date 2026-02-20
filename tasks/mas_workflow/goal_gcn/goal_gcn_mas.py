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


# ================================ Agent Role Prompts ================================
# Domain-specific instructions (e.g. ALFWorld rules) are automatically appended
# by the framework via run.py → agent.add_task_instruction(). Keep these GENERIC.

GOAL_PLANNER_PROMPT = """You are a smart agent designed to solve problems. You MUST strictly follow the output format of other agents' output."""

GOAL_EXECUTOR_PROMPT = """You are a smart agent designed to solve problems. You MUST strictly follow the output format of other agents' output."""

GOAL_CRITIC_PROMPT = """You are a judge. Given a task and an agent's output for that task, your job is to evaluate the agent's output and give your suggestion.
NOTE:
- If you believe the agent's answer is correct, simply output `Support`.
- If you believe the agent's answer is incorrect, provide a concise and strong suggestion."""


# ================================ Lightweight Agent Wrapper ================================

class AgentWrapper:
    """Lightweight wrapper holding an agent and its metadata."""

    def __init__(self, agent_id: str, role: str, agent: Agent):
        self.agent_id = agent_id
        self.role = role
        self.agent = agent


# ================================ Goal-RL MAS ================================

@dataclass
class GoalGCNMAS(MetaMAS):
    """
    Goal-conditioned Multi-Agent System with Skill Discovery and Evolving Experience.

    Architecture:
    - Planner Agent: Decomposes goals into subgoals
    - Executor Agent(s): Execute actions guided by Q-values and skills
    - Critic Agent: Evaluates proposed actions

    Key components:
    - Goal-conditioned RL with HER for value-based policy guidance
    - Automatic skill discovery from successful trajectories
    - Cross-episode insight distillation
    """

    def __post_init__(self):
        self.observers = []
        # NOTE: Do NOT use stop_strs=['\n'] here. Some environments (e.g.
        # ScienceWorld) need multi-line "Thought: ...\nAction: ..." output.
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=None)

        self._planner: AgentWrapper = None
        self._executors: List[AgentWrapper] = []
        self._critic: AgentWrapper = None

        self._current_goal: StructuredGoal = None
        self._current_subgoal: str = None
        self._goal_parser: GoalParser = None

    def build_system(
        self,
        reasoning: ReasoningBase,
        mas_memory: MASMemoryBase,
        env: Env,
        config: dict,
    ):
        """Build the Goal-RL MAS."""

        num_executors = config.get('num_executors', 2)
        use_critic = config.get('use_critic', True)

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
        """Create the agent team."""

        planner_agent = Agent(
            name='planner',
            role='planner',
            system_instruction=GOAL_PLANNER_PROMPT,
            reasoning_module=reasoning,
        )
        self._planner = AgentWrapper('planner', 'planner', planner_agent)
        self.hire([planner_agent])

        for i in range(num_executors):
            executor_agent = Agent(
                name=f'executor_{i}',
                role='executor',
                system_instruction=GOAL_EXECUTOR_PROMPT,
                reasoning_module=reasoning,
            )
            self._executors.append(AgentWrapper(f'executor_{i}', 'executor', executor_agent))
            self.hire([executor_agent])

        if use_critic:
            critic_agent = Agent(
                name='critic',
                role='critic',
                system_instruction=GOAL_CRITIC_PROMPT,
                reasoning_module=reasoning,
            )
            self._critic = AgentWrapper('critic', 'critic', critic_agent)
            self.hire([critic_agent])
    
    def schedule(self, task_config: dict) -> Tuple[float, bool]:
        """
        Execute a task using Goal-RL MAS.

        Flow:
        1. Parse goal and initialize tracking
        2. Retrieve experiences, insights, and skills
        3. Loop:
           a. Planner proposes subgoal
           b. Executors propose actions (guided by Q-values + skills)
           c. Critic evaluates (optional)
           d. Execute best action
           e. Update state and Goal RL
        4. Save results and update memory
        """
        task_main = task_config.get('task_main')
        task_description = task_config.get('task_description')
        few_shots = task_config.get('few_shots', [])

        if not task_main or not task_description:
            raise ValueError("task_main and task_description required")

        env = self.env

        self.meta_memory.init_task_context(task_main, task_description)

        self._current_goal = self._goal_parser.parse(task_main, task_description)

        # ---- Retrieve memory: trajectories, insights, and skills ----
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

        # Format few-shots from memory
        memory_shots = [
            format_task_context(t.task_description, t.task_trajectory, t.get_extra_field('key_steps'))
            for t in successful_trajs
        ]

        # Format skills for prompt injection
        skills_text = ""
        if skills and hasattr(self.meta_memory, 'format_skills_for_prompt'):
            skills_text = self.meta_memory.format_skills_for_prompt(skills, max_skills=2)
        if skills_text:
            self.notify_observers(f"[Skills] Injecting {len(skills)} discovered skill(s) into prompt")

        # ---- Main execution loop ----
        current_state = task_description
        recent_actions: list = []
        max_repeated_actions = 3
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

            subgoal = self._get_planner_output(goal=task_main, state=full_context)
            self._current_subgoal = subgoal

            suggested_action = None
            if isinstance(self.meta_memory, GoalRLMemory):
                suggested_action, q_value, _ = self.meta_memory.get_action_suggestion(current_state)

            proposed_actions = []
            for executor in self._executors:
                action = self._get_executor_output(
                    executor=executor,
                    state=full_context,
                )
                proposed_actions.append((executor.agent_id, action))

            best_action = proposed_actions[0][1] if proposed_actions else suggested_action
            if self._critic and len(proposed_actions) >= 2:
                best_action = self._critic_select(actions=proposed_actions)

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

            step_msg = f"Step {step}: {action}\nObs: {observation}"
            self.notify_observers(step_msg)

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

    def _get_planner_output(self, goal: str, state: str) -> str:
        """Get planner's subgoal recommendation."""
        response = self._planner.agent.response(state, self.reasoning_config)
        return response.strip() if response else "look"

    def _get_executor_output(self, executor: AgentWrapper, state: str) -> str:
        """Get executor's action proposal."""
        response = executor.agent.response(state, self.reasoning_config)
        return response.strip() if response else ""

    def _critic_select(self, actions: List[Tuple[str, str]]) -> str:
        """Critic selects the best action from proposed actions."""
        if not self._critic:
            return actions[0][1] if actions else ""

        if len(actions) >= 2 and actions[0][1] == actions[1][1]:
            return actions[0][1]

        return actions[0][1] if actions else ""

    def add_observer(self, observer):
        """Add an observer for logging."""
        self.observers.append(observer)

    def notify_observers(self, message: str):
        """Notify all observers with a message."""
        for observer in self.observers:
            observer.log(message)

