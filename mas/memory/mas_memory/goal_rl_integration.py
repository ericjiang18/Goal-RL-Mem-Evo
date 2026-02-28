"""
Goal RL Integration Module for Goal-Memory

This module provides the integration layer between the Goal RL Framework
and the Goal-conditioned Memory system, enabling:
1. Seamless experience collection during MAS execution
2. Goal-conditioned retrieval enhancement
3. Policy-guided action suggestions
4. Adaptive reward shaping
5. **Evolving Prompts** - Dynamic prompt optimization via Bandit algorithms

Usage:
    from mas.memory.mas_memory.goal_rl_integration import GoalRLMemory
    
    memory = GoalRLMemory(...)
    memory.init_task_context(task_main, task_description)
    
    # Get evolved prompt for agent role
    prompt, variant = memory.get_evolved_prompt_for_role("solver", domain="alfworld")
    
    # ... during execution ...
    rl_info = memory.process_step(state, action, next_state, reward, done)
    
    # ... after task ...
    memory.save_task_context(label=success)  # This updates prompt stats
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from .gmemory_plus import GMemoryPlus, GMemoryPlusConfig
from .goal_rl_framework import (
    GoalRLFramework,
    GoalRLConfig,
    Experience,
    Episode,
    ReplayStrategy,
)
from .goal_module import GoalParser, GoalMatcher, StructuredGoal
from ..common import MASMessage, AgentMessage, StateChain

from mas.llm import LLMCallable, Message
from mas.utils import EmbeddingFunc


# ================================ RL-Enhanced Configuration ================================

@dataclass
class GoalRLMemoryConfig(GMemoryPlusConfig):
    """Extended configuration for Goal RL Memory."""
    # Goal RL settings
    enable_goal_rl: bool = True
    rl_buffer_capacity: int = 1000
    rl_her_ratio: float = 0.8
    rl_her_strategy: str = "future"  # "future", "final", "episode", "random"
    
    # Value function
    rl_learning_rate: float = 0.3
    rl_discount_factor: float = 0.90
    
    # Reward shaping
    enable_reward_shaping: bool = True
    intrinsic_reward_weight: float = 0.1
    goal_distance_reward: bool = True
    
    # Policy guidance
    enable_policy_guidance: bool = True
    action_suggestion_threshold: float = 0.6  # Min Q-value to suggest action
    
    # Training
    rl_batch_size: int = 32
    rl_train_interval: int = 10


# ================================ Goal RL Memory ================================

@dataclass
class GoalRLMemory(GMemoryPlus):
    """
    Goal Reinforcement Learning enhanced memory system.
    
    Extends GMemoryPlus with:
    - Goal-conditioned value function learning
    - Hindsight Experience Replay
    - Hierarchical goal management
    - Intrinsic motivation rewards
    - Policy-based action suggestions
    """
    
    # RL-specific configuration
    rl_config: GoalRLMemoryConfig = field(default_factory=GoalRLMemoryConfig)
    
    def __post_init__(self):
        # Initialize parent class
        super().__post_init__()
        
        # Override config if needed
        if not hasattr(self, 'rl_config') or self.rl_config is None:
            self.rl_config = GoalRLMemoryConfig()
        
        # Initialize Goal RL Framework
        if self.rl_config.enable_goal_rl:
            self._init_goal_rl()
        else:
            self.goal_rl = None
        
        # Step tracking
        self._current_state: str = ""
        self._step_history: List[Dict[str, Any]] = []
        
        print(f"GoalRLMemory initialized with Goal RL: {self.rl_config.enable_goal_rl}")
    
    def _init_goal_rl(self):
        """Initialize Goal RL Framework."""
        # Map strategy string to enum
        strategy_map = {
            "future": ReplayStrategy.FUTURE,
            "final": ReplayStrategy.FINAL,
            "episode": ReplayStrategy.EPISODE,
            "random": ReplayStrategy.RANDOM,
        }
        
        rl_config = GoalRLConfig(
            buffer_capacity=self.rl_config.rl_buffer_capacity,
            her_ratio=self.rl_config.rl_her_ratio,
            her_strategy=strategy_map.get(
                self.rl_config.rl_her_strategy, 
                ReplayStrategy.FUTURE
            ),
            learning_rate=self.rl_config.rl_learning_rate,
            discount_factor=self.rl_config.rl_discount_factor,
            enable_hierarchical=True,
            enable_intrinsic=True,
            batch_size=self.rl_config.rl_batch_size,
            train_interval=self.rl_config.rl_train_interval,
        )
        
        self.goal_rl = GoalRLFramework(
            llm_model=self.llm_model,
            embedding_func=self.embedding_func,
            goal_parser=self.goal_parser,
            working_dir=os.path.join(self.persist_dir, "goal_rl"),
            config=rl_config,
        )
        
        # Load any saved state
        self.goal_rl.load()
    
    # ================================ Enhanced Task Context ================================
    
    def init_task_context(
        self,
        task_main: str,
        task_description: str = None,
        screenshot: Any = None,
        domain: str = None,
    ) -> MASMessage:
        """
        Initialize task context with Goal RL episode setup.
        """
        # Call parent method
        mas_message = super().init_task_context(
            task_main, task_description, screenshot, domain
        )
        
        # Start RL episode
        if self.goal_rl and self._current_goal:
            self.goal_rl.start_episode(
                task_id=self._current_task_id,
                goal=self._current_goal,
                initial_state=task_description or task_main,
            )
        
        # Initialize step tracking
        self._current_state = task_description or task_main
        self._step_history = []
        
        return mas_message
    
    # ================================ Enhanced Step Processing ================================
    
    def process_step(
        self,
        state: str,
        action: str,
        next_state: str,
        reward: float,
        done: bool,
    ) -> Dict[str, Any]:
        """
        Process a single step with Goal RL.
        
        Args:
            state: Current state description
            action: Action taken
            next_state: Resulting state
            reward: Environment reward
            done: Whether episode is done
        
        Returns:
            Dict containing RL information:
            - shaped_reward: Reward with intrinsic bonuses
            - suggested_action: Best action according to policy
            - current_subgoal: Active subgoal
            - q_values: Q-values for context
            - guidance: Text guidance for agent
        """
        rl_info = {
            "shaped_reward": reward,
            "suggested_action": None,
            "current_subgoal": None,
            "q_values": {},
            "guidance": "",
            "intrinsic_reward": 0.0,
        }
        
        if not self.goal_rl or not self._current_goal:
            return rl_info
        
        # Process step through Goal RL
        step_info = self.goal_rl.step(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
        )
        
        # Update RL info
        rl_info.update({
            "shaped_reward": step_info.get("shaped_reward", reward),
            "suggested_action": step_info.get("suggested_action"),
            "current_subgoal": step_info.get("current_subgoal"),
            "intrinsic_reward": step_info.get("intrinsic_reward", 0),
            "exploration_bonus": step_info.get("exploration_bonus", 0),
        })
        
        # Get guidance
        rl_info["guidance"] = self.goal_rl.get_guidance(next_state, self._current_goal)
        
        # Update current state
        self._current_state = next_state
        
        # Store in history
        self._step_history.append({
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "shaped_reward": rl_info["shaped_reward"],
            "done": done,
        })
        
        return rl_info
    
    def move_memory_state(self, action: str, observation: str, **kwargs) -> Dict[str, Any]:
        """
        Enhanced state transition with RL processing.
        """
        # Get reward from kwargs
        reward = kwargs.get('reward', 0.0)
        done = kwargs.get('done', False)
        
        # Process through RL
        rl_info = self.process_step(
            state=self._current_state,
            action=action,
            next_state=observation,
            reward=reward,
            done=done,
        )
        
        # Call parent method with shaped reward
        kwargs['reward'] = rl_info.get('shaped_reward', reward)
        super().move_memory_state(action, observation, **kwargs)
        
        return rl_info
    
    # ================================ Enhanced Retrieval ================================
    
    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 2,
        failed_topk: int = 1,
        insight_topk: int = 10,
        skill_topk: int = 3,
        threshold: float = 0.3,
        screenshot: Any = None,
        **kwargs
    ) -> Tuple[List[MASMessage], List[MASMessage], List[str], List[Any]]:
        """
        Enhanced memory retrieval with policy-guided ranking.
        """
        # Get base retrieval
        successful_trajs, failed_trajs, insights, skills = super().retrieve_memory(
            query_task=query_task,
            successful_topk=successful_topk * 2,
            failed_topk=failed_topk * 2,
            insight_topk=insight_topk,
            skill_topk=skill_topk,
            threshold=threshold,
            screenshot=screenshot,
            **kwargs
        )
        
        # Re-rank using Q-values if Goal RL is enabled
        if self.goal_rl and self._current_goal:
            successful_trajs = self._rerank_by_policy(
                successful_trajs, 
                self._current_goal,
                successful_topk
            )
        
        return (
            successful_trajs[:successful_topk], 
            failed_trajs[:failed_topk], 
            insights, 
            skills
        )
    
    def _rerank_by_policy(
        self,
        trajectories: List[MASMessage],
        goal: StructuredGoal,
        top_k: int,
    ) -> List[MASMessage]:
        """Re-rank trajectories using learned policy values."""
        if not trajectories:
            return trajectories
        
        scored = []
        for traj in trajectories:
            # Get Q-value for trajectory's actions
            q_sum = 0.0
            action_count = 0
            
            # Parse actions from trajectory
            for line in traj.task_trajectory.split('\n'):
                if line.startswith('>'):
                    action = line[1:].strip()
                    if action:
                        q_value = self.goal_rl.value_function.get_q_value(
                            state="", 
                            action=action, 
                            goal=goal
                        )
                        q_sum += q_value
                        action_count += 1
            
            avg_q = q_sum / max(action_count, 1)
            
            # Combine with success label
            success_bonus = 1.0 if traj.label else 0.0
            score = 0.3 * avg_q + 0.7 * success_bonus
            
            scored.append((traj, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [traj for traj, _ in scored[:top_k]]
    
    # ================================ Action Suggestions ================================
    
    def get_action_suggestion(
        self,
        current_state: str,
        available_actions: List[str] = None,
    ) -> Tuple[Optional[str], float, str]:
        """
        Get action suggestion from learned policy.
        
        Args:
            current_state: Current state description
            available_actions: List of possible actions
        
        Returns:
            Tuple of (best_action, q_value, explanation)
        """
        if not self.goal_rl or not self._current_goal:
            return None, 0.0, "Goal RL not initialized"
        
        best_action, q_value = self.goal_rl.value_function.get_best_action(
            state=current_state,
            goal=self._current_goal,
            available_actions=available_actions,
        )
        
        # Generate explanation
        if best_action and q_value > self.rl_config.action_suggestion_threshold:
            explanation = f"Based on {self.goal_rl.value_function.update_count} learned experiences"
        else:
            explanation = "Insufficient experience for confident suggestion"
        
        return best_action, q_value, explanation
    
    def get_action_values(
        self,
        current_state: str,
        actions: List[str],
    ) -> Dict[str, float]:
        """Get Q-values for a list of actions."""
        if not self.goal_rl or not self._current_goal:
            return {a: 0.0 for a in actions}
        
        return self.goal_rl.get_action_values(
            state=current_state,
            goal=self._current_goal,
            actions=actions,
        )
    
    # ================================ Goal Progress ================================
    
    def get_goal_progress(self) -> Dict[str, Any]:
        """Get current goal progress information."""
        if not self.goal_rl or not self._current_task_id:
            return {"progress": 0.0, "subgoals": []}
        
        progress = {
            "progress": self.goal_rl.goal_manager.get_completion_ratio(
                self._current_task_id
            ) if self.goal_rl.goal_manager else 0.0,
            "current_subgoal": None,
            "subgoals": [],
        }
        
        if self.goal_rl.goal_manager:
            subgoal = self.goal_rl.goal_manager.get_current_subgoal(
                self._current_task_id
            )
            if subgoal:
                progress["current_subgoal"] = subgoal.raw_task
        
        return progress
    
    def get_rl_guidance(self) -> str:
        """Get RL-based guidance for current state."""
        if not self.goal_rl or not self._current_goal:
            return ""
        
        return self.goal_rl.get_guidance(
            state=self._current_state,
            goal=self._current_goal,
        )
    
    # ================================ Evolving Prompts (DISABLED) ================================
    
    # def get_evolved_prompt_for_role(
    #     self,
    #     role: str,
    #     default_prompt: str = None,
    #     domain: str = None,
    # ) -> Tuple[str, Any]:
    #     """
    #     Get an evolved prompt for an agent role using Bandit selection.
    #     
    #     The prompt evolution system uses Thompson Sampling / UCB to balance
    #     exploration and exploitation of prompt variants.
    #     
    #     Args:
    #         role: Agent role name (e.g., "solver", "critic")
    #         default_prompt: Default prompt if no variants exist
    #         domain: Task domain (auto-detected from current goal if not provided)
    #     
    #     Returns:
    #         Tuple of (evolved_prompt_text, PromptVariant object for tracking)
    #     """
    #     if not self.prompt_evolution:
    #         return default_prompt or f"You are a {role} agent.", None
        
    #     # Determine domain
    #     if domain is None and self._current_goal:
    #         domain = self._current_goal.domain
    #     domain = domain or "general"
        
    #     # Register default if needed
    #     if default_prompt:
    #         self.prompt_evolution.register_default_prompt(role, default_prompt, domain)
        
    #     # Select best variant using Bandit algorithm
    #     variant = self.prompt_evolution.select_prompt(role, domain)
        
    #     # Inject insights if available
    #     if hasattr(self, 'insights_cache') and self.insights_cache:
    #         prompt = self.prompt_evolution.inject_insights(
    #             variant.content, 
    #             self.insights_cache[:5]
    #         )
    #     else:
    #         prompt = variant.content
        
    #     # Store current variant for later update
    #     self._current_prompt_variants = getattr(self, '_current_prompt_variants', {})
    #     self._current_prompt_variants[role] = variant
        
    #     return prompt, variant
    
    # def update_prompt_feedback(
    #     self,
    #     role: str,
    #     success: bool,
    #     failure_reason: str = None,
    #     tokens_used: int = 0,
    # ):
    #     """
    #     Update prompt variant statistics after task execution.
    #     
    #     This feedback is used to:
    #     1. Update Bandit statistics (success/failure counts)
    #     2. Collect failure patterns for LLM synthesis
    #     3. Trigger evolution when enough data is collected
    #     
    #     Args:
    #         role: Agent role
    #         success: Whether the task succeeded
    #         failure_reason: Reason for failure (helps LLM synthesize better prompts)
    #         tokens_used: Number of tokens used (for efficiency tracking)
    #     """
    #     if not self.prompt_evolution:
    #         return
        
    #     variants = getattr(self, '_current_prompt_variants', {})
    #     variant = variants.get(role)
        
    #     if variant:
    #         self.prompt_evolution.update_stats(
    #             role=role,
    #             variant=variant,
    #             success=success,
    #             tokens_used=tokens_used,
    #             failure_reason=failure_reason,
    #         )
    
    # def trigger_prompt_evolution(
    #     self,
    #     role: str,
    #     domain: str = None,
    #     insights: List[str] = None,
    # ) -> Optional[Any]:
    #     """
    #     Manually trigger prompt evolution for a role.
    #     
    #     Usually evolution is triggered automatically, but this allows
    #     manual triggering when you have new insights.
    #     
    #     Args:
    #         role: Agent role
    #         domain: Task domain
    #         insights: New insights to incorporate
    #     
    #     Returns:
    #         New PromptVariant if evolution succeeded
    #     """
    #     if not self.prompt_evolution:
    #         return None
        
    #     domain = domain or "general"
    #     insights = insights or []
        
    #     return self.prompt_evolution.evolve_prompt(role, domain, insights)
    
    # def get_prompt_evolution_stats(self, role: str = None) -> Dict[str, Any]:
    #     """
    #     Get prompt evolution statistics.
    #     
    #     Returns information about:
    #     - Number of variants per role
    #     - Success rates
    #     - Evolution history
    #     """
    #     if not self.prompt_evolution:
    #         return {"status": "disabled"}
        
    #     if role:
    #         return self.prompt_evolution.get_stats(role)
        
    #     # Get stats for all roles
    #     all_stats = {}
    #     for r in self.prompt_evolution.variants.keys():
    #         for domain in self.prompt_evolution.variants[r].keys():
    #             all_stats[f"{r}_{domain}"] = self.prompt_evolution.get_stats(r, domain)
        
    #     return all_stats
    
    # ================================ Memory Saving ================================
    
    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        """
        Save task context with RL episode completion and prompt evolution update.
        
        This method:
        1. Ends the RL episode and collects stats
        2. Updates prompt variant statistics based on success/failure
        3. Triggers prompt evolution if enough data collected
        """
        # End RL episode
        if self.goal_rl:
            episode_stats = self.goal_rl.end_episode(success=label)
            
            # Add RL stats to extra fields
            if self.current_task_context:
                self.current_task_context.add_extra_field(
                    'rl_episode_stats', 
                    episode_stats
                )
        
        # Update prompt evolution statistics for all used variants
        current_variants = getattr(self, '_current_prompt_variants', {})
        for role, variant in current_variants.items():
            self.update_prompt_feedback(
                role=role,
                success=label,
                failure_reason=feedback if not label else None,
            )
        
        # Clear current variants
        self._current_prompt_variants = {}
        
        # Call parent save
        return super().save_task_context(label=label, feedback=feedback)
    
    def add_memory(self, mas_message: MASMessage) -> None:
        """Add memory with RL updates."""
        super().add_memory(mas_message)
        
        # Save RL state periodically
        if self.goal_rl and self.memory_size % 10 == 0:
            self.goal_rl.save()
    
    # ================================ Statistics ================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including RL."""
        stats = super().get_stats()
        
        if self.goal_rl:
            stats["goal_rl"] = self.goal_rl.get_statistics()
        
        return stats


# ================================ Factory Function ================================

def create_goal_rl_memory(
    llm_model: LLMCallable,
    embedding_func: EmbeddingFunc,
    working_dir: str,
    enable_all_features: bool = True,
    **kwargs,
) -> GoalRLMemory:
    """
    Factory function to create GoalRLMemory with recommended settings.
    
    Args:
        llm_model: LLM callable
        embedding_func: Embedding function
        working_dir: Directory for persistence
        enable_all_features: Whether to enable all G-Memory++ features
        **kwargs: Additional config overrides
    
    Returns:
        Configured GoalRLMemory instance
    """
    config = GoalRLMemoryConfig(
        enable_goal_module=enable_all_features,
        # enable_prompt_evolution=False,
        enable_skill_miner=enable_all_features,
        
        enable_goal_rl=True,
        rl_buffer_capacity=kwargs.get('buffer_capacity', 10000),
        rl_her_ratio=kwargs.get('her_ratio', 0.8),
        rl_learning_rate=kwargs.get('learning_rate', 0.1),
        enable_reward_shaping=True,
        enable_policy_guidance=True,
    )
    
    return GoalRLMemory(
        namespace="goal_rl_memory",
        global_config={"working_dir": working_dir},
        llm_model=llm_model,
        embedding_func=embedding_func,
        config=config,  # Pass as GMemoryPlusConfig
        rl_config=config,
    )

