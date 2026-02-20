"""
Goal-conditioned Reinforcement Learning Framework for G-Memory++

This module implements a true Goal RL framework with:
1. Goal-conditioned Value Function - Q(s, a, g) learning
2. Hindsight Experience Replay (HER) - Learn from failures
3. Hierarchical Goal Decomposition - Subgoal management
4. Intrinsic Motivation - Curiosity-driven exploration
5. Goal Relabeling - Automatic goal achievement detection
6. Policy Learning - Experience-based policy improvement

References:
- Andrychowicz et al. (2017) "Hindsight Experience Replay"
- Schaul et al. (2015) "Universal Value Function Approximators"
- Nachum et al. (2018) "Data-Efficient Hierarchical Reinforcement Learning"
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import deque, defaultdict
from enum import Enum
import random
import pickle

from mas.llm import LLMCallable, Message
from .goal_module import StructuredGoal, GoalParser

# ================================ Constants & Enums ================================

class GoalStatus(Enum):
    """Status of a goal during execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ReplayStrategy(Enum):
    """HER replay strategies."""
    FUTURE = "future"      # Use future achieved states as goals
    FINAL = "final"        # Use final state as goal
    EPISODE = "episode"    # Random states from same episode
    RANDOM = "random"      # Random achieved goals from buffer


# ================================ Goal RL Prompts ================================

GOAL_ACHIEVEMENT_CHECK_PROMPT = """Determine if the goal has been achieved based on the current state.

## Goal:
{goal}

## Current State:
{state}

## Recent Actions:
{actions}

Has the goal been achieved? Consider partial achievements.

Respond with:
ACHIEVED: [YES/NO/PARTIAL]
CONFIDENCE: [0.0-1.0]
ACHIEVED_SUBGOALS: [list of achieved subgoals if any]
REMAINING: [what remains to be done]
"""

GOAL_DISTANCE_PROMPT = """Estimate how close the current state is to achieving the goal.

## Goal:
{goal}

## Current State:
{state}

Rate the distance to goal on a scale of 0-10 where:
- 0: Goal achieved
- 5: Halfway there
- 10: Just started / very far

Respond with just a number:
"""

POLICY_IMPROVEMENT_PROMPT = """Given the following successful and failed trajectories for similar goals, 
extract key action patterns and decision rules.

## Goal Type: {goal_type}

## Successful Trajectories:
{success_trajectories}

## Failed Trajectories:
{failed_trajectories}

## Analysis:
1. What actions led to success?
2. What actions led to failure?
3. What decision rules should be followed?

Provide actionable guidelines:
"""

INTRINSIC_REWARD_PROMPT = """Evaluate the novelty and information gain of this state transition.

## Previous State:
{prev_state}

## Action Taken:
{action}

## New State:
{new_state}

## Goal:
{goal}

Rate the following (0-10):
1. State novelty (how different from expected/seen states)
2. Information gain (how much was learned)
3. Goal relevance (how related to goal achievement)

Format: NOVELTY: X, INFO_GAIN: Y, RELEVANCE: Z
"""


# ================================ Data Classes ================================

@dataclass
class Experience:
    """A single experience tuple for RL."""
    state: str
    action: str
    next_state: str
    reward: float
    done: bool
    goal: StructuredGoal
    
    # Additional fields for HER
    achieved_goal: Optional[StructuredGoal] = None
    hindsight_reward: float = 0.0
    
    # Embeddings for similarity
    state_embedding: Optional[np.ndarray] = None
    goal_embedding: Optional[np.ndarray] = None
    
    # Metadata
    step_idx: int = 0
    episode_id: str = ""
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "done": self.done,
            "goal": self.goal.to_dict() if self.goal else None,
            "achieved_goal": self.achieved_goal.to_dict() if self.achieved_goal else None,
            "hindsight_reward": self.hindsight_reward,
            "step_idx": self.step_idx,
            "episode_id": self.episode_id,
        }


@dataclass
class Episode:
    """A complete episode of experiences."""
    episode_id: str
    goal: StructuredGoal
    experiences: List[Experience] = field(default_factory=list)
    final_reward: float = 0.0
    success: bool = False
    
    # Achieved goals during episode (for HER)
    achieved_goals: List[StructuredGoal] = field(default_factory=list)
    
    # Statistics
    total_steps: int = 0
    total_reward: float = 0.0
    
    def add_experience(self, exp: Experience):
        exp.episode_id = self.episode_id
        exp.step_idx = len(self.experiences)
        self.experiences.append(exp)
        self.total_steps += 1
        self.total_reward += exp.reward
    
    def get_trajectory_text(self) -> str:
        """Get trajectory as text for LLM processing."""
        parts = []
        for exp in self.experiences:
            parts.append(f"State: {exp.state[:200]}")
            parts.append(f"Action: {exp.action}")
            parts.append(f"Reward: {exp.reward}")
        return "\n".join(parts)


@dataclass
class GoalValueEntry:
    """Entry in the goal-value function table."""
    goal_type: str
    state_pattern: str
    action: str
    q_value: float
    visit_count: int = 0
    success_count: int = 0
    
    def update(self, reward: float, learning_rate: float = 0.1):
        """Update Q-value with new reward."""
        self.visit_count += 1
        self.q_value = self.q_value + learning_rate * (reward - self.q_value)
        # TD lambda
        if reward > 0:
            self.success_count += 1


@dataclass 
class SubgoalNode:
    """A node in the hierarchical goal tree."""
    goal: StructuredGoal
    parent: Optional['SubgoalNode'] = None
    children: List['SubgoalNode'] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    priority: float = 1.0
    estimated_steps: int = 5
    actual_steps: int = 0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def mark_achieved(self):
        self.status = GoalStatus.ACHIEVED
        # Propagate to parent
        if self.parent:
            siblings = self.parent.children
            if all(s.status == GoalStatus.ACHIEVED for s in siblings):
                self.parent.mark_achieved()


# ================================ Experience Replay Buffer ================================

class GoalConditionedReplayBuffer:
    """
    Experience replay buffer with goal-conditioning and HER support.
    
    Implements:
    - Standard experience replay
    - Hindsight Experience Replay (HER)
    - Prioritized replay based on TD-error
    - Goal-based filtering and retrieval
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        her_ratio: float = 0.8,  # Fraction of HER goals in batch
        her_strategy: ReplayStrategy = ReplayStrategy.FUTURE,
        embedding_func: Callable = None,
    ):
        self.capacity = capacity
        self.her_ratio = her_ratio
        self.her_strategy = her_strategy
        self.embedding_func = embedding_func
        
        # Storage
        self.buffer: deque = deque(maxlen=capacity)
        self.episodes: Dict[str, Episode] = {}
        
        # Goal-based indexing
        self.goal_index: Dict[str, List[int]] = defaultdict(list)
        
        # Priority tracking
        self.priorities: np.ndarray = np.zeros(capacity)
        self.max_priority: float = 1.0
        
        # Statistics
        self.total_added = 0
        self.total_sampled = 0
    
    def add_episode(self, episode: Episode):
        """Add a complete episode to the buffer."""
        self.episodes[episode.episode_id] = episode
        
        for exp in episode.experiences:
            self._add_experience(exp)
        
        # Extract achieved goals for HER
        self._extract_achieved_goals(episode)
    
    def _add_experience(self, exp: Experience):
        """Add single experience to buffer."""
        idx = len(self.buffer) % self.capacity
        
        # Compute embedding if function available
        if self.embedding_func and exp.state_embedding is None:
            exp.state_embedding = self.embedding_func.embed_query(exp.state)
        if self.embedding_func and exp.goal_embedding is None:
            exp.goal_embedding = self.embedding_func.embed_query(exp.goal.raw_task)
        
        self.buffer.append(exp)
        
        # Index by goal type
        goal_type = exp.goal.verb if exp.goal else "unknown"
        self.goal_index[goal_type].append(idx)
        
        # Set initial priority
        self.priorities[idx] = self.max_priority
        
        self.total_added += 1
    
    def _extract_achieved_goals(self, episode: Episode):
        """Extract achieved goals from episode for HER."""
        # Use states as implicit achieved goals
        achieved = []
        for i, exp in enumerate(episode.experiences):
            # Create achieved goal from state
            achieved_goal = StructuredGoal(
                domain=episode.goal.domain,
                verb="achieved",
                target=exp.next_state[:100],  # Use state as target
                raw_task=f"Reach state: {exp.next_state[:100]}",
            )
            achieved.append(achieved_goal)
        
        episode.achieved_goals = achieved
    
    def sample_batch(
        self,
        batch_size: int,
        goal_type: str = None,
        use_her: bool = True,
    ) -> List[Experience]:
        """
        Sample a batch of experiences with optional HER.
        
        Args:
            batch_size: Number of experiences to sample
            goal_type: Optional filter by goal type
            use_her: Whether to apply HER
        
        Returns:
            List of Experience objects
        """
        if len(self.buffer) == 0:
            return []
        
        # Filter by goal type if specified
        if goal_type and goal_type in self.goal_index:
            valid_indices = self.goal_index[goal_type]
            valid_indices = [i for i in valid_indices if i < len(self.buffer)]
        else:
            valid_indices = list(range(len(self.buffer)))
        
        if not valid_indices:
            valid_indices = list(range(len(self.buffer)))
        
        # Sample indices
        sample_size = min(batch_size, len(valid_indices))
        sampled_indices = random.sample(valid_indices, sample_size)
        
        batch = []
        for idx in sampled_indices:
            exp = self.buffer[idx]
            
            # Apply HER with probability her_ratio
            if use_her and random.random() < self.her_ratio:
                exp = self._apply_her(exp)
            
            batch.append(exp)
        
        self.total_sampled += len(batch)
        return batch
    
    def _apply_her(self, exp: Experience) -> Experience:
        """Apply Hindsight Experience Replay to an experience."""
        if exp.episode_id not in self.episodes:
            return exp
        
        episode = self.episodes[exp.episode_id]
        
        if self.her_strategy == ReplayStrategy.FUTURE:
            # Sample a future achieved goal
            future_exps = [e for e in episode.experiences if e.step_idx > exp.step_idx]
            if future_exps:
                future_goal = random.choice(future_exps)
                # Create hindsight goal from future state
                hindsight_goal = StructuredGoal(
                    domain=exp.goal.domain,
                    verb="reach",
                    target=future_goal.next_state[:100],
                    raw_task=f"Reach: {future_goal.next_state[:100]}",
                )
                # Check if this goal was achieved
                hindsight_reward = 1.0 if exp.next_state == future_goal.next_state else 0.0
                
                # Create new experience with hindsight goal
                her_exp = Experience(
                    state=exp.state,
                    action=exp.action,
                    next_state=exp.next_state,
                    reward=hindsight_reward,
                    done=hindsight_reward > 0,
                    goal=hindsight_goal,
                    achieved_goal=hindsight_goal if hindsight_reward > 0 else None,
                    hindsight_reward=hindsight_reward,
                    step_idx=exp.step_idx,
                    episode_id=exp.episode_id,
                )
                return her_exp
        
        elif self.her_strategy == ReplayStrategy.FINAL:
            # Use final state as goal
            if episode.experiences:
                final_state = episode.experiences[-1].next_state
                hindsight_goal = StructuredGoal(
                    domain=exp.goal.domain,
                    verb="reach",
                    target=final_state[:100],
                    raw_task=f"Reach: {final_state[:100]}",
                )
                hindsight_reward = 1.0 if exp.next_state == final_state else 0.0
                
                her_exp = Experience(
                    state=exp.state,
                    action=exp.action,
                    next_state=exp.next_state,
                    reward=hindsight_reward,
                    done=hindsight_reward > 0,
                    goal=hindsight_goal,
                    hindsight_reward=hindsight_reward,
                    step_idx=exp.step_idx,
                    episode_id=exp.episode_id,
                )
                return her_exp
        
        return exp
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = abs(td_error) + 1e-6
                self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "num_episodes": len(self.episodes),
            "goal_types": list(self.goal_index.keys()),
        }


# ================================ Goal-conditioned Value Function ================================

class GoalConditionedValueFunction:
    """
    Learns Q(s, a, g) - the value of taking action a in state s to achieve goal g.
    
    Uses a combination of:
    - Tabular Q-learning for discrete state/action patterns
    - Embedding-based similarity for generalization
    - LLM-based value estimation for novel situations
    """
    
    def __init__(
        self,
        llm_model: LLMCallable,
        embedding_func: Callable,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        working_dir: str = None,
    ):
        self.llm_model = llm_model
        self.embedding_func = embedding_func
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.working_dir = working_dir
        
        # Q-table: (goal_type, state_pattern, action) -> GoalValueEntry
        self.q_table: Dict[Tuple[str, str, str], GoalValueEntry] = {}
        
        # Embedding cache for similarity-based lookup
        self.state_embeddings: Dict[str, np.ndarray] = {}
        self.action_embeddings: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.update_count = 0
        
        self._load()
    
    def get_q_value(
        self,
        state: str,
        action: str,
        goal: StructuredGoal,
    ) -> float:
        """
        Get Q-value for state-action-goal triple.
        Uses similarity-based lookup for generalization.
        """
        goal_type = goal.verb
        state_pattern = self._extract_state_pattern(state)
        action_pattern = self._extract_action_pattern(action)
        
        key = (goal_type, state_pattern, action_pattern)
        
        if key in self.q_table:
            return self.q_table[key].q_value
        
        # Try similarity-based lookup
        similar_value = self._find_similar_entry(state, action, goal)
        if similar_value is not None:
            return similar_value
        
        # Default neutral value
        return 0.0
    
    def update_q_value(
        self,
        state: str,
        action: str,
        goal: StructuredGoal,
        reward: float,
        next_state: str,
        done: bool,
    ):
        """Update Q-value with TD learning."""
        goal_type = goal.verb
        state_pattern = self._extract_state_pattern(state)
        action_pattern = self._extract_action_pattern(action)
        
        key = (goal_type, state_pattern, action_pattern)
        
        # Get or create entry
        if key not in self.q_table:
            self.q_table[key] = GoalValueEntry(
                goal_type=goal_type,
                state_pattern=state_pattern,
                action=action_pattern,
                q_value=0.0,
            )
        
        entry = self.q_table[key]
        
        # Compute TD target
        if done:
            td_target = reward
        else:
            next_q = self._get_max_q(next_state, goal)
            td_target = reward + self.discount_factor * next_q
        
        # TD update
        td_error = td_target - entry.q_value
        entry.update(td_target, self.learning_rate)
        
        self.update_count += 1
        
        # Periodic save
        if self.update_count % 100 == 0:
            self._save()
        
        return td_error
    
    def get_best_action(
        self,
        state: str,
        goal: StructuredGoal,
        available_actions: List[str] = None,
    ) -> Tuple[str, float]:
        """Get the best action for a state-goal pair."""
        if not available_actions:
            # Get actions from Q-table for this goal type
            available_actions = self._get_known_actions(goal.verb)
        
        if not available_actions:
            return None, 0.0
        
        best_action = None
        best_value = float('-inf')
        
        for action in available_actions:
            q_value = self.get_q_value(state, action, goal)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action, best_value
    
    def batch_update(self, experiences: List[Experience]):
        """Update Q-values from a batch of experiences."""
        td_errors = []
        for exp in experiences:
            td_error = self.update_q_value(
                state=exp.state,
                action=exp.action,
                goal=exp.goal,
                reward=exp.reward,
                next_state=exp.next_state,
                done=exp.done,
            )
            td_errors.append(td_error)
        
        return td_errors
    
    def _get_max_q(self, state: str, goal: StructuredGoal) -> float:
        """Get maximum Q-value over all actions for a state-goal pair."""
        goal_type = goal.verb
        state_pattern = self._extract_state_pattern(state)
        
        max_q = 0.0
        for key, entry in self.q_table.items():
            if key[0] == goal_type and key[1] == state_pattern:
                max_q = max(max_q, entry.q_value)
        
        return max_q
    
    def _extract_state_pattern(self, state: str) -> str:
        """Extract a pattern from state for generalization."""
        # Normalize and extract key elements
        state_lower = state.lower()
        
        # Remove specific numbers and IDs
        import re
        pattern = re.sub(r'\d+', 'N', state_lower)
        pattern = re.sub(r'\s+', ' ', pattern).strip()
        
        # Truncate for efficiency
        return pattern[:200]
    
    def _extract_action_pattern(self, action: str) -> str:
        """Extract pattern from action."""
        action_lower = action.lower().strip()
        
        # Normalize numbers
        import re
        pattern = re.sub(r'\d+', 'N', action_lower)
        
        # Extract action type
        parts = pattern.split()
        if parts:
            return parts[0]  # Just the action verb
        return pattern
    
    def _find_similar_entry(
        self,
        state: str,
        action: str,
        goal: StructuredGoal,
        threshold: float = 0.7,
    ) -> Optional[float]:
        """Find similar entry using embeddings."""
        if not self.embedding_func:
            return None
        
        goal_type = goal.verb
        
        # Compute embeddings
        state_emb = self.embedding_func.embed_query(state)
        
        best_similarity = 0.0
        best_value = None
        
        for key, entry in self.q_table.items():
            if key[0] != goal_type:
                continue
            
            # Check state similarity
            if key[1] in self.state_embeddings:
                cached_emb = self.state_embeddings[key[1]]
            else:
                cached_emb = self.embedding_func.embed_query(key[1])
                self.state_embeddings[key[1]] = cached_emb
            
            similarity = np.dot(state_emb, cached_emb) / (
                np.linalg.norm(state_emb) * np.linalg.norm(cached_emb) + 1e-8
            )
            
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_value = entry.q_value
        
        return best_value
    
    def _get_known_actions(self, goal_type: str) -> List[str]:
        """Get known actions for a goal type."""
        actions = set()
        for key, entry in self.q_table.items():
            if key[0] == goal_type:
                actions.add(entry.action)
        return list(actions)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get value function statistics."""
        goal_types = defaultdict(int)
        for key in self.q_table:
            goal_types[key[0]] += 1
        
        return {
            "num_entries": len(self.q_table),
            "update_count": self.update_count,
            "goal_type_counts": dict(goal_types),
            "avg_q_value": np.mean([e.q_value for e in self.q_table.values()]) if self.q_table else 0.0,
        }
    
    def _save(self):
        """Persist Q-table."""
        if not self.working_dir:
            return
        
        os.makedirs(self.working_dir, exist_ok=True)
        path = os.path.join(self.working_dir, "q_table.pkl")
        
        # Convert to serializable format
        data = {
            key: asdict(entry) for key, entry in self.q_table.items()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load(self):
        """Load Q-table."""
        if not self.working_dir:
            return
        
        path = os.path.join(self.working_dir, "q_table.pkl")
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            for key, entry_dict in data.items():
                self.q_table[key] = GoalValueEntry(**entry_dict)
            
            print(f"Loaded {len(self.q_table)} Q-table entries")
        except Exception as e:
            print(f"Failed to load Q-table: {e}")


# ================================ Hierarchical Goal Manager ================================

class HierarchicalGoalManager:
    """
    Manages hierarchical goal decomposition and subgoal scheduling.
    
    Features:
    - Automatic goal decomposition using LLM
    - Dynamic subgoal reordering based on progress
    - Subgoal dependency tracking
    - Adaptive difficulty estimation
    """
    
    def __init__(
        self,
        llm_model: LLMCallable,
        max_depth: int = 3,
        max_subgoals: int = 5,
    ):
        self.llm_model = llm_model
        self.max_depth = max_depth
        self.max_subgoals = max_subgoals
        
        # Active goal trees
        self.goal_trees: Dict[str, SubgoalNode] = {}
        
        # Subgoal templates learned from experience
        self.subgoal_templates: Dict[str, List[str]] = defaultdict(list)
    
    def decompose_goal(
        self,
        task_id: str,
        main_goal: StructuredGoal,
        initial_state: str,
        depth: int = 0,
    ) -> SubgoalNode:
        """Decompose a goal into hierarchical subgoals."""
        root = SubgoalNode(goal=main_goal)
        
        if depth >= self.max_depth:
            return root
        
        # Check for learned templates first
        if main_goal.verb in self.subgoal_templates:
            templates = self.subgoal_templates[main_goal.verb]
            for template in templates[:self.max_subgoals]:
                subgoal = StructuredGoal(
                    domain=main_goal.domain,
                    verb=main_goal.verb,
                    raw_task=template.format(
                        target=main_goal.target,
                        objects=", ".join(main_goal.objects),
                    ),
                )
                child = SubgoalNode(goal=subgoal, parent=root)
                root.children.append(child)
        else:
            # Use LLM for decomposition
            subgoals = self._llm_decompose(main_goal, initial_state)
            for sg_text in subgoals[:self.max_subgoals]:
                subgoal = StructuredGoal(
                    domain=main_goal.domain,
                    verb=main_goal.verb,
                    raw_task=sg_text,
                )
                child = SubgoalNode(goal=subgoal, parent=root)
                root.children.append(child)
        
        self.goal_trees[task_id] = root
        return root
    
    def _llm_decompose(
        self,
        goal: StructuredGoal,
        state: str,
    ) -> List[str]:
        """Use LLM to decompose goal into subgoals."""
        prompt = f"""Decompose this goal into 3-5 sequential subgoals:

Goal: {goal.raw_task}
Domain: {goal.domain}
Current State: {state[:500]}

List subgoals in order of execution:
1."""

        try:
            response = self.llm_model(
                messages=[
                    Message("system", "You are an expert at task decomposition."),
                    Message("user", prompt)
                ],
                temperature=0.3,
                max_tokens=500,
            )
            
            # Parse numbered list
            import re
            matches = re.findall(r'\d+\.\s*(.+?)(?:\n|$)', response)
            return [m.strip() for m in matches if m.strip()]
        
        except Exception as e:
            print(f"Goal decomposition failed: {e}")
            return []
    
    def get_current_subgoal(self, task_id: str) -> Optional[StructuredGoal]:
        """Get the current active subgoal for a task."""
        if task_id not in self.goal_trees:
            return None
        
        root = self.goal_trees[task_id]
        return self._find_active_subgoal(root)
    
    def _find_active_subgoal(self, node: SubgoalNode) -> Optional[StructuredGoal]:
        """Find the first non-achieved leaf subgoal."""
        if node.status == GoalStatus.ACHIEVED:
            return None
        
        if node.is_leaf():
            if node.status != GoalStatus.ACHIEVED:
                return node.goal
            return None
        
        for child in node.children:
            result = self._find_active_subgoal(child)
            if result:
                return result
        
        return node.goal
    
    def update_subgoal_status(
        self,
        task_id: str,
        achieved_description: str,
    ):
        """Update subgoal status based on achievement description."""
        if task_id not in self.goal_trees:
            return
        
        root = self.goal_trees[task_id]
        self._check_subgoal_achievement(root, achieved_description)
    
    def _check_subgoal_achievement(
        self,
        node: SubgoalNode,
        description: str,
    ):
        """Recursively check and update subgoal achievement."""
        if node.status == GoalStatus.ACHIEVED:
            return
        
        # Simple keyword matching for achievement
        desc_lower = description.lower()
        goal_keywords = node.goal.raw_task.lower().split()
        
        matches = sum(1 for kw in goal_keywords if kw in desc_lower and len(kw) > 3)
        if matches >= len(goal_keywords) * 0.5:
            node.mark_achieved()
        
        # Check children
        for child in node.children:
            self._check_subgoal_achievement(child, description)
    
    def add_template(self, goal_type: str, subgoals: List[str]):
        """Learn a subgoal template from successful execution."""
        self.subgoal_templates[goal_type].extend(subgoals)
        # Keep unique and limit size
        self.subgoal_templates[goal_type] = list(set(
            self.subgoal_templates[goal_type]
        ))[:10]
    
    def get_completion_ratio(self, task_id: str) -> float:
        """Get completion ratio for a task."""
        if task_id not in self.goal_trees:
            return 0.0
        
        root = self.goal_trees[task_id]
        total, completed = self._count_subgoals(root)
        return completed / total if total > 0 else 0.0
    
    def _count_subgoals(self, node: SubgoalNode) -> Tuple[int, int]:
        """Count total and completed subgoals."""
        total = 1
        completed = 1 if node.status == GoalStatus.ACHIEVED else 0
        
        for child in node.children:
            child_total, child_completed = self._count_subgoals(child)
            total += child_total
            completed += child_completed
        
        return total, completed


# ================================ Intrinsic Motivation Module ================================

class IntrinsicMotivationModule:
    """
    Provides intrinsic rewards for exploration and learning.
    
    Implements:
    - Curiosity-driven exploration (state novelty)
    - Information gain rewards
    - Goal-relevance bonuses
    """
    
    def __init__(
        self,
        llm_model: LLMCallable = None,
        embedding_func: Callable = None,
        novelty_weight: float = 0.3,
        info_gain_weight: float = 0.2,
        relevance_weight: float = 0.5,
    ):
        self.llm_model = llm_model
        self.embedding_func = embedding_func
        self.novelty_weight = novelty_weight
        self.info_gain_weight = info_gain_weight
        self.relevance_weight = relevance_weight
        
        # State history for novelty computation
        self.state_history: deque = deque(maxlen=1000)
        self.state_embeddings: List[np.ndarray] = []
        
        # Action history
        self.action_counts: Dict[str, int] = defaultdict(int)
    
    def compute_intrinsic_reward(
        self,
        prev_state: str,
        action: str,
        new_state: str,
        goal: StructuredGoal,
    ) -> float:
        """Compute intrinsic reward for a transition."""
        novelty = self._compute_novelty(new_state)
        info_gain = self._compute_info_gain(prev_state, action, new_state)
        relevance = self._compute_relevance(new_state, action, goal)
        
        intrinsic_reward = (
            self.novelty_weight * novelty +
            self.info_gain_weight * info_gain +
            self.relevance_weight * relevance
        )
        
        # Update history
        self.state_history.append(new_state)
        self.action_counts[action] += 1
        
        return intrinsic_reward
    
    def _compute_novelty(self, state: str) -> float:
        """Compute state novelty (0-1)."""
        if not self.embedding_func or not self.state_embeddings:
            # Fallback: simple string matching
            similar_count = sum(
                1 for s in self.state_history if s == state
            )
            return 1.0 / (1.0 + similar_count)
        
        # Embedding-based novelty
        state_emb = self.embedding_func.embed_query(state)
        
        if not self.state_embeddings:
            self.state_embeddings.append(state_emb)
            return 1.0
        
        # Find max similarity to existing states
        max_sim = 0.0
        for emb in self.state_embeddings[-100:]:  # Recent states
            sim = np.dot(state_emb, emb) / (
                np.linalg.norm(state_emb) * np.linalg.norm(emb) + 1e-8
            )
            max_sim = max(max_sim, sim)
        
        self.state_embeddings.append(state_emb)
        
        # Novelty is inverse of max similarity
        return 1.0 - max_sim
    
    def _compute_info_gain(
        self,
        prev_state: str,
        action: str,
        new_state: str,
    ) -> float:
        """Compute information gain from transition."""
        # Simple heuristic: state change magnitude
        if prev_state == new_state:
            return 0.0
        
        # Reward actions that cause observable change
        change_magnitude = len(set(new_state.split()) - set(prev_state.split()))
        return min(change_magnitude / 10.0, 1.0)
    
    def _compute_relevance(
        self,
        state: str,
        action: str,
        goal: StructuredGoal,
    ) -> float:
        """Compute goal relevance of state/action."""
        # Simple keyword overlap
        state_lower = state.lower()
        action_lower = action.lower()
        
        goal_keywords = set(goal.raw_task.lower().split())
        goal_keywords.update(goal.objects)
        goal_keywords.add(goal.target.lower())
        
        state_words = set(state_lower.split())
        action_words = set(action_lower.split())
        
        state_overlap = len(goal_keywords & state_words)
        action_overlap = len(goal_keywords & action_words)
        
        relevance = (state_overlap + action_overlap * 2) / max(len(goal_keywords), 1)
        return min(relevance, 1.0)
    
    def get_exploration_bonus(self, action: str) -> float:
        """Get bonus for exploring underused actions."""
        count = self.action_counts.get(action, 0)
        total = sum(self.action_counts.values()) + 1
        
        # UCB-style bonus
        bonus = np.sqrt(2 * np.log(total) / (count + 1))
        return min(bonus, 1.0)


# ================================ Goal Achievement Detector ================================

class GoalAchievementDetector:
    """
    Detects when goals or subgoals are achieved.
    Uses LLM for complex goal checking.
    """
    
    def __init__(self, llm_model: LLMCallable):
        self.llm_model = llm_model
        
        # Cache recent checks
        self.check_cache: Dict[str, Tuple[bool, float]] = {}
    
    def check_achievement(
        self,
        goal: StructuredGoal,
        current_state: str,
        recent_actions: List[str] = None,
    ) -> Tuple[bool, float, List[str]]:
        """
        Check if goal is achieved.
        
        Returns:
            Tuple of (achieved, confidence, achieved_subgoals)
        """
        # Create cache key
        cache_key = f"{goal.raw_task[:50]}_{current_state[:50]}"
        if cache_key in self.check_cache:
            achieved, conf = self.check_cache[cache_key]
            return achieved, conf, []
        
        # LLM-based check
        actions_str = "\n".join(recent_actions[-5:]) if recent_actions else "None"
        
        try:
            response = self.llm_model(
                messages=[
                    Message("system", "You are an expert at evaluating task completion."),
                    Message("user", GOAL_ACHIEVEMENT_CHECK_PROMPT.format(
                        goal=goal.to_str(),
                        state=current_state[:1000],
                        actions=actions_str,
                    ))
                ],
                temperature=0.1,
                max_tokens=200,
            )
            
            achieved, confidence, subgoals = self._parse_response(response)
            
            # Cache result
            self.check_cache[cache_key] = (achieved, confidence)
            
            return achieved, confidence, subgoals
        
        except Exception as e:
            print(f"Achievement check failed: {e}")
            return False, 0.0, []
    
    def _parse_response(self, response: str) -> Tuple[bool, float, List[str]]:
        """Parse LLM response."""
        import re
        
        achieved = False
        confidence = 0.5
        subgoals = []
        
        # Parse ACHIEVED
        ach_match = re.search(r'ACHIEVED:\s*(YES|NO|PARTIAL)', response, re.IGNORECASE)
        if ach_match:
            status = ach_match.group(1).upper()
            achieved = status == "YES"
        
        # Parse CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        # Parse ACHIEVED_SUBGOALS
        sub_match = re.search(r'ACHIEVED_SUBGOALS:\s*\[([^\]]+)\]', response)
        if sub_match:
            subgoals = [s.strip() for s in sub_match.group(1).split(',')]
        
        return achieved, confidence, subgoals
    
    def estimate_goal_distance(
        self,
        goal: StructuredGoal,
        current_state: str,
    ) -> float:
        """Estimate distance to goal (0-1, lower is closer)."""
        try:
            response = self.llm_model(
                messages=[
                    Message("system", "Rate how close the state is to the goal."),
                    Message("user", GOAL_DISTANCE_PROMPT.format(
                        goal=goal.to_str(),
                        state=current_state[:500],
                    ))
                ],
                temperature=0.1,
                max_tokens=10,
            )
            
            import re
            match = re.search(r'(\d+)', response)
            if match:
                distance = int(match.group(1))
                return distance / 10.0
        except:
            pass
        
        return 0.5  # Default middle distance


# ================================ Main Goal RL Framework ================================

@dataclass
class GoalRLConfig:
    """Configuration for Goal RL Framework."""
    
    # Replay buffer
    buffer_capacity: int = 10000
    her_ratio: float = 0.8
    her_strategy: ReplayStrategy = ReplayStrategy.FUTURE
    
    # Value function
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    
    # Hierarchical goals
    enable_hierarchical: bool = True
    max_subgoal_depth: int = 3
    
    # Intrinsic motivation
    enable_intrinsic: bool = True
    novelty_weight: float = 0.3
    info_gain_weight: float = 0.2
    relevance_weight: float = 0.5
    
    # Training
    batch_size: int = 32
    train_interval: int = 10  # Train every N steps
    target_update_interval: int = 100


class GoalRLFramework:
    """
    Main Goal Reinforcement Learning Framework.
    
    Integrates all components:
    - Goal-conditioned value function
    - HER replay buffer
    - Hierarchical goal management
    - Intrinsic motivation
    - Goal achievement detection
    """
    
    def __init__(
        self,
        llm_model: LLMCallable,
        embedding_func: Callable,
        goal_parser: GoalParser,
        working_dir: str,
        config: GoalRLConfig = None,
    ):
        self.llm_model = llm_model
        self.embedding_func = embedding_func
        self.goal_parser = goal_parser
        self.working_dir = working_dir
        self.config = config or GoalRLConfig()
        
        # Initialize components
        self.replay_buffer = GoalConditionedReplayBuffer(
            capacity=self.config.buffer_capacity,
            her_ratio=self.config.her_ratio,
            her_strategy=self.config.her_strategy,
            embedding_func=embedding_func,
        )
        
        self.value_function = GoalConditionedValueFunction(
            llm_model=llm_model,
            embedding_func=embedding_func,
            learning_rate=self.config.learning_rate,
            discount_factor=self.config.discount_factor,
            working_dir=os.path.join(working_dir, "value_function"),
        )
        
        if self.config.enable_hierarchical:
            self.goal_manager = HierarchicalGoalManager(
                llm_model=llm_model,
                max_depth=self.config.max_subgoal_depth,
            )
        else:
            self.goal_manager = None
        
        if self.config.enable_intrinsic:
            self.intrinsic_module = IntrinsicMotivationModule(
                llm_model=llm_model,
                embedding_func=embedding_func,
                novelty_weight=self.config.novelty_weight,
                info_gain_weight=self.config.info_gain_weight,
                relevance_weight=self.config.relevance_weight,
            )
        else:
            self.intrinsic_module = None
        
        self.achievement_detector = GoalAchievementDetector(llm_model)
        
        # Current episode tracking
        self.current_episode: Optional[Episode] = None
        self.step_count = 0
        
        # Statistics
        self.total_episodes = 0
        self.successful_episodes = 0
    
    def start_episode(
        self,
        task_id: str,
        goal: StructuredGoal,
        initial_state: str,
    ):
        """Start a new episode."""
        self.current_episode = Episode(
            episode_id=task_id,
            goal=goal,
        )
        
        # Decompose goal if hierarchical enabled
        if self.goal_manager:
            self.goal_manager.decompose_goal(task_id, goal, initial_state)
        
        self.step_count = 0
    
    def step(
        self,
        state: str,
        action: str,
        next_state: str,
        reward: float,
        done: bool,
    ) -> Dict[str, Any]:
        """
        Process one step of interaction.
        
        Returns:
            Dict with computed rewards, Q-values, and guidance.
        """
        if not self.current_episode:
            return {}
        
        goal = self.current_episode.goal
        
        # Compute intrinsic reward
        intrinsic_reward = 0.0
        if self.intrinsic_module:
            intrinsic_reward = self.intrinsic_module.compute_intrinsic_reward(
                state, action, next_state, goal
            )
        
        # Shape reward
        shaped_reward = reward + 0.1 * intrinsic_reward
        
        # Create experience
        exp = Experience(
            state=state,
            action=action,
            next_state=next_state,
            reward=shaped_reward,
            done=done,
            goal=goal,
        )
        
        # Add to episode
        self.current_episode.add_experience(exp)
        self.step_count += 1
        
        # Update value function
        td_error = self.value_function.update_q_value(
            state, action, goal, shaped_reward, next_state, done
        )
        
        # Periodic training from buffer
        if self.step_count % self.config.train_interval == 0:
            self._train_from_buffer()
        
        # Update subgoal progress
        if self.goal_manager:
            self.goal_manager.update_subgoal_status(
                self.current_episode.episode_id,
                next_state,
            )
        
        # Get current subgoal
        current_subgoal = None
        if self.goal_manager:
            current_subgoal = self.goal_manager.get_current_subgoal(
                self.current_episode.episode_id
            )
        
        # Get best next action suggestion
        best_action, q_value = self.value_function.get_best_action(next_state, goal)
        
        return {
            "shaped_reward": shaped_reward,
            "intrinsic_reward": intrinsic_reward,
            "td_error": td_error,
            "q_value": q_value,
            "suggested_action": best_action,
            "current_subgoal": current_subgoal.raw_task if current_subgoal else None,
            "exploration_bonus": self.intrinsic_module.get_exploration_bonus(action) if self.intrinsic_module else 0,
        }
    
    def end_episode(self, success: bool) -> Dict[str, Any]:
        """End the current episode."""
        if not self.current_episode:
            return {}
        
        self.current_episode.success = success
        self.current_episode.final_reward = 1.0 if success else 0.0
        
        # Add to replay buffer with HER
        self.replay_buffer.add_episode(self.current_episode)
        
        # Update statistics
        self.total_episodes += 1
        if success:
            self.successful_episodes += 1
            
            # Learn subgoal templates from success
            if self.goal_manager:
                goal_type = self.current_episode.goal.verb
                subgoals = [
                    exp.action for exp in self.current_episode.experiences
                ]
                self.goal_manager.add_template(goal_type, subgoals[:5])
        
        # Training pass
        self._train_from_buffer()
        
        episode_stats = {
            "episode_id": self.current_episode.episode_id,
            "total_steps": self.current_episode.total_steps,
            "total_reward": self.current_episode.total_reward,
            "success": success,
            "completion_ratio": self.goal_manager.get_completion_ratio(
                self.current_episode.episode_id
            ) if self.goal_manager else 0.0,
        }
        
        self.current_episode = None
        return episode_stats
    
    def get_action_values(
        self,
        state: str,
        goal: StructuredGoal,
        actions: List[str],
    ) -> Dict[str, float]:
        """Get Q-values for a list of actions."""
        values = {}
        for action in actions:
            values[action] = self.value_function.get_q_value(state, action, goal)
        return values
    
    def get_guidance(
        self,
        state: str,
        goal: StructuredGoal,
    ) -> str:
        """Get guidance string for agent."""
        best_action, q_value = self.value_function.get_best_action(state, goal)
        if best_action:
            return f"[Suggested] {best_action} (Q={q_value:.2f})"
        return ""
    
    def _train_from_buffer(self):
        """Train value function from replay buffer."""
        if len(self.replay_buffer.buffer) < self.config.batch_size:
            return
        
        batch = self.replay_buffer.sample_batch(
            self.config.batch_size,
            use_her=True,
        )
        
        td_errors = self.value_function.batch_update(batch)
        
        # Update priorities
        indices = list(range(len(batch)))
        self.replay_buffer.update_priorities(indices, td_errors)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get framework statistics."""
        return {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": self.successful_episodes / max(self.total_episodes, 1),
            "buffer_stats": self.replay_buffer.get_statistics(),
            "value_function_stats": self.value_function.get_statistics(),
        }
    
    def save(self):
        """Save framework state."""
        self.value_function._save()
        
        # Save statistics
        stats_path = os.path.join(self.working_dir, "goal_rl_stats.json")
        with open(stats_path, 'w') as f:
            json.dump({
                "total_episodes": self.total_episodes,
                "successful_episodes": self.successful_episodes,
            }, f)
    
    def load(self):
        """Load framework state."""
        self.value_function._load()
        
        stats_path = os.path.join(self.working_dir, "goal_rl_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                self.total_episodes = stats.get("total_episodes", 0)
                self.successful_episodes = stats.get("successful_episodes", 0)

