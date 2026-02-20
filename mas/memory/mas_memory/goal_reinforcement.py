"""
Goal Reinforcement Module for G-Memory++
Periodically checks if agents are on track with the goal and provides corrective guidance.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import re

from mas.llm import LLMCallable, Message
from .goal_module import StructuredGoal


# ================================ Goal Reinforcement Prompts ================================

GOAL_CHECK_SYSTEM_PROMPT = """You are an expert at evaluating task execution progress.
Given a task goal and the current execution trajectory, determine:
1. Whether the agent is making progress toward the goal
2. What the next subgoal should be
3. If off-track, what corrective action to suggest

Be concise and actionable in your feedback.
"""

GOAL_CHECK_USER_PROMPT = """## Task Goal:
{goal}

## Current Trajectory:
{trajectory}

## Current State:
{current_state}

## Questions:
1. Is the agent on track toward the goal? (YES/NO/PARTIALLY)
2. What is the immediate next subgoal?
3. If off-track, what corrective hint should be given?

Format your response as:
ON_TRACK: [YES/NO/PARTIALLY]
PROGRESS: [brief description of progress made]
NEXT_SUBGOAL: [what should be done next]
CORRECTIVE_HINT: [hint if off-track, or "None" if on track]
"""

DRIFT_DETECTION_PROMPT = """Analyze the following trajectory for signs of task drift or loops:

## Original Goal:
{goal}

## Trajectory:
{trajectory}

Signs to look for:
- Repeated actions that don't change state
- Actions unrelated to the goal
- Circular patterns (going back to previous states)
- Excessive exploration without progress

Is there task drift or looping? Respond with:
DRIFT_DETECTED: [YES/NO]
DRIFT_TYPE: [LOOP/UNRELATED/CIRCULAR/EXPLORATION/NONE]
RECOMMENDATION: [what to do next]
"""

SUBGOAL_DECOMPOSITION_PROMPT = """Decompose the following goal into subgoals that can be tracked:

## Main Goal:
{goal}

## Current State:
{current_state}

List 3-5 ordered subgoals that would lead to task completion.
Format each as:
1. [subgoal] - [success criterion]
2. [subgoal] - [success criterion]
...
"""


# ================================ Data Classes ================================

@dataclass
class GoalCheckResult:
    """Result of a goal progress check."""
    
    on_track: bool
    progress_status: str  # "YES", "NO", "PARTIALLY"
    progress_description: str
    next_subgoal: str
    corrective_hint: Optional[str] = None
    drift_detected: bool = False
    drift_type: Optional[str] = None
    confidence: float = 1.0
    
    def needs_intervention(self) -> bool:
        """Check if intervention is needed."""
        return not self.on_track or self.drift_detected


@dataclass
class SubgoalProgress:
    """Tracks progress on a subgoal."""
    
    subgoal: str
    success_criterion: str
    completed: bool = False
    attempts: int = 0
    last_check_step: int = 0


@dataclass
class GoalTracker:
    """Tracks overall goal progress with subgoals."""
    
    main_goal: StructuredGoal
    subgoals: List[SubgoalProgress] = field(default_factory=list)
    current_subgoal_idx: int = 0
    total_steps: int = 0
    interventions: int = 0
    
    def current_subgoal(self) -> Optional[SubgoalProgress]:
        if 0 <= self.current_subgoal_idx < len(self.subgoals):
            return self.subgoals[self.current_subgoal_idx]
        return None
    
    def advance_subgoal(self):
        """Mark current subgoal as complete and move to next."""
        if self.current_subgoal_idx < len(self.subgoals):
            self.subgoals[self.current_subgoal_idx].completed = True
            self.current_subgoal_idx += 1
    
    def completion_ratio(self) -> float:
        """Get ratio of completed subgoals."""
        if not self.subgoals:
            return 0.0
        completed = sum(1 for s in self.subgoals if s.completed)
        return completed / len(self.subgoals)


# ================================ Goal Reinforcement Checker ================================

class GoalReinforcementChecker:
    """
    Checks goal progress during task execution and provides corrective guidance.
    
    Features:
    - Periodic goal progress checks
    - Drift and loop detection
    - Subgoal tracking
    - Corrective hint generation
    """
    
    def __init__(
        self,
        llm_model: LLMCallable,
        check_interval: int = 3,  # Check every N steps
        max_stuck_steps: int = 5,  # Max steps without progress before intervention
        enable_subgoal_tracking: bool = True
    ):
        self.llm_model = llm_model
        self.check_interval = check_interval
        self.max_stuck_steps = max_stuck_steps
        self.enable_subgoal_tracking = enable_subgoal_tracking
        
        # Active trackers
        self.active_trackers: Dict[str, GoalTracker] = {}
        
        # History for pattern detection
        self.recent_actions: List[str] = []
        self.recent_states: List[str] = []
    
    def start_tracking(
        self,
        task_id: str,
        goal: StructuredGoal,
        initial_state: str = ""
    ) -> GoalTracker:
        """
        Start tracking a new task.
        
        Args:
            task_id: Unique task identifier
            goal: The structured goal to track
            initial_state: Initial environment state
        
        Returns:
            GoalTracker instance
        """
        tracker = GoalTracker(main_goal=goal)
        
        # Decompose into subgoals if enabled
        if self.enable_subgoal_tracking:
            subgoals = self._decompose_goal(goal, initial_state)
            tracker.subgoals = subgoals
        
        self.active_trackers[task_id] = tracker
        self.recent_actions = []
        self.recent_states = []
        
        return tracker
    
    def check_progress(
        self,
        task_id: str,
        trajectory: str,
        current_state: str,
        step_number: int,
        force_check: bool = False
    ) -> Optional[GoalCheckResult]:
        """
        Check if the agent is making progress toward the goal.
        
        Args:
            task_id: Task identifier
            trajectory: Current trajectory text
            current_state: Current environment state
            step_number: Current step number
            force_check: Force check even if not at interval
        
        Returns:
            GoalCheckResult if check was performed, None otherwise
        """
        tracker = self.active_trackers.get(task_id)
        if not tracker:
            return None
        
        tracker.total_steps = step_number
        
        # Check if we should run a check
        if not force_check and step_number % self.check_interval != 0:
            return None
        
        # Perform the check
        result = self._perform_check(tracker, trajectory, current_state)
        
        # Update tracker based on result
        if result.needs_intervention():
            tracker.interventions += 1
        
        # Update subgoal progress if applicable
        if self.enable_subgoal_tracking and result.on_track:
            self._update_subgoal_progress(tracker, current_state)
        
        return result
    
    def record_action(self, action: str, state: str):
        """Record an action for pattern detection."""
        self.recent_actions.append(action)
        self.recent_states.append(state)
        
        # Keep only recent history
        max_history = 20
        self.recent_actions = self.recent_actions[-max_history:]
        self.recent_states = self.recent_states[-max_history:]
    
    def detect_loop(self, window_size: int = 5) -> bool:
        """
        Detect if the agent is stuck in a loop.
        
        Args:
            window_size: Number of recent actions to check
        
        Returns:
            True if loop detected
        """
        if len(self.recent_actions) < window_size * 2:
            return False
        
        recent = self.recent_actions[-window_size:]
        previous = self.recent_actions[-window_size*2:-window_size]
        
        # Check if recent actions are the same as previous window
        if recent == previous:
            return True
        
        # Check for repeated single action
        if len(set(recent)) == 1:
            return True
        
        return False
    
    def get_corrective_guidance(
        self,
        task_id: str,
        check_result: GoalCheckResult,
        insights: List[str] = None
    ) -> str:
        """
        Generate corrective guidance based on check result.
        
        Args:
            task_id: Task identifier
            check_result: Result from check_progress
            insights: Relevant insights from memory
        
        Returns:
            Guidance string to inject into agent context
        """
        tracker = self.active_trackers.get(task_id)
        if not tracker:
            return ""
        
        parts = []
        
        # Add goal reminder
        parts.append(f"[Goal Reminder] Your objective: {tracker.main_goal.raw_task}")
        
        # Add progress status
        if check_result.progress_status == "PARTIALLY":
            parts.append(f"[Progress] {check_result.progress_description}")
        
        # Add next subgoal
        if check_result.next_subgoal:
            parts.append(f"[Next Step] {check_result.next_subgoal}")
        
        # Add corrective hint if off-track
        if check_result.corrective_hint and check_result.corrective_hint.lower() != "none":
            parts.append(f"[Hint] {check_result.corrective_hint}")
        
        # Add loop warning if detected
        if check_result.drift_detected:
            if check_result.drift_type == "LOOP":
                parts.append("[Warning] Repeated actions detected. Try a different approach.")
            elif check_result.drift_type == "EXPLORATION":
                parts.append("[Warning] Focus on the goal instead of exploring.")
        
        # Add relevant insight if available
        if insights:
            parts.append(f"[Tip] {insights[0]}")
        
        return "\n".join(parts)
    
    def end_tracking(self, task_id: str) -> Optional[GoalTracker]:
        """End tracking for a task and return final tracker state."""
        return self.active_trackers.pop(task_id, None)
    
    def _perform_check(
        self,
        tracker: GoalTracker,
        trajectory: str,
        current_state: str
    ) -> GoalCheckResult:
        """Perform the actual progress check using LLM."""
        goal = tracker.main_goal
        
        # Check for loops first (cheap)
        loop_detected = self.detect_loop()
        
        # LLM-based check
        try:
            response = self.llm_model(
                messages=[
                    Message("system", GOAL_CHECK_SYSTEM_PROMPT),
                    Message("user", GOAL_CHECK_USER_PROMPT.format(
                        goal=goal.to_str(),
                        trajectory=trajectory[-2000:],  # Limit trajectory length
                        current_state=current_state
                    ))
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result = self._parse_check_response(response)
            
            # Add loop detection result
            if loop_detected:
                result.drift_detected = True
                result.drift_type = "LOOP"
            
            return result
            
        except Exception as e:
            print(f"Goal check failed: {e}")
            return GoalCheckResult(
                on_track=True,  # Assume on track if check fails
                progress_status="UNKNOWN",
                progress_description="Check failed",
                next_subgoal="Continue with current approach",
                confidence=0.0
            )
    
    def _parse_check_response(self, response: str) -> GoalCheckResult:
        """Parse LLM response into GoalCheckResult."""
        
        # Defaults
        on_track = True
        progress_status = "YES"
        progress_description = ""
        next_subgoal = ""
        corrective_hint = None
        
        # Parse ON_TRACK
        track_match = re.search(r'ON_TRACK:\s*(YES|NO|PARTIALLY)', response, re.IGNORECASE)
        if track_match:
            progress_status = track_match.group(1).upper()
            on_track = progress_status == "YES"
        
        # Parse PROGRESS
        progress_match = re.search(r'PROGRESS:\s*(.+?)(?:\n|$)', response)
        if progress_match:
            progress_description = progress_match.group(1).strip()
        
        # Parse NEXT_SUBGOAL
        subgoal_match = re.search(r'NEXT_SUBGOAL:\s*(.+?)(?:\n|$)', response)
        if subgoal_match:
            next_subgoal = subgoal_match.group(1).strip()
        
        # Parse CORRECTIVE_HINT
        hint_match = re.search(r'CORRECTIVE_HINT:\s*(.+?)(?:\n|$)', response)
        if hint_match:
            hint = hint_match.group(1).strip()
            if hint.lower() != "none":
                corrective_hint = hint
        
        return GoalCheckResult(
            on_track=on_track,
            progress_status=progress_status,
            progress_description=progress_description,
            next_subgoal=next_subgoal,
            corrective_hint=corrective_hint
        )
    
    def _decompose_goal(
        self,
        goal: StructuredGoal,
        initial_state: str
    ) -> List[SubgoalProgress]:
        """Decompose goal into trackable subgoals."""
        try:
            response = self.llm_model(
                messages=[
                    Message("system", "You are an expert at task decomposition."),
                    Message("user", SUBGOAL_DECOMPOSITION_PROMPT.format(
                        goal=goal.to_str(),
                        current_state=initial_state or "Unknown"
                    ))
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return self._parse_subgoals(response)
            
        except Exception as e:
            print(f"Subgoal decomposition failed: {e}")
            return []
    
    def _parse_subgoals(self, response: str) -> List[SubgoalProgress]:
        """Parse subgoals from LLM response."""
        subgoals = []
        
        # Match numbered list items
        pattern = r'\d+\.\s*(.+?)\s*-\s*(.+?)(?:\n|$)'
        matches = re.findall(pattern, response)
        
        for subgoal, criterion in matches:
            subgoals.append(SubgoalProgress(
                subgoal=subgoal.strip(),
                success_criterion=criterion.strip()
            ))
        
        return subgoals
    
    def _update_subgoal_progress(self, tracker: GoalTracker, current_state: str):
        """Update subgoal completion based on current state."""
        current = tracker.current_subgoal()
        if not current:
            return
        
        current.attempts += 1
        
        # Simple check: if criterion keywords appear in state
        criterion_lower = current.success_criterion.lower()
        state_lower = current_state.lower()
        
        # Look for key indicators
        keywords = criterion_lower.split()
        matches = sum(1 for kw in keywords if kw in state_lower and len(kw) > 3)
        
        if matches >= len(keywords) * 0.5:  # At least 50% match
            tracker.advance_subgoal()

