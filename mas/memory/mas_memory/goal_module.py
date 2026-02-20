"""
Goal Module for G-Memory++
Parses tasks into structured goals and computes goal-based similarity.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import re
import json
import numpy as np

from mas.llm import LLMCallable, Message


# ================================ Goal Prompts ================================

GOAL_PARSE_SYSTEM_PROMPT = """You are an expert at parsing task descriptions into structured goals.
Given a task description, extract the following information:
1. domain: The task domain (e.g., "alfworld", "gui", "compbench", "pddl", "fever", "hotpotqa")
2. verb: The main action type (e.g., "put_clean_then_place", "pick_and_place", "click", "search", "verify")
3. objects: List of objects involved in the task
4. target: The target location or state
5. preconditions: What must be true before the task
6. postconditions: What must be true after the task

Output your response as a valid JSON object with these exact keys.
"""

GOAL_PARSE_USER_PROMPT = """Parse the following task into a structured goal:

Task: {task_description}

Context (if any): {context}

Output the structured goal as JSON:
"""

GOAL_SIMILARITY_SYSTEM_PROMPT = """You are an expert at comparing task goals.
Given two task goals, rate their similarity on a scale of 0-10 where:
- 0: Completely different tasks
- 5: Similar domain but different objectives
- 10: Nearly identical tasks

Consider:
1. Same domain/environment
2. Same action type (verb)
3. Similar objects involved
4. Similar target states
"""

GOAL_SIMILARITY_USER_PROMPT = """Rate the similarity between these two goals:

Goal 1: {goal1}

Goal 2: {goal2}

Respond with only a number from 0-10:
"""


# ================================ Data Classes ================================

@dataclass
class StructuredGoal:
    """Represents a parsed, structured goal from a task description."""
    
    domain: str = "unknown"  # alfworld, gui, compbench, pddl, fever, hotpotqa
    verb: str = "unknown"    # put_clean_then_place, click, search, verify, etc.
    objects: List[str] = field(default_factory=list)
    target: str = ""
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    difficulty: float = 0.5
    raw_task: str = ""
    
    # Optional visual context for GUI tasks
    visual_context: Optional[str] = None
    visual_embedding: Optional[np.ndarray] = None
    
    def to_str(self) -> str:
        """Convert goal to a readable string."""
        parts = [
            f"Domain: {self.domain}",
            f"Action: {self.verb}",
            f"Objects: {', '.join(self.objects) if self.objects else 'none'}",
            f"Target: {self.target}",
        ]
        if self.preconditions:
            parts.append(f"Preconditions: {', '.join(self.preconditions)}")
        if self.postconditions:
            parts.append(f"Postconditions: {', '.join(self.postconditions)}")
        if self.visual_context:
            parts.append(f"Visual Context: {self.visual_context}")
        return "\n".join(parts)
    
    def to_features(self) -> np.ndarray:
        """Convert goal to a feature vector for GCN."""
        # Simple one-hot encoding for domain and verb
        domains = ["alfworld", "gui", "compbench", "pddl", "fever", "hotpotqa", "unknown"]
        verbs = ["put_clean_then_place", "pick_and_place", "pick_heat_then_place", 
                 "pick_cool_then_place", "look_at_obj", "pick_two_obj",
                 "click", "scroll", "type", "search", "verify", "edit", "run", "unknown"]
        
        domain_vec = [1.0 if d == self.domain else 0.0 for d in domains]
        verb_vec = [1.0 if v == self.verb else 0.0 for v in verbs]
        
        # Add difficulty and object count
        extra_features = [
            self.difficulty,
            len(self.objects) / 10.0,  # Normalized object count
            len(self.preconditions) / 5.0,
            len(self.postconditions) / 5.0,
        ]
        
        return np.array(domain_vec + verb_vec + extra_features, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain": self.domain,
            "verb": self.verb,
            "objects": self.objects,
            "target": self.target,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "difficulty": self.difficulty,
            "raw_task": self.raw_task,
            "visual_context": self.visual_context,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StructuredGoal":
        """Create from dictionary."""
        return StructuredGoal(
            domain=data.get("domain", "unknown"),
            verb=data.get("verb", "unknown"),
            objects=data.get("objects", []),
            target=data.get("target", ""),
            preconditions=data.get("preconditions", []),
            postconditions=data.get("postconditions", []),
            difficulty=data.get("difficulty", 0.5),
            raw_task=data.get("raw_task", ""),
            visual_context=data.get("visual_context"),
        )


# ================================ Goal Parser ================================

class GoalParser:
    """Parses task descriptions into structured goals."""
    
    # Domain detection patterns
    DOMAIN_PATTERNS = {
        "alfworld": [
            r"you are in the middle of a room",
            r"your task is to.*(?:put|heat|cool|clean|examine|look at)",
            r"(?:cabinet|fridge|microwave|countertop|drawer)",
        ],
        "pddl": [
            r"the goal is to satisfy",
            r"(?:block|stack|unstack|pick-up|put-down)",
            r"(?:on\s+\w+|clear\s+\w+|holding)",
        ],
        "fever": [
            r"(?:verify|check|confirm|fact)",
            r"(?:true|false|supported|refuted)",
        ],
        "hotpotqa": [
            r"(?:who|what|where|when|how|why)",
            r"(?:compare|both|either|neither)",
        ],
        "gui": [
            r"(?:click|tap|scroll|swipe)",
            r"(?:button|icon|menu|window|screen)",
            r"(?:settings|app|browser)",
        ],
        "compbench": [
            r"(?:open|run|execute|edit|save)",
            r"(?:file|terminal|browser|command)",
        ],
    }
    
    # ALFWorld verb patterns
    ALFWORLD_VERB_PATTERNS = {
        "put_clean_then_place": [r"clean.*put", r"put.*clean"],
        "pick_heat_then_place": [r"heat.*put", r"put.*heat"],
        "pick_cool_then_place": [r"cool.*put", r"put.*cool"],
        "pick_and_place": [r"put\s+\w+\s+(?:in|on)"],
        "look_at_obj": [r"look\s+at", r"examine"],
        "pick_two_obj": [r"two", r"both"],
    }
    
    def __init__(self, llm_model: Optional[LLMCallable] = None):
        self.llm_model = llm_model
    
    def parse(self, task_main: str, task_description: str = "", 
              context: str = "", use_llm: bool = False) -> StructuredGoal:
        """
        Parse a task into a structured goal.
        
        Args:
            task_main: The main task string
            task_description: Additional task description
            context: Optional context (e.g., environment state)
            use_llm: Whether to use LLM for parsing (more accurate but slower)
        
        Returns:
            StructuredGoal object
        """
        full_task = f"{task_main} {task_description}".lower()
        
        if use_llm and self.llm_model:
            return self._parse_with_llm(full_task, context)
        else:
            return self._parse_with_rules(full_task, task_main)
    
    def _parse_with_rules(self, full_task: str, raw_task: str) -> StructuredGoal:
        """Rule-based parsing (fast, no API calls)."""
        
        # Detect domain
        domain = self._detect_domain(full_task)
        
        # Detect verb based on domain
        verb = self._detect_verb(full_task, domain)
        
        # Extract objects
        objects = self._extract_objects(full_task, domain)
        
        # Extract target
        target = self._extract_target(full_task, domain)
        
        # Estimate difficulty
        difficulty = self._estimate_difficulty(full_task, domain, verb)
        
        # Extract conditions
        preconditions, postconditions = self._extract_conditions(full_task, domain, verb)
        
        return StructuredGoal(
            domain=domain,
            verb=verb,
            objects=objects,
            target=target,
            preconditions=preconditions,
            postconditions=postconditions,
            difficulty=difficulty,
            raw_task=raw_task,
        )
    
    def _parse_with_llm(self, full_task: str, context: str) -> StructuredGoal:
        """LLM-based parsing (more accurate)."""
        if not self.llm_model:
            return self._parse_with_rules(full_task, full_task)
        
        prompt = GOAL_PARSE_USER_PROMPT.format(
            task_description=full_task,
            context=context or "None"
        )
        
        try:
            response = self.llm_model(
                messages=[
                    Message("system", GOAL_PARSE_SYSTEM_PROMPT),
                    Message("user", prompt)
                ],
                temperature=0.1
            )
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return StructuredGoal(
                    domain=data.get("domain", "unknown"),
                    verb=data.get("verb", "unknown"),
                    objects=data.get("objects", []),
                    target=data.get("target", ""),
                    preconditions=data.get("preconditions", []),
                    postconditions=data.get("postconditions", []),
                    raw_task=full_task,
                )
        except Exception as e:
            print(f"LLM parsing failed: {e}, falling back to rules")
        
        return self._parse_with_rules(full_task, full_task)
    
    def _detect_domain(self, task: str) -> str:
        """Detect task domain from patterns."""
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, task, re.IGNORECASE):
                    return domain
        return "unknown"
    
    def _detect_verb(self, task: str, domain: str) -> str:
        """Detect action verb based on domain."""
        if domain == "alfworld":
            for verb, patterns in self.ALFWORLD_VERB_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, task, re.IGNORECASE):
                        return verb
            return "pick_and_place"
        
        elif domain == "pddl":
            if "stack" in task:
                return "stack"
            elif "unstack" in task:
                return "unstack"
            return "arrange"
        
        elif domain == "gui":
            if "click" in task:
                return "click"
            elif "scroll" in task:
                return "scroll"
            elif "type" in task:
                return "type"
            return "interact"
        
        elif domain == "fever":
            return "verify"
        
        elif domain == "hotpotqa":
            return "search"
        
        return "unknown"
    
    def _extract_objects(self, task: str, domain: str) -> List[str]:
        """Extract objects mentioned in the task."""
        objects = []
        
        if domain == "alfworld":
            # Common ALFWorld objects
            obj_patterns = [
                r'\b(apple|egg|potato|tomato|bread|lettuce|mug|cup|bowl|plate|knife|fork|spoon|spatula|pan|pot|book|pen|pencil|cd|laptop|remote|cloth|towel|soap|candle|pillow|watch|cellphone|creditcard|newspaper|vase|box|basket|desk|table|shelf|drawer|cabinet|fridge|microwave|stove|sink|toilet|bed|sofa|armchair|lamp|desklamp|floorlamp)\s*\d*\b'
            ]
            for pattern in obj_patterns:
                matches = re.findall(pattern, task, re.IGNORECASE)
                objects.extend(matches)
        
        elif domain == "pddl":
            # Block world objects
            block_matches = re.findall(r'\b(b\d+|block\s*\d+)\b', task, re.IGNORECASE)
            objects.extend(block_matches)
        
        return list(set(objects))
    
    def _extract_target(self, task: str, domain: str) -> str:
        """Extract target location or state."""
        if domain == "alfworld":
            # Look for "in/on [location]" patterns
            match = re.search(r'(?:put|place).*(?:in|on)\s+(\w+\s*\d*)', task, re.IGNORECASE)
            if match:
                return match.group(1)
        
        elif domain == "pddl":
            # Look for goal state
            match = re.search(r'(?:goal|satisfy).*?:\s*(.+?)(?:\.|$)', task, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _estimate_difficulty(self, task: str, domain: str, verb: str) -> float:
        """Estimate task difficulty (0-1)."""
        difficulty = 0.5
        
        # More objects = harder
        num_objects = len(re.findall(r'\b(?:a|an|the)\s+\w+', task))
        difficulty += min(num_objects * 0.05, 0.2)
        
        # Multi-step tasks are harder
        if verb in ["put_clean_then_place", "pick_heat_then_place", "pick_cool_then_place"]:
            difficulty += 0.2
        
        # Two objects = harder
        if verb == "pick_two_obj":
            difficulty += 0.15
        
        return min(difficulty, 1.0)
    
    def _extract_conditions(self, task: str, domain: str, verb: str) -> tuple:
        """Extract pre and post conditions."""
        preconditions = []
        postconditions = []
        
        if domain == "alfworld":
            if "clean" in verb:
                preconditions.append("object is dirty or needs cleaning")
                postconditions.append("object is clean")
            if "heat" in verb:
                preconditions.append("object is cold")
                postconditions.append("object is heated")
            if "cool" in verb:
                preconditions.append("object is warm")
                postconditions.append("object is cooled")
            if "put" in verb or "place" in verb:
                postconditions.append("object is at target location")
        
        return preconditions, postconditions
    
    def compute_similarity(self, goal1: StructuredGoal, goal2: StructuredGoal,
                          use_llm: bool = False) -> float:
        """
        Compute similarity between two goals.
        
        Args:
            goal1: First goal
            goal2: Second goal
            use_llm: Whether to use LLM for similarity (more accurate)
        
        Returns:
            Similarity score (0-1)
        """
        if use_llm and self.llm_model:
            return self._compute_similarity_llm(goal1, goal2)
        else:
            return self._compute_similarity_rules(goal1, goal2)
    
    def _compute_similarity_rules(self, goal1: StructuredGoal, goal2: StructuredGoal) -> float:
        """Rule-based similarity computation."""
        score = 0.0
        
        # Domain match (40%)
        if goal1.domain == goal2.domain:
            score += 0.4
        
        # Verb match (30%)
        if goal1.verb == goal2.verb:
            score += 0.3
        elif self._verbs_related(goal1.verb, goal2.verb):
            score += 0.15
        
        # Object overlap (20%)
        if goal1.objects and goal2.objects:
            obj1_set = set(o.lower() for o in goal1.objects)
            obj2_set = set(o.lower() for o in goal2.objects)
            overlap = len(obj1_set & obj2_set) / max(len(obj1_set | obj2_set), 1)
            score += 0.2 * overlap
        
        # Target similarity (10%)
        if goal1.target and goal2.target:
            if goal1.target.lower() == goal2.target.lower():
                score += 0.1
        
        return score
    
    def _compute_similarity_llm(self, goal1: StructuredGoal, goal2: StructuredGoal) -> float:
        """LLM-based similarity computation."""
        if not self.llm_model:
            return self._compute_similarity_rules(goal1, goal2)
        
        prompt = GOAL_SIMILARITY_USER_PROMPT.format(
            goal1=goal1.to_str(),
            goal2=goal2.to_str()
        )
        
        try:
            response = self.llm_model(
                messages=[
                    Message("system", GOAL_SIMILARITY_SYSTEM_PROMPT),
                    Message("user", prompt)
                ],
                temperature=0.1
            )
            
            # Extract number
            match = re.search(r'\d+', response)
            if match:
                return int(match.group()) / 10.0
        except Exception as e:
            print(f"LLM similarity failed: {e}")
        
        return self._compute_similarity_rules(goal1, goal2)
    
    def _verbs_related(self, verb1: str, verb2: str) -> bool:
        """Check if two verbs are related."""
        related_groups = [
            {"put_clean_then_place", "pick_and_place", "pick_heat_then_place", "pick_cool_then_place"},
            {"click", "tap", "interact"},
            {"search", "verify", "find"},
        ]
        
        for group in related_groups:
            if verb1 in group and verb2 in group:
                return True
        return False


# ================================ Goal Matcher ================================

class GoalMatcher:
    """Matches new goals against historical goals for retrieval."""
    
    def __init__(self, goal_parser: GoalParser):
        self.goal_parser = goal_parser
    
    def rank_by_goal_similarity(self, query_goal: StructuredGoal,
                                 candidate_goals: List[StructuredGoal],
                                 top_k: int = 5) -> List[tuple]:
        """
        Rank candidate goals by similarity to query goal.
        
        Args:
            query_goal: The goal to match against
            candidate_goals: List of candidate goals
            top_k: Number of top matches to return
        
        Returns:
            List of (goal, similarity_score) tuples, sorted by score
        """
        scored = []
        for candidate in candidate_goals:
            score = self.goal_parser.compute_similarity(query_goal, candidate)
            scored.append((candidate, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_k]
    
    def filter_by_domain(self, goals: List[StructuredGoal], 
                         domain: str) -> List[StructuredGoal]:
        """Filter goals by domain."""
        return [g for g in goals if g.domain == domain]
    
    def filter_by_verb(self, goals: List[StructuredGoal],
                       verb: str) -> List[StructuredGoal]:
        """Filter goals by verb/action type."""
        return [g for g in goals if g.verb == verb]

