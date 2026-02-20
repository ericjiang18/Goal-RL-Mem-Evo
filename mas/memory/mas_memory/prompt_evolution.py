"""
Prompt Evolution Module for G-Memory++
Manages and evolves prompts for each agent role based on success/failure feedback.
Uses bandit-style selection and LLM-based synthesis for prompt improvement.
"""

import os
import json
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from collections import defaultdict

from mas.llm import LLMCallable, Message


# ================================ Prompts for Evolution ================================

PROMPT_SYNTHESIS_SYSTEM = """You are an expert at improving agent prompts based on performance data.
Given:
1. The best-performing prompt variants for a role
2. Insights/lessons learned from successful task executions
3. Common failure patterns

Your task is to synthesize an improved prompt that:
- Incorporates the successful patterns from top-performing variants
- Addresses the common failure patterns
- Is clear, concise, and actionable
- Maintains the core role identity

Output only the improved prompt text, no explanations.
"""

PROMPT_SYNTHESIS_USER = """## Role: {role}

## Top Performing Prompts (sorted by success rate):
{top_prompts}

## Relevant Insights from Successful Tasks:
{insights}

## Common Failure Patterns to Avoid:
{failure_patterns}

## Domain Context: {domain}

Generate an improved prompt for this role:
"""

PROMPT_INJECT_INSIGHT_TEMPLATE = """
## Important Guidelines Based on Experience:
{insights}
"""


# ================================ Data Classes ================================

@dataclass
class PromptVariant:
    """Represents a single prompt variant with performance statistics."""
    
    role: str
    version: int
    content: str
    domain: str = "general"
    
    # Performance statistics
    success_count: int = 0
    failure_count: int = 0
    total_tokens: int = 0
    
    # Metadata
    created_from: str = "default"  # "default", "evolved", "manual"
    parent_version: Optional[int] = None
    
    @property
    def total_uses(self) -> int:
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.5  # Prior for unexplored variants
        return self.success_count / self.total_uses
    
    @property
    def avg_tokens(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.total_tokens / self.total_uses
    
    def ucb_score(self, total_trials: int, c: float = 2.0) -> float:
        """
        Upper Confidence Bound score for exploration/exploitation balance.
        Higher c = more exploration.
        """
        if self.total_uses == 0:
            return float('inf')  # Unexplored = high priority
        
        exploitation = self.success_rate
        exploration = c * np.sqrt(np.log(total_trials + 1) / self.total_uses)
        
        return exploitation + exploration
    
    def thompson_sample(self) -> float:
        """
        Thompson Sampling: sample from Beta distribution.
        Returns a sample success rate for probabilistic selection.
        """
        # Beta prior: α = success + 1, β = failure + 1
        alpha = self.success_count + 1
        beta = self.failure_count + 1
        return np.random.beta(alpha, beta)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "version": self.version,
            "content": self.content,
            "domain": self.domain,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_tokens": self.total_tokens,
            "created_from": self.created_from,
            "parent_version": self.parent_version,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptVariant":
        return PromptVariant(
            role=data["role"],
            version=data["version"],
            content=data["content"],
            domain=data.get("domain", "general"),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            total_tokens=data.get("total_tokens", 0),
            created_from=data.get("created_from", "default"),
            parent_version=data.get("parent_version"),
        )


@dataclass
class PromptEvolutionConfig:
    """Configuration for prompt evolution."""
    
    # Selection strategy: "ucb", "thompson", "epsilon_greedy", "best"
    selection_strategy: str = "thompson"
    
    # UCB exploration parameter
    ucb_c: float = 2.0
    
    # Epsilon for epsilon-greedy
    epsilon: float = 0.1
    
    # Evolution triggers
    min_trials_before_evolution: int = 10
    evolve_every_n_trials: int = 20
    
    # Variant limits
    max_variants_per_role: int = 10
    
    # Pruning
    prune_below_success_rate: float = 0.3
    min_trials_before_prune: int = 5


# ================================ Prompt Evolution Manager ================================

class PromptEvolutionManager:
    """
    Manages prompt variants for all agent roles.
    Implements bandit-style selection and LLM-based evolution.
    """
    
    def __init__(
        self,
        llm_model: LLMCallable,
        working_dir: str,
        config: Optional[PromptEvolutionConfig] = None
    ):
        self.llm_model = llm_model
        self.working_dir = working_dir
        self.config = config or PromptEvolutionConfig()
        
        # Storage: role -> domain -> list of variants
        self.variants: Dict[str, Dict[str, List[PromptVariant]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Track total trials per role-domain pair
        self.total_trials: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        # Failure patterns collected for evolution
        self.failure_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # File path for persistence
        self.persist_path = os.path.join(working_dir, "prompt_evolution.json")
        
        # Load existing data
        self._load()
    
    def register_default_prompt(self, role: str, content: str, domain: str = "general"):
        """
        Register a default prompt for a role.
        Called during system initialization.
        """
        if not self.variants[role][domain]:
            variant = PromptVariant(
                role=role,
                version=0,
                content=content,
                domain=domain,
                created_from="default"
            )
            self.variants[role][domain].append(variant)
            self._save()
    
    def select_prompt(self, role: str, domain: str = "general") -> PromptVariant:
        """
        Select a prompt variant using the configured strategy.
        
        Args:
            role: Agent role name
            domain: Task domain (e.g., "alfworld", "gui")
        
        Returns:
            Selected PromptVariant
        """
        variants = self.variants[role][domain]
        
        if not variants:
            # No variants registered, return a placeholder
            return PromptVariant(
                role=role, version=0, 
                content=f"You are a {role} agent.", 
                domain=domain
            )
        
        if len(variants) == 1:
            return variants[0]
        
        total = self.total_trials[role][domain]
        
        if self.config.selection_strategy == "ucb":
            # Upper Confidence Bound selection
            scores = [(v, v.ucb_score(total, self.config.ucb_c)) for v in variants]
            return max(scores, key=lambda x: x[1])[0]
        
        elif self.config.selection_strategy == "thompson":
            # Thompson Sampling
            samples = [(v, v.thompson_sample()) for v in variants]
            return max(samples, key=lambda x: x[1])[0]
        
        elif self.config.selection_strategy == "epsilon_greedy":
            # Epsilon-greedy
            if random.random() < self.config.epsilon:
                return random.choice(variants)
            else:
                return max(variants, key=lambda v: v.success_rate)
        
        else:  # "best"
            return max(variants, key=lambda v: v.success_rate)
    
    def update_stats(
        self, 
        role: str, 
        variant: PromptVariant, 
        success: bool, 
        tokens_used: int = 0,
        failure_reason: Optional[str] = None
    ):
        """
        Update statistics after a task is completed.
        
        Args:
            role: Agent role
            variant: The prompt variant that was used
            success: Whether the task succeeded
            tokens_used: Number of tokens used
            failure_reason: Reason for failure (if failed)
        """
        if success:
            variant.success_count += 1
        else:
            variant.failure_count += 1
            if failure_reason:
                self.failure_patterns[role].append(failure_reason)
                # Keep only recent patterns
                self.failure_patterns[role] = self.failure_patterns[role][-50:]
        
        variant.total_tokens += tokens_used
        self.total_trials[role][variant.domain] += 1
        
        # Check if evolution should be triggered
        total = self.total_trials[role][variant.domain]
        if (total >= self.config.min_trials_before_evolution and 
            total % self.config.evolve_every_n_trials == 0):
            self._maybe_evolve(role, variant.domain)
        
        self._save()
    
    def inject_insights(self, prompt: str, insights: List[str], max_insights: int = 5) -> str:
        """
        Inject relevant insights into a prompt.
        
        Args:
            prompt: Base prompt content
            insights: List of insight strings
            max_insights: Maximum number of insights to inject
        
        Returns:
            Enhanced prompt with insights
        """
        if not insights:
            return prompt
        
        selected = insights[:max_insights]
        insight_text = "\n".join(f"- {ins}" for ins in selected)
        injection = PROMPT_INJECT_INSIGHT_TEMPLATE.format(insights=insight_text)
        
        return prompt + "\n" + injection
    
    def evolve_prompt(
        self, 
        role: str, 
        domain: str, 
        insights: List[str]
    ) -> Optional[PromptVariant]:
        """
        Evolve a new prompt variant using LLM synthesis.
        
        Args:
            role: Agent role
            domain: Task domain
            insights: Relevant insights to incorporate
        
        Returns:
            New PromptVariant if evolution succeeded, None otherwise
        """
        variants = self.variants[role][domain]
        if not variants:
            return None
        
        # Get top performing variants
        sorted_variants = sorted(variants, key=lambda v: v.success_rate, reverse=True)
        top_variants = sorted_variants[:3]
        
        top_prompts_text = "\n\n".join([
            f"### Variant {v.version} (Success Rate: {v.success_rate:.2%})\n{v.content}"
            for v in top_variants
        ])
        
        # Get failure patterns
        patterns = self.failure_patterns.get(role, [])
        failure_text = "\n".join(f"- {p}" for p in patterns[-10:]) if patterns else "None recorded"
        
        # Synthesize new prompt
        try:
            response = self.llm_model(
                messages=[
                    Message("system", PROMPT_SYNTHESIS_SYSTEM),
                    Message("user", PROMPT_SYNTHESIS_USER.format(
                        role=role,
                        top_prompts=top_prompts_text,
                        insights="\n".join(f"- {ins}" for ins in insights[:10]) if insights else "None",
                        failure_patterns=failure_text,
                        domain=domain
                    ))
                ],
                temperature=0.7,  # Some creativity
                max_tokens=1024
            )
            
            if response and len(response) > 50:  # Sanity check
                new_version = max(v.version for v in variants) + 1
                new_variant = PromptVariant(
                    role=role,
                    version=new_version,
                    content=response.strip(),
                    domain=domain,
                    created_from="evolved",
                    parent_version=top_variants[0].version
                )
                
                self.variants[role][domain].append(new_variant)
                self._prune_variants(role, domain)
                self._save()
                
                return new_variant
                
        except Exception as e:
            print(f"Prompt evolution failed for {role}: {e}")
        
        return None
    
    def _maybe_evolve(self, role: str, domain: str):
        """Check conditions and potentially trigger evolution."""
        variants = self.variants[role][domain]
        
        # Only evolve if we have enough data
        if not variants or variants[0].total_uses < self.config.min_trials_before_evolution:
            return
        
        # Collect insights from the best variant's history (placeholder - would come from memory)
        insights = []  # Would be populated from GMemory insights
        
        self.evolve_prompt(role, domain, insights)
    
    def _prune_variants(self, role: str, domain: str):
        """Remove poorly performing variants."""
        variants = self.variants[role][domain]
        
        if len(variants) <= self.config.max_variants_per_role:
            return
        
        # Never prune variants with too few trials
        prunable = [
            v for v in variants 
            if v.total_uses >= self.config.min_trials_before_prune
            and v.success_rate < self.config.prune_below_success_rate
            and v.created_from != "default"  # Never prune defaults
        ]
        
        # Sort by success rate, prune worst
        prunable.sort(key=lambda v: v.success_rate)
        
        to_remove = len(variants) - self.config.max_variants_per_role
        for v in prunable[:to_remove]:
            variants.remove(v)
    
    def get_stats(self, role: str, domain: str = "general") -> Dict[str, Any]:
        """Get statistics for a role-domain pair."""
        variants = self.variants[role][domain]
        
        return {
            "role": role,
            "domain": domain,
            "num_variants": len(variants),
            "total_trials": self.total_trials[role][domain],
            "best_success_rate": max((v.success_rate for v in variants), default=0.0),
            "variants": [
                {
                    "version": v.version,
                    "success_rate": v.success_rate,
                    "total_uses": v.total_uses,
                    "created_from": v.created_from
                }
                for v in sorted(variants, key=lambda x: x.success_rate, reverse=True)
            ]
        }
    
    def _save(self):
        """Persist state to disk."""
        data = {
            "variants": {
                role: {
                    domain: [v.to_dict() for v in variants]
                    for domain, variants in domains.items()
                }
                for role, domains in self.variants.items()
            },
            "total_trials": dict(self.total_trials),
            "failure_patterns": dict(self.failure_patterns)
        }
        
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load state from disk."""
        if not os.path.exists(self.persist_path):
            return
        
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            for role, domains in data.get("variants", {}).items():
                for domain, variants in domains.items():
                    self.variants[role][domain] = [
                        PromptVariant.from_dict(v) for v in variants
                    ]
            
            for role, domains in data.get("total_trials", {}).items():
                for domain, count in domains.items():
                    self.total_trials[role][domain] = count
            
            self.failure_patterns = defaultdict(list, data.get("failure_patterns", {}))
            
        except Exception as e:
            print(f"Failed to load prompt evolution state: {e}")

