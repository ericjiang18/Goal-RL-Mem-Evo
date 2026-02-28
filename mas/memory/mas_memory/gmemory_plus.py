"""
G-Memory++ Main Module
Integrates enhancement modules:
- Goal Module (goal parsing and matching)
- Prompt Evolution (adaptive prompt improvement)
- Skill Miner (experience collection)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .memory_base import MASMemoryBase
from .GMemory import GMemory, TaskLayer, InsightsManager
from ..common import MASMessage, AgentMessage, StateChain

from .goal_module import GoalParser, GoalMatcher, StructuredGoal
# from .prompt_evolution import PromptEvolutionManager, PromptEvolutionConfig, PromptVariant
from .skill_miner import SkillMiner, Skill

from mas.llm import LLMCallable, Message
from mas.utils import EmbeddingFunc


@dataclass
class GMemoryPlusConfig:
    """Configuration for G-Memory++"""
    
    enable_goal_module: bool = True
    # enable_prompt_evolution: bool = False
    enable_skill_miner: bool = True
    
    use_llm_for_goal_parsing: bool = False
    # prompt_evolution_config: Optional[PromptEvolutionConfig] = None
    
    min_cluster_size: int = 3
    skill_similarity_threshold: float = 0.7


@dataclass
class GMemoryPlus(GMemory):
    """
    G-Memory++ extends G-Memory with:
    1. Goal Module - parses tasks into structured goals for better matching
    2. Prompt Evolution - evolves agent prompts based on success/failure (DISABLED)
    3. Skill Miner - extracts reusable skills from successful trajectories
    """
    
    config: GMemoryPlusConfig = field(default_factory=GMemoryPlusConfig)
    
    def __post_init__(self):
        super().__post_init__()
        
        if not hasattr(self, 'config') or self.config is None:
            self.config = GMemoryPlusConfig()
        
        self._init_modules()
        self._task_counter = 0
        self._current_task_id: Optional[str] = None
        self._current_goal: Optional[StructuredGoal] = None
        
        print(f"G-Memory++ initialized with config: {self._get_config_summary()}")
    
    def _init_modules(self):
        if self.config.enable_goal_module:
            self.goal_parser = GoalParser(
                llm_model=self.llm_model if self.config.use_llm_for_goal_parsing else None
            )
            self.goal_matcher = GoalMatcher(self.goal_parser)
        else:
            self.goal_parser = None
            self.goal_matcher = None
        
        # if self.config.enable_prompt_evolution:
        #     self.prompt_evolution = PromptEvolutionManager(
        #         llm_model=self.llm_model,
        #         working_dir=self.persist_dir,
        #         config=self.config.prompt_evolution_config
        #     )
        # else:
        self.prompt_evolution = None
        
        if self.config.enable_skill_miner:
            self.skill_miner = SkillMiner(
                llm_model=self.llm_model,
                embedding_func=self.embedding_func,
                working_dir=self.persist_dir,
                min_cluster_size=self.config.min_cluster_size,
                similarity_threshold=self.config.skill_similarity_threshold
            )
        else:
            self.skill_miner = None
    
    def _get_config_summary(self) -> str:
        enabled = []
        if self.config.enable_goal_module:
            enabled.append("Goal")
        # if self.config.enable_prompt_evolution:
        #     enabled.append("PromptEvo")
        if self.config.enable_skill_miner:
            enabled.append("Skills")
        return f"[{', '.join(enabled)}]"
    
    # ================================ Task Context ================================
    
    def init_task_context(
        self,
        task_main: str,
        task_description: str = None,
        screenshot: Any = None,
        domain: str = None
    ) -> MASMessage:
        mas_message = super().init_task_context(task_main, task_description)
        
        self._current_task_id = f"task_{self._task_counter}"
        self._task_counter += 1
        
        if self.goal_parser:
            self._current_goal = self.goal_parser.parse(
                task_main=task_main,
                task_description=task_description or "",
                use_llm=self.config.use_llm_for_goal_parsing
            )
            mas_message.add_extra_field('structured_goal', self._current_goal.to_dict())
        
        return mas_message
    
    # ================================ Retrieval ================================
    
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
    ) -> Tuple[List[MASMessage], List[MASMessage], List[str], List[Skill]]:
        query_goal = None
        if self.goal_parser:
            query_goal = self.goal_parser.parse(query_task, "")
        
        successful_trajs, failed_trajs, insights = super().retrieve_memory(
            query_task=query_task,
            successful_topk=successful_topk * 2,
            failed_topk=failed_topk * 2,
            insight_topk=insight_topk,
            threshold=threshold
        )
        
        if query_goal and successful_trajs:
            successful_trajs = self._filter_by_goal_similarity(
                query_goal, successful_trajs, successful_topk
            )
        
        skills = []
        if self.skill_miner and query_goal:
            skill_results = self.skill_miner.retrieve_skills(query_goal, top_k=skill_topk)
            skills = [skill for skill, score in skill_results]
        
        return successful_trajs[:successful_topk], failed_trajs[:failed_topk], insights, skills
    
    def _filter_by_goal_similarity(
        self,
        query_goal: StructuredGoal,
        trajectories: List[MASMessage],
        top_k: int
    ) -> List[MASMessage]:
        scored = []
        for traj in trajectories:
            goal_dict = traj.get_extra_field('structured_goal')
            if goal_dict:
                traj_goal = StructuredGoal.from_dict(goal_dict)
                sim = self.goal_parser.compute_similarity(query_goal, traj_goal)
            else:
                sim = 0.5 if query_goal.verb in traj.task_main.lower() else 0.3
            scored.append((traj, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [traj for traj, score in scored[:top_k]]
    
    # ================================ Prompt Management (DISABLED) ================================
    
    # def get_evolved_prompt(
    #     self,
    #     role: str,
    #     default_prompt: str,
    #     domain: str = "general",
    #     insights: List[str] = None
    # ) -> str:
    #     if not self.prompt_evolution:
    #         return default_prompt
    #     
    #     self.prompt_evolution.register_default_prompt(role, default_prompt, domain)
    #     variant = self.prompt_evolution.select_prompt(role, domain)
    #     prompt = variant.content
    #     
    #     if insights:
    #         prompt = self.prompt_evolution.inject_insights(prompt, insights)
    #     
    #     return prompt
    
    # def update_prompt_stats(
    #     self,
    #     role: str,
    #     variant: "PromptVariant",
    #     success: bool,
    #     tokens_used: int = 0,
    #     failure_reason: str = None
    # ):
    #     if self.prompt_evolution:
    #         self.prompt_evolution.update_stats(
    #             role=role,
    #             variant=variant,
    #             success=success,
    #             tokens_used=tokens_used,
    #             failure_reason=failure_reason
    #         )
    
    # ================================ Memory Management ================================
    
    def add_memory(self, mas_message: MASMessage) -> None:
        super().add_memory(mas_message)
        
        if self.skill_miner and mas_message.label == True:
            goal = None
            goal_dict = mas_message.get_extra_field('structured_goal')
            if goal_dict:
                goal = StructuredGoal.from_dict(goal_dict)
            else:
                goal = self.goal_parser.parse(mas_message.task_main, "") if self.goal_parser else None
            
            if goal:
                key_steps = mas_message.get_extra_field('key_steps') or []
                if isinstance(key_steps, str):
                    key_steps = [s.strip() for s in key_steps.split('\n') if s.strip()]
                
                self.skill_miner.add_trajectory(
                    task_id=mas_message.task_main,
                    goal=goal,
                    trajectory=mas_message.task_trajectory,
                    key_steps=key_steps,
                    success=True
                )
    
    # ================================ Skill Access ================================
    
    def get_relevant_skills(self, top_k: int = 3) -> List[Tuple[Skill, float]]:
        if not self.skill_miner or not self._current_goal:
            return []
        return self.skill_miner.retrieve_skills(self._current_goal, top_k=top_k)
    
    def format_skills_for_prompt(self, skills: List[Skill], max_skills: int = 2) -> str:
        if not skills:
            return ""
        
        parts = ["## Relevant Skills from Past Experience:"]
        for skill in skills[:max_skills]:
            parts.append(f"\n### {skill.name}")
            parts.append(f"Description: {skill.description}")
            parts.append("Steps:")
            for i, step in enumerate(skill.steps[:5], 1):
                parts.append(f"  {i}. {step}")
        
        return "\n".join(parts)
    
    # ================================ Statistics ================================
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "memory_size": self.memory_size,
            "task_counter": self._task_counter,
            "modules_enabled": self._get_config_summary(),
        }
        
        if self.skill_miner:
            stats["num_skills"] = len(self.skill_miner.skills)
        
        # if self.prompt_evolution:
        #     stats["prompt_evolution"] = {
        #         role: self.prompt_evolution.get_stats(role)
        #         for role in self.prompt_evolution.variants.keys()
        #     }
        
        return stats
