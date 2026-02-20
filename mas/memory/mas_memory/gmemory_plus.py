"""
G-Memory++ Main Module
Integrates all enhancement modules:
- Goal Module (goal parsing and matching)
- Prompt Evolution (adaptive prompt improvement)
- GCN Retriever (graph-based retrieval)
- Visual Encoder (Qwen-VL for GUI tasks)
- Skill Miner (experience collection)
- Goal Reinforcement (progress tracking)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

from langchain_chroma import Chroma

from .memory_base import MASMemoryBase
from .GMemory import GMemory, TaskLayer, InsightsManager
from ..common import MASMessage, AgentMessage, StateChain

from .goal_module import GoalParser, GoalMatcher, StructuredGoal
from .prompt_evolution import PromptEvolutionManager, PromptEvolutionConfig, PromptVariant
from .gcn_retriever import GCNRetriever, NodeType
from .visual_encoder import VisualEncoderBase, create_visual_encoder, VisualContext
from .skill_miner import SkillMiner, Skill
from .goal_reinforcement import GoalReinforcementChecker, GoalCheckResult, GoalTracker

from mas.llm import LLMCallable, Message
from mas.utils import EmbeddingFunc


# ================================ Configuration ================================

@dataclass
class GMemoryPlusConfig:
    """Configuration for G-Memory++"""
    
    # Module enablement
    enable_goal_module: bool = True
    enable_prompt_evolution: bool = False
    enable_gcn_retriever: bool = False
    enable_visual_encoder: bool = False
    enable_skill_miner: bool = True
    enable_goal_reinforcement: bool = True
    
    # Goal module settings
    use_llm_for_goal_parsing: bool = False  # Faster with rules
    
    # Prompt evolution settings
    prompt_evolution_config: Optional[PromptEvolutionConfig] = None
    
    # GCN settings
    gcn_hidden_dim: int = 128
    gcn_output_dim: int = 64
    gcn_model_type: str = "sage"
    train_gcn_every_n_tasks: int = 20
    
    # Visual encoder settings
    visual_encoder_type: str = "qwen-vl"  # "qwen-vl", "fallback", or "auto"
    visual_model_name: str = "qwen3-vl-plus"  # Qwen VL model to use
    use_visual_idealab: bool = True  # Use Idealab API (True) or Aliyun DashScope (False)
    
    # Skill miner settings
    min_cluster_size: int = 3
    skill_similarity_threshold: float = 0.7
    
    # Goal reinforcement settings
    goal_check_interval: int = 3
    max_stuck_steps: int = 5
    
    # Retrieval settings
    use_gcn_for_retrieval: bool = True
    combine_retrieval_scores: bool = True  # Combine GCN + embedding scores


# ================================ G-Memory++ ================================

@dataclass
class GMemoryPlus(GMemory):
    """
    G-Memory++ extends G-Memory with:
    
    1. **Goal Module**: Parses tasks into structured goals for better matching
    2. **Prompt Evolution**: Evolves agent prompts based on success/failure
    3. **GCN Retriever**: Uses graph neural networks for learned retrieval
    4. **Visual Encoder**: Handles GUI/visual tasks with Qwen-VL
    5. **Skill Miner**: Extracts reusable skills from successful trajectories
    6. **Goal Reinforcement**: Keeps agents focused on goals during execution
    """
    
    # Configuration
    config: GMemoryPlusConfig = field(default_factory=GMemoryPlusConfig)
    
    def __post_init__(self):
        # Initialize base G-Memory
        super().__post_init__()
        
        # Initialize config if not provided
        if not hasattr(self, 'config') or self.config is None:
            self.config = GMemoryPlusConfig()
        
        # Initialize enhanced modules
        self._init_modules()
        
        # Task counter for periodic operations
        self._task_counter = 0
        
        # Current task tracking
        self._current_task_id: Optional[str] = None
        self._current_goal: Optional[StructuredGoal] = None
        
        print(f"G-Memory++ initialized with config: {self._get_config_summary()}")
    
    def _init_modules(self):
        """Initialize all enhancement modules."""
        
        # Goal Module
        if self.config.enable_goal_module:
            self.goal_parser = GoalParser(
                llm_model=self.llm_model if self.config.use_llm_for_goal_parsing else None
            )
            self.goal_matcher = GoalMatcher(self.goal_parser)
        else:
            self.goal_parser = None
            self.goal_matcher = None
        
        # Prompt Evolution
        if self.config.enable_prompt_evolution:
            self.prompt_evolution = PromptEvolutionManager(
                llm_model=self.llm_model,
                working_dir=self.persist_dir,
                config=self.config.prompt_evolution_config
            )
        else:
            self.prompt_evolution = None
        
        # GCN Retriever
        if self.config.enable_gcn_retriever:
            try:
                self.gcn_retriever = GCNRetriever(
                    working_dir=self.persist_dir,
                    hidden_dim=self.config.gcn_hidden_dim,
                    output_dim=self.config.gcn_output_dim,
                    model_type=self.config.gcn_model_type
                )
            except Exception as e:
                print(f"GCN Retriever initialization failed: {e}")
                self.gcn_retriever = None
        else:
            self.gcn_retriever = None
        
        # Visual Encoder (using Qwen VL API like AgentNet)
        if self.config.enable_visual_encoder:
            try:
                self.visual_encoder = create_visual_encoder(
                    encoder_type=self.config.visual_encoder_type,
                    model_name=self.config.visual_model_name,
                    use_idealab=self.config.use_visual_idealab,
                )
            except Exception as e:
                print(f"Visual Encoder initialization failed: {e}")
                self.visual_encoder = None
        else:
            self.visual_encoder = None
        
        # Skill Miner
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
        
        # Goal Reinforcement
        if self.config.enable_goal_reinforcement:
            self.goal_reinforcer = GoalReinforcementChecker(
                llm_model=self.llm_model,
                check_interval=self.config.goal_check_interval,
                max_stuck_steps=self.config.max_stuck_steps
            )
        else:
            self.goal_reinforcer = None
    
    def _get_config_summary(self) -> str:
        """Get summary of enabled modules."""
        enabled = []
        if self.config.enable_goal_module:
            enabled.append("Goal")
        if self.config.enable_prompt_evolution:
            enabled.append("PromptEvo")
        if self.config.enable_gcn_retriever:
            enabled.append("GCN")
        if self.config.enable_visual_encoder:
            enabled.append("Visual")
        if self.config.enable_skill_miner:
            enabled.append("Skills")
        if self.config.enable_goal_reinforcement:
            enabled.append("GoalReinf")
        return f"[{', '.join(enabled)}]"
    
    # ================================ Enhanced Initialization ================================
    
    def init_task_context(
        self,
        task_main: str,
        task_description: str = None,
        screenshot: Any = None,  # PIL Image for GUI tasks
        domain: str = None
    ) -> MASMessage:
        """
        Initialize task context with goal parsing and tracking.
        
        Args:
            task_main: Main task string
            task_description: Additional description
            screenshot: Optional screenshot for GUI tasks
            domain: Task domain (auto-detected if not provided)
        """
        # Call parent method
        mas_message = super().init_task_context(task_main, task_description)
        
        # Generate task ID
        self._current_task_id = f"task_{self._task_counter}"
        self._task_counter += 1
        
        # Parse goal
        if self.goal_parser:
            self._current_goal = self.goal_parser.parse(
                task_main=task_main,
                task_description=task_description or "",
                use_llm=self.config.use_llm_for_goal_parsing
            )
            
            # Store goal in message
            mas_message.add_extra_field('structured_goal', self._current_goal.to_dict())
        
        # Handle visual context
        if screenshot and self.visual_encoder:
            visual_context = VisualContext(
                description=self.visual_encoder.describe_ui(screenshot),
                embedding=self.visual_encoder.encode_image(screenshot)
            )
            mas_message.add_extra_field('visual_context', visual_context.to_dict())
        
        # Start goal tracking
        if self.goal_reinforcer and self._current_goal:
            initial_state = task_description or task_main
            self.goal_reinforcer.start_tracking(
                task_id=self._current_task_id,
                goal=self._current_goal,
                initial_state=initial_state
            )
        
        return mas_message
    
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
    ) -> Tuple[List[MASMessage], List[MASMessage], List[str], List[Skill]]:
        """
        Enhanced memory retrieval with goal matching, GCN, and skill retrieval.
        
        Returns:
            Tuple of (successful_trajectories, failed_trajectories, insights, skills)
        """
        # Parse query goal
        query_goal = None
        if self.goal_parser:
            query_goal = self.goal_parser.parse(query_task, "")
        
        # Visual embedding for GUI tasks
        visual_embedding = None
        if screenshot and self.visual_encoder:
            visual_embedding = self.visual_encoder.encode_image(screenshot)
        
        # Get base retrieval from parent
        successful_trajs, failed_trajs, insights = super().retrieve_memory(
            query_task=query_task,
            successful_topk=successful_topk * 2,  # Get more, then filter
            failed_topk=failed_topk * 2,
            insight_topk=insight_topk,
            threshold=threshold
        )
        
        # Enhanced retrieval with GCN
        if self.gcn_retriever and self.config.use_gcn_for_retrieval:
            query_embedding = self.embedding_func.embed_query(query_task)
            goal_features = query_goal.to_features() if query_goal else None
            
            gcn_results = self.gcn_retriever.retrieve(
                query_text_embed=query_embedding,
                query_goal_features=goal_features,
                top_k=successful_topk + failed_topk,
                node_type_filter=NodeType.QUERY
            )
            
            if gcn_results and self.config.combine_retrieval_scores:
                # Re-rank trajectories based on GCN scores
                successful_trajs = self._rerank_with_gcn(
                    successful_trajs, gcn_results, successful_topk
                )
        
        # Goal-based filtering
        if query_goal and successful_trajs:
            successful_trajs = self._filter_by_goal_similarity(
                query_goal, successful_trajs, successful_topk
            )
        
        # Retrieve relevant skills
        skills = []
        if self.skill_miner and query_goal:
            skill_results = self.skill_miner.retrieve_skills(query_goal, top_k=skill_topk)
            skills = [skill for skill, score in skill_results]
        
        return successful_trajs[:successful_topk], failed_trajs[:failed_topk], insights, skills
    
    def _rerank_with_gcn(
        self,
        trajectories: List[MASMessage],
        gcn_results: List[Tuple[str, float]],
        top_k: int
    ) -> List[MASMessage]:
        """Re-rank trajectories using GCN similarity scores."""
        gcn_scores = {node_id: score for node_id, score in gcn_results}
        
        scored = []
        for traj in trajectories:
            task_main = traj.task_main
            gcn_score = gcn_scores.get(task_main, 0.0)
            
            # Combine with original ranking (assume index = rank)
            original_rank_score = 1.0 / (scored.__len__() + 1)
            combined_score = 0.6 * gcn_score + 0.4 * original_rank_score
            
            scored.append((traj, combined_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [traj for traj, score in scored[:top_k]]
    
    def _filter_by_goal_similarity(
        self,
        query_goal: StructuredGoal,
        trajectories: List[MASMessage],
        top_k: int
    ) -> List[MASMessage]:
        """Filter trajectories by goal similarity."""
        scored = []
        
        for traj in trajectories:
            goal_dict = traj.get_extra_field('structured_goal')
            if goal_dict:
                traj_goal = StructuredGoal.from_dict(goal_dict)
                sim = self.goal_parser.compute_similarity(query_goal, traj_goal)
            else:
                # Fallback: use verb matching from task text
                sim = 0.5 if query_goal.verb in traj.task_main.lower() else 0.3
            
            scored.append((traj, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [traj for traj, score in scored[:top_k]]
    
    # ================================ Prompt Management ================================
    
    def get_evolved_prompt(
        self,
        role: str,
        default_prompt: str,
        domain: str = "general",
        insights: List[str] = None
    ) -> str:
        """
        Get an evolved prompt for an agent role.
        
        Args:
            role: Agent role name
            default_prompt: Default prompt to use if no evolved version
            domain: Task domain
            insights: Insights to inject into prompt
        
        Returns:
            Evolved prompt string
        """
        if not self.prompt_evolution:
            return default_prompt
        
        # Register default if not exists
        self.prompt_evolution.register_default_prompt(role, default_prompt, domain)
        
        # Select best variant
        variant = self.prompt_evolution.select_prompt(role, domain)
        prompt = variant.content
        
        # Inject insights
        if insights:
            prompt = self.prompt_evolution.inject_insights(prompt, insights)
        
        return prompt
    
    def update_prompt_stats(
        self,
        role: str,
        variant: PromptVariant,
        success: bool,
        tokens_used: int = 0,
        failure_reason: str = None
    ):
        """Update prompt statistics after task completion."""
        if self.prompt_evolution:
            self.prompt_evolution.update_stats(
                role=role,
                variant=variant,
                success=success,
                tokens_used=tokens_used,
                failure_reason=failure_reason
            )
    
    # ================================ Goal Reinforcement ================================
    
    def check_goal_progress(
        self,
        trajectory: str,
        current_state: str,
        step_number: int,
        force_check: bool = False
    ) -> Optional[GoalCheckResult]:
        """
        Check if the agent is making progress toward the goal.
        
        Args:
            trajectory: Current trajectory text
            current_state: Current environment state
            step_number: Current step number
            force_check: Force check even if not at interval
        
        Returns:
            GoalCheckResult if check was performed
        """
        if not self.goal_reinforcer or not self._current_task_id:
            return None
        
        return self.goal_reinforcer.check_progress(
            task_id=self._current_task_id,
            trajectory=trajectory,
            current_state=current_state,
            step_number=step_number,
            force_check=force_check
        )
    
    def get_corrective_guidance(
        self,
        check_result: GoalCheckResult,
        insights: List[str] = None
    ) -> str:
        """Get corrective guidance based on goal check result."""
        if not self.goal_reinforcer or not self._current_task_id:
            return ""
        
        return self.goal_reinforcer.get_corrective_guidance(
            task_id=self._current_task_id,
            check_result=check_result,
            insights=insights
        )
    
    def record_action(self, action: str, state: str):
        """Record an action for loop detection."""
        if self.goal_reinforcer:
            self.goal_reinforcer.record_action(action, state)
    
    # ================================ Memory Management ================================
    
    def add_memory(self, mas_message: MASMessage) -> None:
        """
        Add memory with enhanced processing.
        """
        # Call parent method
        super().add_memory(mas_message)
        
        # Add to GCN graph
        if self.gcn_retriever:
            embedding = self.embedding_func.embed_query(mas_message.task_main)
            goal_features = None
            
            goal_dict = mas_message.get_extra_field('structured_goal')
            if goal_dict:
                goal = StructuredGoal.from_dict(goal_dict)
                goal_features = goal.to_features()
            
            self.gcn_retriever.add_node(
                node_id=mas_message.task_main,
                node_type=NodeType.QUERY,
                text_embedding=embedding,
                goal_features=goal_features,
                metadata={
                    "label": mas_message.label,
                    "task_id": self._current_task_id
                }
            )
        
        # Add to skill miner (successful trajectories only)
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
        
        # Periodically train GCN
        if (self.gcn_retriever and 
            self.memory_size > 0 and 
            self.memory_size % self.config.train_gcn_every_n_tasks == 0):
            self._train_gcn()
        
        # End goal tracking
        if self.goal_reinforcer and self._current_task_id:
            self.goal_reinforcer.end_tracking(self._current_task_id)
    
    def _train_gcn(self):
        """Train the GCN model on accumulated data."""
        if not self.gcn_retriever:
            return
        
        print(f"Training GCN model on {self.memory_size} tasks...")
        try:
            self.gcn_retriever.train(epochs=10)
            print("GCN training complete")
        except Exception as e:
            print(f"GCN training failed: {e}")
    
    # ================================ Visual Support ================================
    
    def process_visual_input(
        self,
        image: Any,
        instruction: str = None
    ) -> Dict[str, Any]:
        """
        Process visual input (screenshot) for GUI tasks.
        
        Args:
            image: PIL Image or path to image
            instruction: Optional instruction for action detection
        
        Returns:
            Dictionary with visual analysis results
        """
        if not self.visual_encoder:
            return {"error": "Visual encoder not available"}
        
        result = {
            "description": self.visual_encoder.describe_ui(image),
            "embedding": self.visual_encoder.encode_image(image).tolist()
        }
        
        if instruction:
            action_target = self.visual_encoder.get_action_target(image, instruction)
            result["action_target"] = action_target
        
        return result
    
    # ================================ Skill Access ================================
    
    def get_relevant_skills(
        self,
        top_k: int = 3
    ) -> List[Tuple[Skill, float]]:
        """Get skills relevant to the current task."""
        if not self.skill_miner or not self._current_goal:
            return []
        
        return self.skill_miner.retrieve_skills(self._current_goal, top_k=top_k)
    
    def format_skills_for_prompt(
        self,
        skills: List[Skill],
        max_skills: int = 2
    ) -> str:
        """Format skills for injection into agent prompts."""
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
        """Get comprehensive statistics."""
        stats = {
            "memory_size": self.memory_size,
            "task_counter": self._task_counter,
            "modules_enabled": self._get_config_summary(),
        }
        
        if self.skill_miner:
            stats["num_skills"] = len(self.skill_miner.skills)
        
        if self.prompt_evolution:
            stats["prompt_evolution"] = {
                role: self.prompt_evolution.get_stats(role)
                for role in self.prompt_evolution.variants.keys()
            }
        
        if self.gcn_retriever:
            stats["gcn_nodes"] = len(self.gcn_retriever.graph_data.node_ids)
            stats["gcn_edges"] = len(self.gcn_retriever.graph_data.edges)
        
        return stats

