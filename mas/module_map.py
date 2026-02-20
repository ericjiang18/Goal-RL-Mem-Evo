from typing import Type

from .reasoning import ReasoningBase, ReasoningIO
from .memory import *
from .memory.mas_memory.gmemory_plus import GMemoryPlus
from .memory.mas_memory.goal_rl_integration import GoalRLMemory

def module_map(
    reasoning: str, mas_memory: str = None
) -> tuple[Type[ReasoningBase], Type[MASMemoryBase]]:
    
    reasoning_map = {
        'io': ReasoningIO,
    }
    mas_memory_map = {
        'empty': MASMemoryBase,
        'voyager': VoyagerMASMemory,
        'memorybank': MemoryBankMASMemory,
        'chatdev': ChatDevMASMemory,
        'generative': GenerativeMASMemory,
        'metagpt': MetaGPTMASMemory,
        'g-memory': GMemory,
        # G-Memory++ with all enhancements
        'g-memory-plus': GMemoryPlus,
        'gmemory-plus': GMemoryPlus,
        'gm+': GMemoryPlus,
        # Goal RL enhanced memory
        'goal-rl': GoalRLMemory,
        'goal-rl-memory': GoalRLMemory,
        'grl': GoalRLMemory,
    }

    if reasoning not in reasoning_map:
        raise ValueError(f"Invalid reasoning type '{reasoning}'. Allowed values: {list(reasoning_map.keys())}")

    if mas_memory is not None and mas_memory not in mas_memory_map:
        raise ValueError(f"Invalid MAS memory type '{mas_memory}'. Allowed values: {list(mas_memory_map.keys())}")

    return (
        reasoning_map[reasoning],
        mas_memory_map.get(mas_memory, None)
    )
    
    