"""
Autonomous ML Agent - Strategic ML automation with LLM-driven planning
"""

__version__ = "1.0.0"
__author__ = "Hasibur Rashid"

from .core import AutonomousMLAgent
from .strategic_orchestrator import StrategicOrchestrator
from .experience_memory import ExperienceMemory
from .plan_executor import PlanExecutor

__all__ = [
    "AutonomousMLAgent",
    "StrategicOrchestrator", 
    "ExperienceMemory",
    "PlanExecutor"
]
