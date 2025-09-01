"""
Strategic Orchestrator - LLM-driven experiment planning with context awareness
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd

@dataclass
class DatasetProfile:
    """Rich dataset characterization for strategic planning"""
    shape: Tuple[int, int]
    target_type: str
    target_distribution: Dict[str, float]
    feature_types: Dict[str, str]
    missing_patterns: Dict[str, float]
    complexity_score: float
    domain_hints: List[str]

@dataclass
class ExperimentPlan:
    """Structured experiment plan from LLM"""
    experiment_id: str
    preprocessing_strategy: Dict[str, Any]
    model_selection: Dict[str, Any]
    hyperparameter_strategy: Dict[str, Any]
    time_budget: int
    rationale: str
    confidence: float

@dataclass
class ExperimentResult:
    """Comprehensive experiment results"""
    experiment_id: str
    plan: ExperimentPlan
    performance_metrics: Dict[str, float]
    execution_time: float
    success: bool
    insights: List[str]

class StrategicOrchestrator:
    """LLM-driven strategic orchestrator with experience-based learning"""
    
    def __init__(self, llm_client, experience_memory=None):
        self.llm_client = llm_client
        self.memory = experience_memory
        self.experiment_state = {
            'best_score': 0.0,
            'experiment_history': [],
            'time_remaining': 3600,
            'iteration': 0,
            'max_iterations': 20
        }
    
    def generate_experiment_plan(self, dataset_profile: DatasetProfile) -> ExperimentPlan:
        """Generate strategic experiment plan using LLM with context"""
        
        # Get relevant past experiences
        past_experiences = []
        if self.memory:
            past_experiences = self.memory.find_similar_experiences(dataset_profile)
        
        # Build contextual prompt
        prompt = self._build_strategic_prompt(dataset_profile, past_experiences)
        
        # Generate structured response
        response = self.llm_client.generate_structured_response(prompt)
        
        # Validate and enhance plan
        return self._validate_plan(response, dataset_profile)
    
    def _build_strategic_prompt(self, dataset_profile: DatasetProfile, 
                               past_experiences: List[Dict]) -> str:
        """Build rich contextual prompt for strategic planning"""
        
        return f"""
        You are an expert ML strategist. Design the next experiment given:

        DATASET: {dataset_profile.shape[0]} samples, {dataset_profile.shape[1]} features
        TARGET: {dataset_profile.target_type} - {dataset_profile.target_distribution}
        FEATURES: {dataset_profile.feature_types}
        MISSING: {dataset_profile.missing_patterns}
        COMPLEXITY: {dataset_profile.complexity_score:.2f}
        DOMAIN: {dataset_profile.domain_hints}

        EXPERIMENT STATE:
        - Best score: {self.experiment_state['best_score']:.4f}
        - Time remaining: {self.experiment_state['time_remaining']}s
        - Iteration: {self.experiment_state['iteration']}/{self.experiment_state['max_iterations']}

        RECENT EXPERIMENTS: {self._format_recent_experiments()}
        PAST EXPERIENCES: {self._format_past_experiences(past_experiences)}

        STRATEGY: Balance exploration vs exploitation. Consider dataset-specific challenges.
        Focus on highest-impact experiment given time constraints.

        OUTPUT JSON:
        {{
            "experiment_id": "unique_id",
            "preprocessing_strategy": {{
                "missing_handling": "strategy",
                "categorical_encoding": "strategy",
                "feature_engineering": ["steps"],
                "scaling": "strategy"
            }},
            "model_selection": {{
                "primary_model": "model_name",
                "ensemble_candidates": ["models"],
                "rationale": "why these models"
            }},
            "hyperparameter_strategy": {{
                "search_method": "random|bayesian|grid",
                "n_trials": number,
                "parameter_space": {{"param": "distribution"}},
                "cv_strategy": "stratified|kfold"
            }},
            "time_budget": seconds,
            "rationale": "detailed explanation",
            "confidence": 0.0-1.0
        }}
        """
    
    def _format_recent_experiments(self) -> str:
        """Format recent experiments for context"""
        if not self.experiment_state['experiment_history']:
            return "None"
        
        recent = self.experiment_state['experiment_history'][-3:]
        return " | ".join([f"{exp['model']}:{exp['score']:.3f}" for exp in recent])
    
    def _format_past_experiences(self, experiences: List[Dict]) -> str:
        """Format past experiences for context"""
        if not experiences:
            return "None"
        
        return " | ".join([f"{exp['model']}:{exp['score']:.3f}" for exp in experiences[:3]])
    
    def _validate_plan(self, response: Dict[str, Any], 
                      dataset_profile: DatasetProfile) -> ExperimentPlan:
        """Validate and enhance the generated plan"""
        
        # Extract plan data
        plan_data = response if 'experiment_id' in response else response.get('plan', response)
        
        # Validate required fields
        required = ['experiment_id', 'preprocessing_strategy', 'model_selection', 
                   'hyperparameter_strategy', 'time_budget', 'rationale']
        
        for field in required:
            if field not in plan_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Enhance with metadata
        plan_data['experiment_id'] = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.experiment_state['iteration']}"
        plan_data['confidence'] = plan_data.get('confidence', 0.7)
        
        # Validate time budget
        if plan_data['time_budget'] > self.experiment_state['time_remaining']:
            plan_data['time_budget'] = self.experiment_state['time_remaining']
        
        return ExperimentPlan(**plan_data)
    
    def update_experiment_state(self, result: ExperimentResult):
        """Update experiment state based on results"""
        self.experiment_state['experiment_history'].append({
            'model': result.plan.model_selection['primary_model'],
            'score': result.performance_metrics.get('test_accuracy', 0.0),
            'time': result.execution_time,
            'success': result.success
        })
        
        # Update best score
        current_score = result.performance_metrics.get('test_accuracy', 0.0)
        if current_score > self.experiment_state['best_score']:
            self.experiment_state['best_score'] = current_score
        
        # Update time and iteration
        self.experiment_state['time_remaining'] -= result.execution_time
        self.experiment_state['iteration'] += 1
        
        # Store in memory if available
        if self.memory and result.success:
            self.memory.store_experiment(result.plan, result)
    
    def should_continue(self) -> bool:
        """Determine if experiments should continue"""
        return (self.experiment_state['time_remaining'] > 0 and 
                self.experiment_state['iteration'] < self.experiment_state['max_iterations'])
    
    def generate_ensemble_strategy(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate ensemble strategy based on experiment results"""
        successful = [r for r in results if r.success]
        if len(successful) < 2:
            return {'strategy': 'no_ensemble', 'reason': 'insufficient models'}
        
        # Sort by performance
        successful.sort(key=lambda x: x.performance_metrics.get('test_accuracy', 0), reverse=True)
        
        prompt = f"""
        Design ensemble strategy for top models:
        {[{'model': r.plan.model_selection['primary_model'], 'score': r.performance_metrics.get('test_accuracy', 0)} for r in successful[:5]]}
        
        Consider model diversity, performance, and deployment constraints.
        Output JSON with strategy, selected models, and weights.
        """
        
        response = self.llm_client.generate_structured_response(prompt)
        return response.get('ensemble_strategy', {'strategy': 'voting', 'models': successful[:3]})
