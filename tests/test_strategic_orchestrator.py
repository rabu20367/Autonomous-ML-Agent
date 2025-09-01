"""
Tests for the strategic orchestrator module
"""

import pytest
import json
from unittest.mock import Mock, patch
from autonomous_ml.strategic_orchestrator import (
    StrategicOrchestrator, DatasetProfile, ExperimentPlan, ExperimentResult
)

class TestStrategicOrchestrator:
    """Test cases for the StrategicOrchestrator class"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client"""
        client = Mock()
        client.generate_structured_response.return_value = {
            'experiment_id': 'test_exp_001',
            'preprocessing_strategy': {
                'missing_handling': 'median',
                'categorical_encoding': 'onehot',
                'scaling': 'standard'
            },
            'model_selection': {
                'primary_model': 'logistic_regression',
                'ensemble_candidates': ['random_forest', 'gradient_boosting'],
                'rationale': 'Good for this dataset type'
            },
            'hyperparameter_strategy': {
                'search_method': 'random',
                'n_trials': 20,
                'parameter_space': {'C': [0.1, 1.0, 10.0]},
                'cv_strategy': 'stratified'
            },
            'time_budget': 300,
            'rationale': 'Strategic experiment for classification',
            'confidence': 0.8
        }
        return client
    
    @pytest.fixture
    def mock_experience_memory(self):
        """Create mock experience memory"""
        memory = Mock()
        memory.find_similar_experiences.return_value = [
            {
                'model': 'logistic_regression',
                'score': 0.85,
                'experiment_plan': {'model_selection': {'primary_model': 'logistic_regression'}},
                'results': {'performance_metrics': {'test_accuracy': 0.85}}
            }
        ]
        return memory
    
    @pytest.fixture
    def sample_dataset_profile(self):
        """Create sample dataset profile"""
        return DatasetProfile(
            shape=(1000, 10),
            target_type='classification',
            target_distribution={'0': 0.6, '1': 0.4},
            feature_types={'feature_1': 'numerical', 'feature_2': 'categorical'},
            missing_patterns={'feature_1': 0.05, 'feature_2': 0.0},
            complexity_score=0.3,
            domain_hints=['financial_data', 'demographic_data']
        )
    
    @pytest.fixture
    def orchestrator(self, mock_llm_client, mock_experience_memory):
        """Create test orchestrator"""
        return StrategicOrchestrator(mock_llm_client, mock_experience_memory)
    
    def test_orchestrator_initialization(self, mock_llm_client):
        """Test orchestrator initialization"""
        orchestrator = StrategicOrchestrator(mock_llm_client)
        
        assert orchestrator.llm_client == mock_llm_client
        assert orchestrator.memory is None
        assert orchestrator.experiment_state['best_score'] == 0.0
        assert orchestrator.experiment_state['time_remaining'] == 3600
        assert orchestrator.experiment_state['iteration'] == 0
    
    def test_generate_experiment_plan(self, orchestrator, sample_dataset_profile):
        """Test experiment plan generation"""
        plan = orchestrator.generate_experiment_plan(sample_dataset_profile)
        
        assert isinstance(plan, ExperimentPlan)
        assert plan.experiment_id.startswith('exp_')
        assert plan.model_selection['primary_model'] == 'logistic_regression'
        assert plan.time_budget == 300
        assert plan.confidence == 0.8
        assert 'preprocessing_strategy' in plan.__dict__
        assert 'hyperparameter_strategy' in plan.__dict__
    
    def test_build_strategic_prompt(self, orchestrator, sample_dataset_profile):
        """Test strategic prompt building"""
        prompt = orchestrator._build_strategic_prompt(sample_dataset_profile, [])
        
        assert isinstance(prompt, str)
        assert '1000 samples' in prompt
        assert 'classification' in prompt
        assert 'financial_data' in prompt
        assert 'JSON' in prompt
        assert 'experiment_id' in prompt
    
    def test_format_recent_experiments(self, orchestrator):
        """Test recent experiments formatting"""
        # Test with no experiments
        formatted = orchestrator._format_recent_experiments()
        assert formatted == "None"
        
        # Test with experiments
        orchestrator.experiment_state['experiment_history'] = [
            {'model': 'logistic_regression', 'score': 0.85, 'time': 10.0, 'success': True},
            {'model': 'random_forest', 'score': 0.82, 'time': 15.0, 'success': True}
        ]
        
        formatted = orchestrator._format_recent_experiments()
        assert 'logistic_regression:0.850' in formatted
        assert 'random_forest:0.820' in formatted
    
    def test_format_past_experiences(self, orchestrator):
        """Test past experiences formatting"""
        # Test with no experiences
        formatted = orchestrator._format_past_experiences([])
        assert formatted == "None"
        
        # Test with experiences
        experiences = [
            {'model': 'logistic_regression', 'score': 0.85},
            {'model': 'random_forest', 'score': 0.82}
        ]
        
        formatted = orchestrator._format_past_experiences(experiences)
        assert 'logistic_regression:0.850' in formatted
        assert 'random_forest:0.820' in formatted
    
    def test_validate_plan(self, orchestrator, sample_dataset_profile):
        """Test plan validation"""
        response = {
            'experiment_id': 'test_exp',
            'preprocessing_strategy': {'missing_handling': 'median'},
            'model_selection': {'primary_model': 'logistic_regression'},
            'hyperparameter_strategy': {'search_method': 'random'},
            'time_budget': 300,
            'rationale': 'Test experiment'
        }
        
        plan = orchestrator._validate_plan(response, sample_dataset_profile)
        
        assert isinstance(plan, ExperimentPlan)
        assert plan.experiment_id.startswith('exp_')
        assert plan.confidence == 0.7  # Default value
        assert plan.time_budget == 300
    
    def test_validate_plan_missing_fields(self, orchestrator, sample_dataset_profile):
        """Test plan validation with missing fields"""
        response = {
            'experiment_id': 'test_exp',
            'preprocessing_strategy': {'missing_handling': 'median'}
            # Missing required fields
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            orchestrator._validate_plan(response, sample_dataset_profile)
    
    def test_validate_plan_time_budget_adjustment(self, orchestrator, sample_dataset_profile):
        """Test time budget adjustment"""
        orchestrator.experiment_state['time_remaining'] = 100  # Less than requested budget
        
        response = {
            'experiment_id': 'test_exp',
            'preprocessing_strategy': {'missing_handling': 'median'},
            'model_selection': {'primary_model': 'logistic_regression'},
            'hyperparameter_strategy': {'search_method': 'random'},
            'time_budget': 300,  # More than remaining time
            'rationale': 'Test experiment'
        }
        
        plan = orchestrator._validate_plan(response, sample_dataset_profile)
        assert plan.time_budget == 100  # Should be adjusted to remaining time
    
    def test_update_experiment_state(self, orchestrator):
        """Test experiment state update"""
        # Create mock experiment result
        plan = Mock()
        plan.model_selection = {'primary_model': 'logistic_regression'}
        
        result = Mock()
        result.plan = plan
        result.performance_metrics = {'test_accuracy': 0.85}
        result.execution_time = 10.0
        result.success = True
        
        # Update state
        orchestrator.update_experiment_state(result)
        
        # Check state updates
        assert orchestrator.experiment_state['best_score'] == 0.85
        assert orchestrator.experiment_state['time_remaining'] == 3590  # 3600 - 10
        assert orchestrator.experiment_state['iteration'] == 1
        assert len(orchestrator.experiment_state['experiment_history']) == 1
        
        # Check history entry
        history_entry = orchestrator.experiment_state['experiment_history'][0]
        assert history_entry['model'] == 'logistic_regression'
        assert history_entry['score'] == 0.85
        assert history_entry['success'] == True
    
    def test_should_continue(self, orchestrator):
        """Test continuation logic"""
        # Should continue initially
        assert orchestrator.should_continue() == True
        
        # Should not continue when time is up
        orchestrator.experiment_state['time_remaining'] = 0
        assert orchestrator.should_continue() == False
        
        # Reset time and test iteration limit
        orchestrator.experiment_state['time_remaining'] = 1000
        orchestrator.experiment_state['iteration'] = 20
        assert orchestrator.should_continue() == False
    
    def test_generate_ensemble_strategy(self, orchestrator):
        """Test ensemble strategy generation"""
        # Create mock experiment results
        plan1 = Mock()
        plan1.model_selection = {'primary_model': 'logistic_regression'}
        
        plan2 = Mock()
        plan2.model_selection = {'primary_model': 'random_forest'}
        
        result1 = Mock()
        result1.success = True
        result1.performance_metrics = {'test_accuracy': 0.85}
        result1.plan = plan1
        
        result2 = Mock()
        result2.success = True
        result2.performance_metrics = {'test_accuracy': 0.82}
        result2.plan = plan2
        
        result3 = Mock()
        result3.success = False  # Failed experiment
        
        results = [result1, result2, result3]
        
        # Mock LLM response for ensemble strategy
        orchestrator.llm_client.generate_structured_response.return_value = {
            'ensemble_strategy': {
                'strategy': 'voting',
                'models': ['logistic_regression', 'random_forest'],
                'weights': {'logistic_regression': 0.6, 'random_forest': 0.4}
            }
        }
        
        ensemble_strategy = orchestrator.generate_ensemble_strategy(results)
        
        assert ensemble_strategy['strategy'] == 'voting'
        assert 'logistic_regression' in ensemble_strategy['models']
        assert 'random_forest' in ensemble_strategy['models']
    
    def test_generate_ensemble_strategy_insufficient_models(self, orchestrator):
        """Test ensemble strategy with insufficient successful models"""
        # Create mock experiment results with only one successful model
        plan = Mock()
        plan.model_selection = {'primary_model': 'logistic_regression'}
        
        result = Mock()
        result.success = True
        result.performance_metrics = {'test_accuracy': 0.85}
        result.plan = plan
        
        results = [result]
        
        ensemble_strategy = orchestrator.generate_ensemble_strategy(results)
        
        assert ensemble_strategy['strategy'] == 'no_ensemble'
        assert ensemble_strategy['reason'] == 'insufficient models'
