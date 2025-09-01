"""
Tests for the core Autonomous ML Agent
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from autonomous_ml.core import AutonomousMLAgent
from autonomous_ml.config import Config

class TestAutonomousMLAgent:
    """Test cases for the main AutonomousMLAgent class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = Config()
        config.LLM.api_key = "test_key"
        return config
    
    @pytest.fixture
    def agent(self, config):
        """Create test agent with mocked LLM client"""
        with patch('autonomous_ml.core.StructuredLLMClient') as mock_llm:
            mock_llm.return_value.generate_structured_response.return_value = {
                'experiment_id': 'test_exp',
                'preprocessing_strategy': {'missing_handling': 'median', 'scaling': 'standard'},
                'model_selection': {'primary_model': 'logistic_regression'},
                'hyperparameter_strategy': {'search_method': 'random', 'n_trials': 5},
                'time_budget': 60,
                'rationale': 'Test experiment',
                'confidence': 0.8
            }
            
            agent = AutonomousMLAgent(config)
            return agent
    
    def test_agent_initialization(self, config):
        """Test agent initialization"""
        agent = AutonomousMLAgent(config)
        assert agent.config == config
        assert agent.strategy == 'adaptive'
        assert agent.results == {}
        assert agent.experiment_history == []
    
    def test_load_data_csv(self, agent, tmp_path):
        """Test loading CSV data"""
        # Create test CSV file
        test_data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })
        
        csv_path = tmp_path / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Test loading
        loaded_data = agent._load_data(str(csv_path))
        pd.testing.assert_frame_equal(loaded_data, test_data)
    
    def test_create_dataset_profile(self, agent):
        """Test dataset profile creation"""
        analysis = {
            'shape': (100, 5),
            'target_type': 'classification',
            'target_distribution': {'0': 0.6, '1': 0.4},
            'feature_types': {'feature_1': 'numerical', 'feature_2': 'categorical'},
            'missing_patterns': {'feature_1': 0.0, 'feature_2': 0.1},
            'complexity_score': 0.5,
            'domain_hints': ['test_data']
        }
        
        profile = agent._create_dataset_profile(analysis)
        
        assert profile.shape == (100, 5)
        assert profile.target_type == 'classification'
        assert profile.complexity_score == 0.5
        assert 'test_data' in profile.domain_hints
    
    @patch('autonomous_ml.core.DataPipeline')
    def test_run_pipeline_basic(self, mock_pipeline, agent, sample_data, tmp_path):
        """Test basic pipeline execution"""
        # Mock data pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.analyze_data.return_value = {
            'shape': (100, 6),
            'target_type': 'classification',
            'target_distribution': {'0': 0.6, '1': 0.4},
            'feature_types': {'feature_0': 'numerical'},
            'missing_patterns': {'feature_0': 0.0},
            'complexity_score': 0.5,
            'domain_hints': ['test_data']
        }
        mock_pipeline_instance.preprocess_data.return_value = (
            pd.DataFrame(np.random.randn(80, 5)),  # X_train
            pd.DataFrame(np.random.randn(20, 5)),  # X_test
            pd.Series(np.random.randint(0, 2, 80)),  # y_train
            pd.Series(np.random.randint(0, 2, 20))   # y_test
        )
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create test CSV file
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Mock plan executor
        with patch('autonomous_ml.core.PlanExecutor') as mock_executor:
            mock_executor_instance = Mock()
            mock_executor_instance.execute_plan.return_value = {
                'success': True,
                'test_metrics': {'test_accuracy': 0.85, 'test_f1': 0.82},
                'execution_time': 10.0,
                'insights': ['Model performed well']
            }
            mock_executor.return_value = mock_executor_instance
            
            # Run pipeline
            results = agent.run_pipeline(str(csv_path), 'target', max_experiments=1)
            
            # Assertions
            assert 'pipeline_metadata' in results
            assert 'leaderboard' in results
            assert 'best_model' in results
            assert results['pipeline_metadata']['total_experiments'] == 1
    
    def test_compile_results(self, agent):
        """Test results compilation"""
        # Mock dataset profile
        profile = Mock()
        profile.__dict__ = {
            'shape': (100, 5),
            'target_type': 'classification',
            'complexity_score': 0.5
        }
        
        # Mock experiment results
        result1 = Mock()
        result1.success = True
        result1.performance_metrics = {'test_accuracy': 0.85, 'test_f1': 0.82}
        result1.execution_time = 10.0
        result1.insights = ['Good performance']
        result1.plan = Mock()
        result1.plan.model_selection = {'primary_model': 'logistic_regression'}
        result1.__dict__ = {
            'experiment_id': 'exp1',
            'plan': result1.plan,
            'performance_metrics': result1.performance_metrics,
            'execution_time': result1.execution_time,
            'success': result1.success,
            'insights': result1.insights
        }
        
        experiment_results = [result1]
        ensemble_strategy = {'strategy': 'voting', 'models': ['logistic_regression']}
        data_pipeline = Mock()
        
        results = agent._compile_results(profile, experiment_results, ensemble_strategy, data_pipeline)
        
        assert 'pipeline_metadata' in results
        assert 'leaderboard' in results
        assert 'best_model' in results
        assert len(results['leaderboard']) == 1
        assert results['leaderboard'][0]['test_accuracy'] == 0.85
    
    def test_generate_insights(self, agent):
        """Test insights generation"""
        # Mock results
        agent.results = {
            'leaderboard': [
                {
                    'model': 'logistic_regression',
                    'test_accuracy': 0.85,
                    'insights': ['Good performance']
                },
                {
                    'model': 'random_forest',
                    'test_accuracy': 0.82,
                    'insights': ['Decent performance']
                }
            ]
        }
        
        insights = agent._generate_insights()
        
        assert 'performance_summary' in insights
        assert 'recommendations' in insights
        assert 'technical_insights' in insights
        assert insights['performance_summary']['best_accuracy'] == 0.85
        assert len(insights['recommendations']) > 0
