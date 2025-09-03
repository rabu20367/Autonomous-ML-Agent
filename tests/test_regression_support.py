"""
Test regression support across the system
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from autonomous_ml.data_pipeline import DataPipeline
from autonomous_ml.plan_executor import PlanExecutor
from autonomous_ml.config import Config


class TestRegressionSupport:
    """Test regression functionality"""
    
    def setup_method(self):
        """Set up test data"""
        # Create regression dataset
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        self.regression_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        self.regression_df['target'] = y
        
        # Create classification dataset for comparison
        from sklearn.datasets import make_classification
        X_clf, y_clf = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        self.classification_df = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(5)])
        self.classification_df['target'] = y_clf
        
        self.config = Config()
    
    def test_regression_target_detection(self):
        """Test that regression targets are correctly identified"""
        pipeline = DataPipeline('target', 0.2)
        analysis = pipeline.analyze_data(self.regression_df)
        
        assert analysis['target_type'] == 'regression'
        assert 'complexity_score' in analysis
        assert 'feature_types' in analysis
    
    def test_classification_target_detection(self):
        """Test that classification targets are correctly identified"""
        pipeline = DataPipeline('target', 0.2)
        analysis = pipeline.analyze_data(self.classification_df)
        
        assert analysis['target_type'] == 'classification'
    
    def test_regression_preprocessing(self):
        """Test regression data preprocessing"""
        pipeline = DataPipeline('target', 0.2)
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(self.regression_df)
        
        # Check that target is not encoded for regression
        assert y_train.dtype in [np.float64, np.float32, np.int64, np.int32]
        assert len(X_train) > 0
        assert len(X_test) > 0
    
    def test_classification_preprocessing(self):
        """Test classification data preprocessing"""
        pipeline = DataPipeline('target', 0.2)
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(self.classification_df)
        
        # Check that target is encoded for classification
        assert y_train.dtype in [np.int64, np.int32]
        assert len(X_train) > 0
        assert len(X_test) > 0
    
    def test_regression_models_in_config(self):
        """Test that regression models are available in config"""
        regression_models = [
            'linear_regression', 'ridge_regression', 'lasso_regression',
            'random_forest_regressor', 'gradient_boosting_regressor',
            'knn_regressor', 'mlp_regressor'
        ]
        
        for model_name in regression_models:
            assert model_name in self.config.MODELS
            assert 'Regressor' in self.config.MODELS[model_name].estimator_class
    
    def test_regression_plan_executor(self):
        """Test plan executor with regression models"""
        executor = PlanExecutor(self.config)
        
        # Test regression model registry
        assert 'linear_regression' in executor.model_registry
        assert 'random_forest_regressor' in executor.model_registry
        
        # Test regression evaluation
        X_train, X_test, y_train, y_test = self._get_regression_data()
        
        # Create a simple regression plan
        plan = {
            'model_selection': {'primary_model': 'linear_regression'},
            'preprocessing_strategy': {'scaling': 'standard', 'missing_handling': 'median'},
            'hyperparameter_strategy': {
                'search_method': 'random',
                'n_trials': 5,
                'parameter_space': {}
            },
            'time_budget': 60
        }
        
        result = executor.execute_plan(plan, X_train, y_train, X_test, y_test)
        
        assert result['success'] is True
        assert 'test_r2' in result['test_metrics']
        assert 'test_rmse' in result['test_metrics']
        assert 'test_mae' in result['test_metrics']
    
    def test_classification_plan_executor(self):
        """Test plan executor with classification models"""
        executor = PlanExecutor(self.config)
        
        X_train, X_test, y_train, y_test = self._get_classification_data()
        
        # Create a simple classification plan
        plan = {
            'model_selection': {'primary_model': 'logistic_regression'},
            'preprocessing_strategy': {'scaling': 'standard', 'missing_handling': 'median'},
            'hyperparameter_strategy': {
                'search_method': 'random',
                'n_trials': 5,
                'parameter_space': {}
            },
            'time_budget': 60
        }
        
        result = executor.execute_plan(plan, X_train, y_train, X_test, y_test)
        
        assert result['success'] is True
        assert 'test_accuracy' in result['test_metrics']
        assert 'test_precision' in result['test_metrics']
        assert 'test_recall' in result['test_metrics']
        assert 'test_f1' in result['test_metrics']
    
    def _get_regression_data(self):
        """Get regression test data"""
        pipeline = DataPipeline('target', 0.2)
        return pipeline.preprocess_data(self.regression_df)
    
    def _get_classification_data(self):
        """Get classification test data"""
        pipeline = DataPipeline('target', 0.2)
        return pipeline.preprocess_data(self.classification_df)
