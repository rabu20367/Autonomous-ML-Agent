"""
Configuration management with environment-based settings
"""

import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    name: str
    estimator_class: str
    hyperparameter_space: Dict[str, Any]
    default_params: Dict[str, Any]

@dataclass
class LLMConfig:
    provider: str = os.getenv('LLM_PROVIDER', 'openai')
    model: str = os.getenv('LLM_MODEL', 'gpt-4')
    api_key: str = os.getenv('LLM_API_KEY', '')
    max_tokens: int = int(os.getenv('LLM_MAX_TOKENS', '4000'))
    temperature: float = float(os.getenv('LLM_TEMPERATURE', '0.1'))

class Config:
    """Centralized configuration with intelligent defaults"""
    
    # Model registry with optimized hyperparameter spaces
    MODELS = {
        # Classification models
        'logistic_regression': ModelConfig(
            name='Logistic Regression',
            estimator_class='sklearn.linear_model.LogisticRegression',
            hyperparameter_space={
                'C': (0.01, 100.0),
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': (100, 1000)
            },
            default_params={'random_state': 42}
        ),
        'random_forest': ModelConfig(
            name='Random Forest',
            estimator_class='sklearn.ensemble.RandomForestClassifier',
            hyperparameter_space={
                'n_estimators': (50, 500),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            default_params={'random_state': 42}
        ),
        'gradient_boosting': ModelConfig(
            name='Gradient Boosting',
            estimator_class='sklearn.ensemble.GradientBoostingClassifier',
            hyperparameter_space={
                'n_estimators': (50, 500),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            },
            default_params={'random_state': 42}
        ),
        'knn': ModelConfig(
            name='k-Nearest Neighbors',
            estimator_class='sklearn.neighbors.KNeighborsClassifier',
            hyperparameter_space={
                'n_neighbors': (3, 50),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': (1, 2)
            },
            default_params={}
        ),
        'mlp': ModelConfig(
            name='Multi-layer Perceptron',
            estimator_class='sklearn.neural_network.MLPClassifier',
            hyperparameter_space={
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': (0.0001, 0.1),
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': (200, 1000)
            },
            default_params={'random_state': 42}
        ),
        # Regression models
        'linear_regression': ModelConfig(
            name='Linear Regression',
            estimator_class='sklearn.linear_model.LinearRegression',
            hyperparameter_space={
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            default_params={}
        ),
        'ridge_regression': ModelConfig(
            name='Ridge Regression',
            estimator_class='sklearn.linear_model.Ridge',
            hyperparameter_space={
                'alpha': (0.01, 100.0),
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            default_params={'random_state': 42}
        ),
        'lasso_regression': ModelConfig(
            name='Lasso Regression',
            estimator_class='sklearn.linear_model.Lasso',
            hyperparameter_space={
                'alpha': (0.01, 10.0),
                'fit_intercept': [True, False],
                'normalize': [True, False],
                'max_iter': (100, 2000)
            },
            default_params={'random_state': 42}
        ),
        'random_forest_regressor': ModelConfig(
            name='Random Forest Regressor',
            estimator_class='sklearn.ensemble.RandomForestRegressor',
            hyperparameter_space={
                'n_estimators': (50, 500),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            default_params={'random_state': 42}
        ),
        'gradient_boosting_regressor': ModelConfig(
            name='Gradient Boosting Regressor',
            estimator_class='sklearn.ensemble.GradientBoostingRegressor',
            hyperparameter_space={
                'n_estimators': (50, 500),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            },
            default_params={'random_state': 42}
        ),
        'knn_regressor': ModelConfig(
            name='k-Nearest Neighbors Regressor',
            estimator_class='sklearn.neighbors.KNeighborsRegressor',
            hyperparameter_space={
                'n_neighbors': (3, 50),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': (1, 2)
            },
            default_params={}
        ),
        'mlp_regressor': ModelConfig(
            name='Multi-layer Perceptron Regressor',
            estimator_class='sklearn.neural_network.MLPRegressor',
            hyperparameter_space={
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': (0.0001, 0.1),
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': (200, 1000)
            },
            default_params={'random_state': 42}
        )
    }
    
    # Optimization settings
    OPTIMIZATION = {
        'method': 'bayesian',
        'n_trials': int(os.getenv('OPT_N_TRIALS', '50')),
        'timeout': int(os.getenv('OPT_TIMEOUT', '3600')),
        'cv_folds': int(os.getenv('OPT_CV_FOLDS', '5')),
        'scoring': os.getenv('OPT_SCORING', 'accuracy')
    }
    
    # LLM configuration
    LLM = LLMConfig()
    
    # Paths with automatic creation
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    RESULTS_DIR = BASE_DIR / 'results'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Create directories
    for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
        dir_path.mkdir(exist_ok=True)
