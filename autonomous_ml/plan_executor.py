"""
Plan Executor - Converts structured plans to executable ML pipelines with budget awareness
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import time
import warnings
warnings.filterwarnings('ignore')

class PlanExecutor:
    """Executes structured ML experiment plans with budget awareness"""
    
    def __init__(self, config):
        self.config = config
        self.model_registry = self._build_model_registry()
        self.preprocessor_registry = self._build_preprocessor_registry()
        
    def _build_model_registry(self) -> Dict[str, Any]:
        """Build registry of available models"""
        return {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'knn': KNeighborsClassifier,
            'mlp': MLPClassifier
        }
    
    def _build_preprocessor_registry(self) -> Dict[str, Any]:
        """Build registry of preprocessing strategies"""
        return {
            'standard_scaler': StandardScaler,
            'robust_scaler': RobustScaler,
            'simple_imputer': SimpleImputer,
            'knn_imputer': KNNImputer,
            'onehot_encoder': OneHotEncoder,
            'select_k_best': SelectKBest
        }
    
    def execute_plan(self, plan: Dict[str, Any], X_train: pd.DataFrame, 
                    y_train: pd.Series, X_test: pd.DataFrame, 
                    y_test: pd.Series) -> Dict[str, Any]:
        """Execute a structured experiment plan with budget constraints"""
        
        start_time = time.time()
        
        try:
            # Build preprocessing pipeline
            preprocessor = self._build_preprocessor(plan['preprocessing_strategy'], X_train)
            
            # Build model pipeline
            model_pipeline = self._build_model_pipeline(
                plan['model_selection'], 
                plan['hyperparameter_strategy']
            )
            
            # Create complete pipeline
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model_pipeline)
            ])
            
            # Execute hyperparameter search with budget awareness
            search_results = self._execute_hyperparameter_search(
                full_pipeline, 
                plan['hyperparameter_strategy'],
                X_train, y_train,
                plan.get('time_budget', 300)
            )
            
            # Evaluate on test set
            best_pipeline = search_results['best_estimator']
            test_metrics = self._evaluate_pipeline(best_pipeline, X_test, y_test)
            
            # Extract feature importance if available
            feature_importance = self._extract_feature_importance(best_pipeline, X_train.columns)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'best_pipeline': best_pipeline,
                'best_params': search_results['best_params'],
                'best_score': search_results['best_score'],
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'execution_time': execution_time,
                'search_results': search_results,
                'insights': self._generate_insights(search_results, test_metrics)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'insights': [f"Experiment failed: {str(e)}"]
            }
    
    def _build_preprocessor(self, preprocessing_strategy: Dict[str, Any], 
                           X_train: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline from strategy"""
        
        # Identify column types
        numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Build numerical transformer
        numerical_steps = []
        
        # Missing value handling
        missing_strategy = preprocessing_strategy.get('missing_handling', 'median')
        if missing_strategy == 'knn':
            numerical_steps.append(('imputer', KNNImputer()))
        else:
            numerical_steps.append(('imputer', SimpleImputer(strategy=missing_strategy)))
        
        # Scaling
        scaling_strategy = preprocessing_strategy.get('scaling', 'standard')
        if scaling_strategy == 'robust':
            numerical_steps.append(('scaler', RobustScaler()))
        else:
            numerical_steps.append(('scaler', StandardScaler()))
        
        # Build categorical transformer
        categorical_steps = []
        categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
        encoding_strategy = preprocessing_strategy.get('categorical_encoding', 'onehot')
        if encoding_strategy == 'onehot':
            categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        
        # Create transformers
        transformers = []
        if numerical_columns:
            transformers.append(('num', Pipeline(numerical_steps), numerical_columns))
        if categorical_columns:
            transformers.append(('cat', Pipeline(categorical_steps), categorical_columns))
        
        return ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    def _build_model_pipeline(self, model_selection: Dict[str, Any], 
                             hyperparameter_strategy: Dict[str, Any]) -> Any:
        """Build model with hyperparameter strategy"""
        
        model_name = model_selection['primary_model']
        model_class = self.model_registry.get(model_name)
        
        if not model_class:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create base model with default parameters
        return model_class()
    
    def _execute_hyperparameter_search(self, pipeline: Pipeline, 
                                     hyperparameter_strategy: Dict[str, Any],
                                     X_train: pd.DataFrame, y_train: pd.Series,
                                     time_budget: int) -> Dict[str, Any]:
        """Execute hyperparameter search with budget constraints"""
        
        search_method = hyperparameter_strategy.get('search_method', 'random')
        n_trials = hyperparameter_strategy.get('n_trials', 20)
        param_space = hyperparameter_strategy.get('parameter_space', {})
        
        # Adjust n_trials based on time budget
        n_trials = min(n_trials, max(5, time_budget // 10))  # Rough estimate
        
        # Prepare parameter space for sklearn
        sklearn_params = {}
        for param, space in param_space.items():
            if isinstance(space, list):
                sklearn_params[param] = space
            elif isinstance(space, tuple) and len(space) == 2:
                # Convert to uniform distribution
                sklearn_params[param] = np.linspace(space[0], space[1], 10).tolist()
        
        # Execute search
        if search_method == 'grid':
            search = GridSearchCV(
                pipeline, sklearn_params, 
                cv=3, scoring='accuracy', n_jobs=-1
            )
        else:  # random or bayesian
            search = RandomizedSearchCV(
                pipeline, sklearn_params,
                n_iter=n_trials, cv=3, scoring='accuracy', n_jobs=-1
            )
        
        search.fit(X_train, y_train)
        
        return {
            'best_estimator': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
    
    def _evaluate_pipeline(self, pipeline: Pipeline, X_test: pd.DataFrame, 
                          y_test: pd.Series) -> Dict[str, float]:
        """Evaluate pipeline on test set"""
        
        y_pred = pipeline.predict(X_test)
        
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'test_f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def _extract_feature_importance(self, pipeline: Pipeline, 
                                   feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from trained pipeline"""
        
        try:
            model = pipeline.named_steps['model']
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                return None
            
            # Map to feature names (simplified - in practice, handle preprocessing)
            feature_importance = dict(zip(feature_names[:len(importance)], importance))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception:
            return None
    
    def _generate_insights(self, search_results: Dict[str, Any], 
                          test_metrics: Dict[str, float]) -> List[str]:
        """Generate insights from experiment results"""
        
        insights = []
        
        # Performance insights
        best_score = search_results['best_score']
        test_accuracy = test_metrics['test_accuracy']
        
        if test_accuracy > best_score:
            insights.append(f"Model generalizes well: test accuracy ({test_accuracy:.3f}) > CV score ({best_score:.3f})")
        elif test_accuracy < best_score - 0.05:
            insights.append(f"Potential overfitting: test accuracy ({test_accuracy:.3f}) < CV score ({best_score:.3f})")
        
        # Parameter insights
        best_params = search_results['best_params']
        if 'n_estimators' in best_params and best_params['n_estimators'] > 200:
            insights.append("High n_estimators suggests complex patterns in data")
        
        if 'max_depth' in best_params and best_params['max_depth'] > 10:
            insights.append("Deep trees indicate non-linear relationships")
        
        return insights
