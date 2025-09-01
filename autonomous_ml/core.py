"""
Core Autonomous ML Agent - Main orchestrator that coordinates all components
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
import joblib
import zipfile
import tempfile

from .config import Config
from .data_pipeline import DataPipeline
from .strategic_orchestrator import StrategicOrchestrator, DatasetProfile, ExperimentResult
from .experience_memory import ExperienceMemory
from .plan_executor import PlanExecutor
from .llm_client import StructuredLLMClient

class AutonomousMLAgent:
    """Main autonomous ML agent that orchestrates the entire pipeline"""
    
    def __init__(self, config: Optional[Config] = None, strategy: str = 'adaptive'):
        self.config = config or Config()
        self.strategy = strategy
        
        # Initialize components
        self.llm_client = StructuredLLMClient(
            provider=self.config.LLM.provider,
            model=self.config.LLM.model,
            api_key=self.config.LLM.api_key,
            max_tokens=self.config.LLM.max_tokens,
            temperature=self.config.LLM.temperature
        )
        
        self.experience_memory = ExperienceMemory()
        self.strategic_orchestrator = StrategicOrchestrator(
            self.llm_client, 
            self.experience_memory
        )
        self.plan_executor = PlanExecutor(self.config)
        
        self.results = {}
        self.experiment_history = []
        
        # Web interface state
        self.current_experiment = None
        self.experiment_progress = 0
        self.experiment_status = "idle"
        self.real_time_updates = []
        self.progress_callback = None
        
    def run_pipeline(self, data_path: str, target_column: str, 
                    test_size: float = 0.2, max_experiments: int = 10) -> Dict[str, Any]:
        """Run the complete autonomous ML pipeline"""
        
        print("🚀 Starting Autonomous ML Agent Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Load and analyze data
            print("\n📊 Step 1: Loading and analyzing data...")
            df = self._load_data(data_path)
            print(f"   Loaded dataset: {df.shape}")
            
            # Step 2: Initialize data pipeline and analyze
            data_pipeline = DataPipeline(target_column, test_size)
            dataset_profile = self._create_dataset_profile(data_pipeline.analyze_data(df))
            print(f"   Dataset complexity score: {dataset_profile.complexity_score:.3f}")
            
            # Step 3: Preprocess data
            print("\n🔧 Step 2: Preprocessing data...")
            X_train, X_test, y_train, y_test = data_pipeline.preprocess_data(df)
            print(f"   Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Step 4: Run experiments
            print("\n🧪 Step 3: Running strategic experiments...")
            experiment_results = self._run_experiments(
                dataset_profile, X_train, y_train, X_test, y_test, max_experiments
            )
            
            # Step 5: Generate ensemble strategy
            print("\n🎯 Step 4: Generating ensemble strategy...")
            ensemble_strategy = self.strategic_orchestrator.generate_ensemble_strategy(experiment_results)
            
            # Step 6: Compile results
            print("\n📋 Step 5: Compiling results...")
            self.results = self._compile_results(
                dataset_profile, experiment_results, ensemble_strategy, data_pipeline
            )
            
            # Step 7: Generate insights
            print("\n💡 Step 6: Generating insights...")
            self.results['insights'] = self._generate_insights()
            
            # Step 8: Save results
            self._save_results()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n✅ Pipeline completed in {total_time:.2f} seconds")
            print("=" * 60)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            raise
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from various formats"""
        path = Path(data_path)
        
        if path.suffix == '.csv':
            return pd.read_csv(data_path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(data_path)
        elif path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _create_dataset_profile(self, analysis: Dict[str, Any]) -> DatasetProfile:
        """Create dataset profile from analysis"""
        return DatasetProfile(
            shape=analysis['shape'],
            target_type=analysis['target_type'],
            target_distribution=analysis['target_distribution'],
            feature_types=analysis['feature_types'],
            missing_patterns=analysis['missing_patterns'],
            complexity_score=analysis['complexity_score'],
            domain_hints=analysis['domain_hints']
        )
    
    def _run_experiments(self, dataset_profile: DatasetProfile, 
                        X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        max_experiments: int) -> List[ExperimentResult]:
        """Run strategic experiments"""
        
        experiment_results = []
        
        for i in range(max_experiments):
            if not self.strategic_orchestrator.should_continue():
                break
            
            print(f"   Experiment {i+1}/{max_experiments}...")
            
            try:
                # Generate experiment plan
                plan = self.strategic_orchestrator.generate_experiment_plan(dataset_profile)
                
                # Execute plan
                result = self.plan_executor.execute_plan(
                    plan.__dict__, X_train, y_train, X_test, y_test
                )
                
                # Create experiment result
                experiment_result = ExperimentResult(
                    experiment_id=plan.experiment_id,
                    plan=plan,
                    performance_metrics=result.get('test_metrics', {}),
                    execution_time=result.get('execution_time', 0),
                    success=result.get('success', False),
                    insights=result.get('insights', [])
                )
                
                experiment_results.append(experiment_result)
                
                # Update orchestrator state
                self.strategic_orchestrator.update_experiment_state(experiment_result)
                
                # Print result
                if experiment_result.success:
                    score = experiment_result.performance_metrics.get('test_accuracy', 0)
                    print(f"     ✅ {plan.model_selection['primary_model']}: {score:.4f} accuracy")
                else:
                    print(f"     ❌ {plan.model_selection['primary_model']}: Failed")
                
            except Exception as e:
                print(f"     ❌ Experiment failed: {str(e)}")
                continue
        
        return experiment_results
    
    def _compile_results(self, dataset_profile: DatasetProfile, 
                        experiment_results: List[ExperimentResult],
                        ensemble_strategy: Dict[str, Any],
                        data_pipeline: DataPipeline) -> Dict[str, Any]:
        """Compile comprehensive results"""
        
        # Get best model
        successful_results = [r for r in experiment_results if r.success]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.performance_metrics.get('test_accuracy', 0))
        else:
            best_result = None
        
        # Create leaderboard
        leaderboard = []
        for result in successful_results:
            leaderboard.append({
                'model': result.plan.model_selection['primary_model'],
                'test_accuracy': result.performance_metrics.get('test_accuracy', 0),
                'test_precision': result.performance_metrics.get('test_precision', 0),
                'test_recall': result.performance_metrics.get('test_recall', 0),
                'test_f1': result.performance_metrics.get('test_f1', 0),
                'execution_time': result.execution_time,
                'insights': result.insights
            })
        
        # Sort by accuracy
        leaderboard.sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        return {
            'pipeline_metadata': {
                'timestamp': datetime.now().isoformat(),
                'dataset_profile': dataset_profile.__dict__,
                'strategy': self.strategy,
                'total_experiments': len(experiment_results),
                'successful_experiments': len(successful_results)
            },
            'leaderboard': leaderboard,
            'best_model': {
                'result': best_result.__dict__ if best_result else None,
                'performance': best_result.performance_metrics if best_result else {}
            },
            'ensemble_strategy': ensemble_strategy,
            'experiment_history': [r.__dict__ for r in experiment_results]
        }
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate insights from results"""
        
        if not self.results.get('leaderboard'):
            return {'error': 'No successful experiments'}
        
        leaderboard = self.results['leaderboard']
        best_model = leaderboard[0] if leaderboard else None
        
        insights = {
            'performance_summary': {
                'best_accuracy': best_model['test_accuracy'] if best_model else 0,
                'model_diversity': len(set(exp['model'] for exp in leaderboard)),
                'average_accuracy': np.mean([exp['test_accuracy'] for exp in leaderboard])
            },
            'recommendations': [
                f"Best performing model: {best_model['model']} with {best_model['test_accuracy']:.4f} accuracy" if best_model else "No successful models",
                f"Model diversity: {len(set(exp['model'] for exp in leaderboard))} different models tested",
                f"Average performance: {np.mean([exp['test_accuracy'] for exp in leaderboard]):.4f} accuracy"
            ],
            'technical_insights': []
        }
        
        # Add technical insights
        for exp in leaderboard[:3]:  # Top 3 models
            insights['technical_insights'].extend(exp.get('insights', []))
        
        return insights
    
    def _save_results(self):
        """Save results to files"""
        
        # Save main results
        results_path = self.config.RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"   Results saved to: {results_path}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best trained model"""
        
        if not self.results or not self.results.get('best_model', {}).get('result'):
            raise ValueError("No trained model available. Run pipeline first.")
        
        # This would require storing the trained pipeline
        # For now, return a placeholder
        return np.zeros(len(data))
    
    def generate_deployment_package(self) -> str:
        """Generate deployment package for the best model"""
        
        if not self.results or not self.results.get('best_model'):
            raise ValueError("No trained model available. Run pipeline first.")
        
        # Generate deployment code using LLM
        best_model = self.results['best_model']
        
        prompt = f"""
        Generate a complete FastAPI service for deploying this ML model:
        
        Model: {best_model.get('result', {}).get('plan', {}).get('model_selection', {}).get('primary_model', 'unknown')}
        Performance: {best_model.get('performance', {})}
        
        Include:
        1. FastAPI app with prediction endpoint
        2. Input validation using Pydantic
        3. Error handling
        4. Health check endpoint
        5. Docker configuration
        6. Requirements file
        
        Return complete, production-ready code.
        """
        
        deployment_code = self.llm_client.generate_code(prompt)
        
        # Save deployment package
        deployment_path = self.config.RESULTS_DIR / "deployment_package.py"
        with open(deployment_path, 'w') as f:
            f.write(deployment_code)
        
        print(f"   Deployment package saved to: {deployment_path}")
        return str(deployment_path)
    
    async def run_pipeline_async(
        self,
        data_path: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        time_budget: int = 900,
        target_metric: str = "accuracy",
        llm_provider: str = "openai",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run the complete ML pipeline with real-time progress updates"""
        
        try:
            self.experiment_status = "initializing"
            if progress_callback:
                await progress_callback("Initializing pipeline...", 0)
            
            # Load and profile data
            self.experiment_status = "loading_data"
            if progress_callback:
                await progress_callback("Loading and profiling data...", 10)
            
            df = self._load_data(data_path)
            data_pipeline = DataPipeline(target_column, 0.2)
            dataset_profile = self._create_dataset_profile(data_pipeline.analyze_data(df))
            
            # Find similar past experiences
            self.experiment_status = "learning_from_experience"
            if progress_callback:
                await progress_callback("Learning from past experiences...", 20)
            
            similar_experiences = self.experience_memory.find_similar_experiences(
                dataset_profile.__dict__
            )
            
            # Generate strategic plan
            self.experiment_status = "planning"
            if progress_callback:
                await progress_callback("Generating strategic plan...", 30)
            
            experiment_plan = self.strategic_orchestrator.generate_experiment_plan(dataset_profile)
            
            # Preprocess data
            X_train, X_test, y_train, y_test = data_pipeline.preprocess_data(df)
            
            # Execute experiments
            self.experiment_status = "training"
            if progress_callback:
                await progress_callback("Training models...", 40)
            
            results = await self._execute_experiments_async(
                dataset_profile, X_train, y_train, X_test, y_test, 
                min(10, time_budget // 60), progress_callback
            )
            
            # Compile results
            self.experiment_status = "analyzing"
            if progress_callback:
                await progress_callback("Analyzing results...", 90)
            
            self.results = self._compile_results(dataset_profile, results, {}, data_pipeline)
            self.results['insights'] = self._generate_insights()
            
            # Store experience
            self.experience_memory.store_experiment(
                dataset_profile.__dict__, experiment_plan.__dict__, results
            )
            
            self.experiment_status = "completed"
            if progress_callback:
                await progress_callback("Pipeline completed!", 100)
            
            return self.results
            
        except Exception as e:
            self.experiment_status = "error"
            if progress_callback:
                await progress_callback(f"Error: {str(e)}", -1)
            raise
    
    async def _execute_experiments_async(
        self, dataset_profile, X_train, y_train, X_test, y_test, 
        max_experiments, progress_callback
    ) -> List[ExperimentResult]:
        """Execute experiments with real-time progress updates"""
        
        results = []
        
        for i in range(max_experiments):
            if not self.strategic_orchestrator.should_continue():
                break
            
            if progress_callback:
                progress = 40 + (i / max_experiments) * 50
                await progress_callback(f"Training experiment {i+1}/{max_experiments}...", progress)
            
            try:
                # Generate experiment plan
                plan = self.strategic_orchestrator.generate_experiment_plan(dataset_profile)
                
                # Execute plan
                result = self.plan_executor.execute_plan(
                    plan.__dict__, X_train, y_train, X_test, y_test
                )
                
                # Create experiment result
                experiment_result = ExperimentResult(
                    experiment_id=plan.experiment_id,
                    plan=plan,
                    performance_metrics=result.get('test_metrics', {}),
                    execution_time=result.get('execution_time', 0),
                    success=result.get('success', False),
                    insights=result.get('insights', [])
                )
                
                results.append(experiment_result)
                
                # Update orchestrator state
                self.strategic_orchestrator.update_experiment_state(experiment_result)
                
                # Real-time update
                self.real_time_updates.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": plan.model_selection['primary_model'],
                    "score": result.get('test_metrics', {}).get('test_accuracy', 0),
                    "status": "completed" if experiment_result.success else "failed"
                })
                
            except Exception as e:
                self.real_time_updates.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": "unknown",
                    "score": 0,
                    "status": f"error: {str(e)}"
                })
                continue
        
        return results
