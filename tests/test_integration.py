"""
Integration tests for the complete system
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from autonomous_ml.core import AutonomousMLAgent
from autonomous_ml.config import Config


class TestIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.config = Config()
        self.agent = AutonomousMLAgent(self.config)
    
    def test_classification_pipeline(self):
        """Test complete classification pipeline"""
        # Create classification dataset
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            data_path = tmp_file.name
        
        try:
            # Run pipeline
            results = self.agent.run_pipeline(data_path, 'target', max_experiments=3)
            
            # Verify results structure
            assert 'pipeline_metadata' in results
            assert 'leaderboard' in results
            assert 'best_model' in results
            assert 'ensemble_strategy' in results
            assert 'insights' in results
            
            # Verify metadata
            metadata = results['pipeline_metadata']
            assert 'timestamp' in metadata
            assert 'dataset_profile' in metadata
            assert 'total_experiments' in metadata
            assert 'successful_experiments' in metadata
            
            # Verify leaderboard
            leaderboard = results['leaderboard']
            assert len(leaderboard) > 0
            assert 'test_accuracy' in leaderboard[0]
            assert 'model' in leaderboard[0]
            assert 'execution_time' in leaderboard[0]
            
            # Verify best model
            best_model = results['best_model']
            assert best_model['result'] is not None
            assert 'performance' in best_model
            
        finally:
            # Clean up
            import os
            os.unlink(data_path)
    
    def test_regression_pipeline(self):
        """Test complete regression pipeline"""
        # Create regression dataset
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            data_path = tmp_file.name
        
        try:
            # Run pipeline
            results = self.agent.run_pipeline(data_path, 'target', max_experiments=3)
            
            # Verify results structure
            assert 'pipeline_metadata' in results
            assert 'leaderboard' in results
            assert 'best_model' in results
            
            # Verify leaderboard has regression metrics
            leaderboard = results['leaderboard']
            if len(leaderboard) > 0:
                assert 'test_r2' in leaderboard[0]
                assert 'test_rmse' in leaderboard[0]
                assert 'test_mae' in leaderboard[0]
                assert 'task_type' in leaderboard[0]
                assert leaderboard[0]['task_type'] == 'regression'
            
            # Verify insights are regression-specific
            insights = results['insights']
            if 'performance_summary' in insights:
                assert 'best_r2' in insights['performance_summary']
                assert 'average_r2' in insights['performance_summary']
            
        finally:
            # Clean up
            import os
            os.unlink(data_path)
    
    def test_deployment_package_generation(self):
        """Test deployment package generation"""
        # First run a pipeline to have a trained model
        X, y = make_classification(n_samples=50, n_features=3, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])
        df['target'] = y
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            data_path = tmp_file.name
        
        try:
            # Run pipeline
            results = self.agent.run_pipeline(data_path, 'target', max_experiments=2)
            
            # Generate deployment package
            deployment_path = self.agent.generate_deployment_package()
            
            # Verify deployment directory exists
            from pathlib import Path
            deployment_dir = Path(deployment_path)
            assert deployment_dir.exists()
            assert deployment_dir.is_dir()
            
            # Verify required files exist
            assert (deployment_dir / "deployment_service.py").exists()
            assert (deployment_dir / "trained_pipeline.joblib").exists()
            assert (deployment_dir / "requirements.txt").exists()
            assert (deployment_dir / "Dockerfile").exists()
            
        finally:
            # Clean up
            import os
            os.unlink(data_path)
    
    def test_predict_functionality(self):
        """Test prediction functionality"""
        # Run a pipeline first
        X, y = make_classification(n_samples=50, n_features=3, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])
        df['target'] = y
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            data_path = tmp_file.name
        
        try:
            # Run pipeline
            results = self.agent.run_pipeline(data_path, 'target', max_experiments=2)
            
            # Test prediction
            test_data = pd.DataFrame({
                'feature_0': [1.0, 2.0, 3.0],
                'feature_1': [1.5, 2.5, 3.5],
                'feature_2': [0.5, 1.5, 2.5]
            })
            
            predictions = self.agent.predict(test_data)
            
            # Verify predictions
            assert len(predictions) == 3
            assert isinstance(predictions, np.ndarray)
            
        finally:
            # Clean up
            import os
            os.unlink(data_path)
    
    def test_error_handling(self):
        """Test error handling in pipeline"""
        # Test with invalid data path
        with pytest.raises(Exception):
            self.agent.run_pipeline("nonexistent_file.csv", "target")
        
        # Test with invalid target column
        X, y = make_classification(n_samples=50, n_features=3, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])
        df['target'] = y
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            data_path = tmp_file.name
        
        try:
            with pytest.raises(Exception):
                self.agent.run_pipeline(data_path, "nonexistent_column")
        finally:
            import os
            os.unlink(data_path)
    
    def test_async_pipeline(self):
        """Test async pipeline execution"""
        import asyncio
        
        # Create test data
        X, y = make_classification(n_samples=50, n_features=3, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])
        df['target'] = y
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            data_path = tmp_file.name
        
        try:
            # Test async execution
            async def run_async_test():
                results = await self.agent.run_pipeline_async(
                    data_path, 'target', time_budget=60
                )
                return results
            
            # Run async test
            results = asyncio.run(run_async_test())
            
            # Verify results
            assert 'pipeline_metadata' in results
            assert 'leaderboard' in results
            
        finally:
            import os
            os.unlink(data_path)
