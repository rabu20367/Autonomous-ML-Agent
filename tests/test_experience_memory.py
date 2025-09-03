"""
Test experience memory functionality
"""

import pytest
import tempfile
import os
from pathlib import Path
from autonomous_ml.experience_memory import ExperienceMemory


class TestExperienceMemory:
    """Test experience memory system"""
    
    def setup_method(self):
        """Set up test environment"""
        # Create temporary directory for test databases
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_experiences.db")
        self.chroma_path = os.path.join(self.temp_dir, "test_chroma")
        
        # Set environment variables
        os.environ['EXPERIENCE_DB_PATH'] = self.db_path
        os.environ['CHROMA_DB_PATH'] = self.chroma_path
        
        self.memory = ExperienceMemory()
    
    def teardown_method(self):
        """Clean up test environment"""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_memory_initialization(self):
        """Test that memory initializes correctly"""
        assert self.memory.db_path == Path(self.db_path).resolve()
        assert self.memory.collection_name == "ml_experiences"
        assert self.memory.chroma_client is not None
        assert self.memory.collection is not None
    
    def test_store_experiment(self):
        """Test storing an experiment"""
        experiment_plan = {
            "experiment_id": "test_exp_1",
            "model_selection": {"primary_model": "random_forest"},
            "preprocessing_strategy": {"scaling": "standard"}
        }
        
        results = {
            "success": True,
            "performance_metrics": {"test_accuracy": 0.85},
            "execution_time": 120.5
        }
        
        # Store experiment
        self.memory.store_experiment(experiment_plan, results)
        
        # Verify it was stored in SQLite
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", ("test_exp_1",))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[5] == 1  # success = True
        assert row[6] == 0.85  # performance_score
    
    def test_find_similar_experiences(self):
        """Test finding similar experiences"""
        # Store a few experiments
        experiments = [
            {
                "experiment_id": "exp_1",
                "model_selection": {"primary_model": "random_forest"},
                "preprocessing_strategy": {"scaling": "standard"}
            },
            {
                "experiment_id": "exp_2", 
                "model_selection": {"primary_model": "logistic_regression"},
                "preprocessing_strategy": {"scaling": "robust"}
            }
        ]
        
        results = [
            {"success": True, "performance_metrics": {"test_accuracy": 0.85}},
            {"success": True, "performance_metrics": {"test_accuracy": 0.78}}
        ]
        
        for exp, res in zip(experiments, results):
            self.memory.store_experiment(exp, res)
        
        # Find similar experiences
        dataset_profile = {
            "shape": (1000, 10),
            "target_type": "classification"
        }
        
        similar = self.memory.find_similar_experiences(dataset_profile, n_results=2)
        
        assert len(similar) <= 2
        for exp in similar:
            assert "experiment_plan" in exp
            assert "results" in exp
            assert "similarity_score" in exp
            assert "metadata" in exp
    
    def test_global_insights(self):
        """Test getting global insights"""
        # Store some experiments
        experiments = [
            {
                "experiment_id": f"exp_{i}",
                "model_selection": {"primary_model": "random_forest"},
                "preprocessing_strategy": {"scaling": "standard"}
            }
            for i in range(3)
        ]
        
        results = [
            {"success": True, "performance_metrics": {"test_accuracy": 0.85 + i*0.01}}
            for i in range(3)
        ]
        
        for exp, res in zip(experiments, results):
            self.memory.store_experiment(exp, res)
        
        # Get global insights
        insights = self.memory.get_global_insights()
        
        assert "total_experiments" in insights
        assert "average_performance" in insights
        assert "unique_datasets" in insights
        assert "top_models" in insights
        
        assert insights["total_experiments"] == 3
        assert insights["average_performance"] > 0.85
        assert len(insights["top_models"]) > 0
    
    def test_path_handling(self):
        """Test that paths are handled correctly"""
        # Test with relative paths
        memory_rel = ExperienceMemory("relative_path.db", "test_collection")
        assert memory_rel.db_path.is_absolute()
        
        # Test with absolute paths
        abs_path = os.path.abspath("absolute_path.db")
        memory_abs = ExperienceMemory(abs_path, "test_collection")
        assert memory_abs.db_path == Path(abs_path).resolve()
    
    def test_error_handling(self):
        """Test error handling in memory operations"""
        # Test with invalid experiment data
        invalid_plan = {"invalid": "data"}
        invalid_results = {"invalid": "results"}
        
        # Should not raise exception
        self.memory.store_experiment(invalid_plan, invalid_results)
        
        # Test finding similar with empty memory
        dataset_profile = {"shape": (100, 5), "target_type": "regression"}
        similar = self.memory.find_similar_experiences(dataset_profile)
        assert isinstance(similar, list)
