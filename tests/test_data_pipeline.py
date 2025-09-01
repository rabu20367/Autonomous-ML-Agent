"""
Tests for the data pipeline module
"""

import pytest
import pandas as pd
import numpy as np
from autonomous_ml.data_pipeline import DataPipeline

class TestDataPipeline:
    """Test cases for the DataPipeline class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'numerical_feature': np.random.randn(n_samples),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
            'datetime_feature': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def pipeline(self):
        """Create test pipeline"""
        return DataPipeline('target', test_size=0.2)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = DataPipeline('target', test_size=0.3)
        assert pipeline.target_column == 'target'
        assert pipeline.test_size == 0.3
        assert pipeline.preprocessor is None
        assert pipeline.label_encoder is None
    
    def test_analyze_data(self, pipeline, sample_data):
        """Test data analysis"""
        analysis = pipeline.analyze_data(sample_data)
        
        assert 'shape' in analysis
        assert 'target_type' in analysis
        assert 'target_distribution' in analysis
        assert 'feature_types' in analysis
        assert 'missing_patterns' in analysis
        assert 'complexity_score' in analysis
        assert 'domain_hints' in analysis
        
        assert analysis['shape'] == (100, 4)
        assert analysis['target_type'] == 'classification'
        assert 'numerical_feature' in analysis['feature_types']
        assert 'categorical_feature' in analysis['feature_types']
        assert 'datetime_feature' in analysis['feature_types']
        assert 0 <= analysis['complexity_score'] <= 1
    
    def test_determine_target_type(self, pipeline):
        """Test target type determination"""
        # Classification case
        classification_target = pd.Series([0, 1, 0, 1, 0])
        assert pipeline._determine_target_type(classification_target) == 'classification'
        
        # Categorical case
        categorical_target = pd.Series(['A', 'B', 'A', 'C'])
        assert pipeline._determine_target_type(categorical_target) == 'classification'
        
        # Regression case
        regression_target = pd.Series(np.random.randn(100))
        assert pipeline._determine_target_type(regression_target) == 'regression'
    
    def test_classify_features(self, pipeline, sample_data):
        """Test feature classification"""
        feature_types = pipeline._classify_features(sample_data)
        
        assert feature_types['numerical_feature'] == 'numerical'
        assert feature_types['categorical_feature'] == 'categorical'
        assert feature_types['datetime_feature'] == 'datetime'
        assert 'target' not in feature_types  # Target should be excluded
    
    def test_analyze_missing_values(self, pipeline, sample_data):
        """Test missing value analysis"""
        # Add some missing values
        sample_data.loc[0:5, 'numerical_feature'] = np.nan
        sample_data.loc[10:15, 'categorical_feature'] = np.nan
        
        missing_patterns = pipeline._analyze_missing_values(sample_data)
        
        assert missing_patterns['numerical_feature'] > 0
        assert missing_patterns['categorical_feature'] > 0
        assert 'target' not in missing_patterns  # Target should be excluded
    
    def test_calculate_complexity_score(self, pipeline, sample_data):
        """Test complexity score calculation"""
        complexity_score = pipeline._calculate_complexity_score(sample_data)
        
        assert 0 <= complexity_score <= 1
        assert isinstance(complexity_score, float)
    
    def test_extract_domain_hints(self, pipeline):
        """Test domain hint extraction"""
        # Test with age column
        data_with_age = pd.DataFrame({
            'age': [25, 30, 35],
            'price': [100, 200, 300],
            'target': [0, 1, 0]
        })
        
        hints = pipeline._extract_domain_hints(data_with_age)
        
        assert 'demographic_data' in hints
        assert 'financial_data' in hints
    
    def test_handle_datetime_features(self, pipeline, sample_data):
        """Test datetime feature handling"""
        processed_data = pipeline._handle_datetime_features(sample_data)
        
        # Check that datetime features are expanded
        assert 'datetime_feature_year' in processed_data.columns
        assert 'datetime_feature_month' in processed_data.columns
        assert 'datetime_feature_day' in processed_data.columns
        assert 'datetime_feature_dayofweek' in processed_data.columns
        assert 'datetime_feature_hour' in processed_data.columns
        
        # Check that original datetime column is removed
        assert 'datetime_feature' not in processed_data.columns
    
    def test_preprocess_data(self, pipeline, sample_data):
        """Test complete preprocessing pipeline"""
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(sample_data)
        
        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == 100
        assert y_train.shape[0] + y_test.shape[0] == 100
        assert X_train.shape[1] == X_test.shape[1]
        
        # Check that preprocessor is fitted
        assert pipeline.preprocessor is not None
        assert pipeline.feature_names is not None
        
        # Check that target is encoded if categorical
        if sample_data['target'].dtype == 'object':
            assert pipeline.label_encoder is not None
    
    def test_transform_new_data(self, pipeline, sample_data):
        """Test transforming new data"""
        # First fit the pipeline
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(sample_data)
        
        # Create new data
        new_data = sample_data.head(10)
        
        # Transform new data
        transformed_data = pipeline.transform_new_data(new_data)
        
        # Check that transformation works
        assert transformed_data.shape[0] == 10
        assert transformed_data.shape[1] == len(pipeline.feature_names)
        assert list(transformed_data.columns) == pipeline.feature_names
    
    def test_preprocess_data_with_missing_values(self, pipeline):
        """Test preprocessing with missing values"""
        # Create data with missing values
        data_with_missing = pd.DataFrame({
            'numerical_feature': [1, 2, np.nan, 4, 5],
            'categorical_feature': ['A', 'B', np.nan, 'C', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(data_with_missing)
        
        # Check that missing values are handled
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()
        assert not y_train.isnull().any()
        assert not y_test.isnull().any()
    
    def test_preprocess_data_with_categorical_target(self, pipeline):
        """Test preprocessing with categorical target"""
        # Create data with categorical target
        data_with_categorical_target = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': ['class_A', 'class_B', 'class_A', 'class_B', 'class_A']
        })
        
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(data_with_categorical_target)
        
        # Check that target is encoded
        assert pipeline.label_encoder is not None
        assert y_train.dtype in ['int64', 'int32']
        assert y_test.dtype in ['int64', 'int32']
