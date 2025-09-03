"""
Intelligent Data Pipeline - Automated data analysis and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPipeline:
    """Intelligent data analysis and preprocessing pipeline"""
    
    def __init__(self, target_column: str, test_size: float = 0.2):
        self.target_column = target_column
        self.test_size = test_size
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data analysis for strategic planning"""
        
        analysis = {
            'shape': df.shape,
            'target_type': self._determine_target_type(df[self.target_column]),
            'target_distribution': df[self.target_column].value_counts(normalize=True).to_dict(),
            'feature_types': self._classify_features(df),
            'missing_patterns': self._analyze_missing_values(df),
            'complexity_score': self._calculate_complexity_score(df),
            'domain_hints': self._extract_domain_hints(df)
        }
        
        return analysis
    
    def _determine_target_type(self, target_series: pd.Series) -> str:
        """Determine if target is classification or regression"""
        # Check data type first
        if target_series.dtype in ['object', 'category']:
            return 'classification'
        
        # For numeric data, check if values are integers and have limited unique values
        if target_series.dtype in ['int64', 'int32', 'int16', 'int8']:
            unique_count = target_series.nunique()
            # If it's integer with few unique values, likely classification
            if unique_count <= 20:
                return 'classification'
            # If it's integer with many unique values, likely regression
            return 'regression'
        
        # For float data, check if values are actually integers
        if target_series.dtype in ['float64', 'float32', 'float16']:
            # Check if all values are effectively integers
            if target_series.dropna().apply(lambda x: x.is_integer()).all():
                unique_count = target_series.nunique()
                if unique_count <= 20:
                    return 'classification'
                return 'regression'
            else:
                # Has non-integer values, definitely regression
                return 'regression'
        
        # Default to regression for other numeric types
        return 'regression'
    
    def _classify_features(self, df: pd.DataFrame) -> Dict[str, str]:
        """Classify features by type"""
        feature_types = {}
        
        for col in df.columns:
            if col == self.target_column:
                continue
                
            if df[col].dtype in ['object', 'category']:
                feature_types[col] = 'categorical'
            elif df[col].dtype in ['datetime64']:
                feature_types[col] = 'datetime'
            else:
                feature_types[col] = 'numerical'
        
        return feature_types
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze missing value patterns"""
        missing_analysis = {}
        
        for col in df.columns:
            if col == self.target_column:
                continue
                
            missing_pct = df[col].isnull().sum() / len(df)
            missing_analysis[col] = missing_pct
        
        return missing_analysis
    
    def _calculate_complexity_score(self, df: pd.DataFrame) -> float:
        """Calculate dataset complexity score"""
        complexity_factors = []
        
        # Feature count factor
        n_features = len(df.columns) - 1  # Exclude target
        complexity_factors.append(min(n_features / 100, 1.0))
        
        # Missing values factor
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        complexity_factors.append(missing_pct)
        
        # Categorical features factor
        categorical_pct = sum(1 for col in df.columns if col != self.target_column and df[col].dtype in ['object', 'category']) / (len(df.columns) - 1)
        complexity_factors.append(categorical_pct)
        
        # Target imbalance factor
        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts(normalize=True)
            imbalance = 1 - target_dist.max()  # 0 = balanced, 1 = completely imbalanced
            complexity_factors.append(imbalance)
        
        return np.mean(complexity_factors)
    
    def _extract_domain_hints(self, df: pd.DataFrame) -> List[str]:
        """Extract domain hints from column names and data"""
        hints = []
        
        # Analyze column names
        column_names = [col.lower() for col in df.columns]
        
        if any('age' in name for name in column_names):
            hints.append('demographic_data')
        if any('price' in name or 'cost' in name for name in column_names):
            hints.append('financial_data')
        if any('date' in name or 'time' in name for name in column_names):
            hints.append('temporal_data')
        if any('score' in name or 'rating' in name for name in column_names):
            hints.append('scoring_data')
        
        # Analyze data patterns
        if df.select_dtypes(include=['datetime64']).shape[1] > 0:
            hints.append('time_series_data')
        
        return hints
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Main preprocessing pipeline"""
        
        # Handle datetime features
        df_processed = self._handle_datetime_features(df)
        
        # Separate features and target
        if self.target_column not in df_processed.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]
        
        # Build preprocessing pipeline
        self.preprocessor = self._build_preprocessor(X)
        
        # Fit and transform features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        self.feature_names = self._get_feature_names(X)
        
        # Handle target variable based on type
        target_type = self._determine_target_type(y)
        if target_type == 'classification' and (y.dtype == 'object' or y.dtype.name == 'category'):
            self.label_encoder = LabelEncoder()
            y_processed = self.label_encoder.fit_transform(y)
        else:
            # For regression or already numeric classification targets
            y_processed = y.values
        
        # Convert back to DataFrame/Series
        X_processed = pd.DataFrame(X_processed, columns=self.feature_names, index=X.index)
        y_processed = pd.Series(y_processed, index=y.index, name=self.target_column)
        
        # Split data
        if target_type == 'classification' and len(y_processed.unique()) > 1:
            # Use stratification for classification with multiple classes
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=self.test_size, random_state=42, stratify=y_processed
            )
        else:
            # No stratification for regression or single-class classification
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=self.test_size, random_state=42
            )
        
        return X_train, X_test, y_train, y_test
    
    def _handle_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns"""
        df_processed = df.copy()
        
        for col in df.select_dtypes(include=['datetime64']).columns:
            df_processed[f'{col}_year'] = df_processed[col].dt.year
            df_processed[f'{col}_month'] = df_processed[col].dt.month
            df_processed[f'{col}_day'] = df_processed[col].dt.day
            df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
            df_processed[f'{col}_hour'] = df_processed[col].dt.hour
            
            # Drop original datetime column
            df_processed = df_processed.drop(columns=[col])
        
        return df_processed
    
    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline"""
        
        # Identify column types
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Build transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('cat', categorical_transformer, categorical_columns)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get feature names after preprocessing"""
        try:
            # Try to get feature names from the preprocessor
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                return self.preprocessor.get_feature_names_out().tolist()
        except Exception:
            pass
        
        # Fallback: manual feature name construction
        feature_names = []
        
        # Numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        feature_names.extend(numerical_columns)
        
        # Categorical features (after one-hot encoding)
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_columns and 'cat' in self.preprocessor.named_transformers_:
            try:
                cat_transformer = self.preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                    onehot_encoder = cat_transformer.named_steps['onehot']
                    if hasattr(onehot_encoder, 'get_feature_names_out'):
                        cat_features = onehot_encoder.get_feature_names_out(categorical_columns)
                        feature_names.extend(cat_features)
                    else:
                        # Fallback for older sklearn versions
                        feature_names.extend(categorical_columns)
                else:
                    feature_names.extend(categorical_columns)
            except Exception:
                # If anything fails, just use original column names
                feature_names.extend(categorical_columns)
        
        # Ensure we have the right number of features
        if len(feature_names) != X.shape[1]:
            # Generate generic feature names
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        return feature_names
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call preprocess_data first.")
        
        # Handle datetime features
        df_processed = self._handle_datetime_features(df)
        
        # Remove target column if present
        if self.target_column in df_processed.columns:
            df_processed = df_processed.drop(columns=[self.target_column])
        
        # Transform features
        X_processed = self.preprocessor.transform(df_processed)
        X_processed = pd.DataFrame(X_processed, columns=self.feature_names, index=df_processed.index)
        
        return X_processed
