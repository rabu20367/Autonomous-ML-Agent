"""
Generate Iris dataset for testing the Autonomous ML Agent
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def create_iris_dataset():
    """Create and save Iris dataset"""
    
    # Load the iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    # Add some missing values for testing
    np.random.seed(42)
    missing_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[missing_indices, 'sepal length (cm)'] = np.nan
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, df.shape)
    df.iloc[:, :-1] += noise[:, :-1]
    
    # Save to CSV
    os.makedirs('examples/datasets', exist_ok=True)
    df.to_csv('examples/datasets/iris.csv', index=False)
    
    print("✅ Iris dataset created successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Target distribution: {df['species'].value_counts().to_dict()}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df

if __name__ == "__main__":
    create_iris_dataset()
