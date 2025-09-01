"""
Generate Housing dataset for testing the Autonomous ML Agent
"""

import pandas as pd
import numpy as np
import os

def create_housing_dataset():
    """Create and save Housing-like dataset"""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic housing data
    data = {
        'crim': np.random.exponential(1.0, n_samples),  # Crime rate
        'zn': np.random.exponential(10.0, n_samples),   # Residential land zoned
        'indus': np.random.normal(10.0, 7.0, n_samples).clip(0, 30),  # Industry
        'chas': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # Charles River
        'nox': np.random.normal(0.5, 0.1, n_samples).clip(0.3, 0.9),  # Nitric oxides
        'rm': np.random.normal(6.0, 0.7, n_samples).clip(4.0, 9.0),  # Rooms
        'age': np.random.exponential(70.0, n_samples).clip(0, 100),  # Age
        'dis': np.random.exponential(4.0, n_samples).clip(1.0, 15.0),  # Distance to employment
        'rad': np.random.choice(range(1, 25), n_samples),  # Accessibility to highways
        'tax': np.random.normal(400, 150, n_samples).clip(150, 800),  # Property tax
        'ptratio': np.random.normal(18.0, 2.0, n_samples).clip(12.0, 23.0),  # Pupil-teacher ratio
        'b': np.random.normal(350, 100, n_samples).clip(0, 400),  # Black proportion
        'lstat': np.random.exponential(12.0, n_samples).clip(1.0, 40.0)  # Lower status
    }
    
    df = pd.DataFrame(data)
    
    # Generate house prices based on features (regression target)
    price = (
        50.0 - 0.1 * df['crim'] - 0.05 * df['nox'] + 5.0 * df['rm'] +
        0.1 * df['age'] - 1.0 * df['dis'] - 0.01 * df['tax'] -
        0.5 * df['ptratio'] + 0.1 * df['b'] - 0.5 * df['lstat'] +
        np.random.normal(0, 5, n_samples)
    )
    
    df['medv'] = price.clip(5, 50)  # Median value of homes
    
    # Add some missing values
    missing_crim = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_crim, 'crim'] = np.nan
    
    missing_rm = np.random.choice(df.index, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_rm, 'rm'] = np.nan
    
    # Save to CSV
    os.makedirs('examples/datasets', exist_ok=True)
    df.to_csv('examples/datasets/housing.csv', index=False)
    
    print("✅ Housing dataset created successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Target range: {df['medv'].min():.2f} - {df['medv'].max():.2f}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df

if __name__ == "__main__":
    create_housing_dataset()
