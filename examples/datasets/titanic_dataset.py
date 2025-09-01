"""
Generate Titanic dataset for testing the Autonomous ML Agent
"""

import pandas as pd
import numpy as np
import os

def create_titanic_dataset():
    """Create and save Titanic-like dataset"""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic Titanic-like data
    data = {
        'passenger_id': range(1, n_samples + 1),
        'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'age': np.random.normal(30, 15, n_samples).clip(0, 80),
        'sibsp': np.random.poisson(0.5, n_samples),
        'parch': np.random.poisson(0.4, n_samples),
        'ticket': [f'Ticket_{i}' for i in range(1, n_samples + 1)],
        'fare': np.random.lognormal(2.5, 1.2, n_samples).clip(0, 500),
        'cabin': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', np.nan], n_samples, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]),
        'embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.2, 0.1, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Generate survival based on features (more realistic)
    survival_prob = (
        0.8 * (df['sex'] == 'female').astype(int) +
        0.3 * (df['pclass'] == 1).astype(int) +
        0.2 * (df['pclass'] == 2).astype(int) +
        0.1 * (df['age'] < 18).astype(int) +
        0.2 * (df['fare'] > df['fare'].quantile(0.7)).astype(int)
    ) / 3.0
    
    df['survived'] = np.random.binomial(1, survival_prob.clip(0, 1))
    
    # Add some missing values
    missing_age = np.random.choice(df.index, size=int(n_samples * 0.2), replace=False)
    df.loc[missing_age, 'age'] = np.nan
    
    missing_embarked = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_embarked, 'embarked'] = np.nan
    
    # Save to CSV
    os.makedirs('examples/datasets', exist_ok=True)
    df.to_csv('examples/datasets/titanic.csv', index=False)
    
    print("✅ Titanic dataset created successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Target distribution: {df['survived'].value_counts().to_dict()}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df

if __name__ == "__main__":
    create_titanic_dataset()
