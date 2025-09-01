"""
Generate all example datasets for testing the Autonomous ML Agent
"""

import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iris_dataset import create_iris_dataset
from titanic_dataset import create_titanic_dataset
from housing_dataset import create_housing_dataset

def generate_all_datasets():
    """Generate all example datasets"""
    
    print("🚀 Generating example datasets for Autonomous ML Agent...")
    print("=" * 60)
    
    # Create datasets directory
    os.makedirs('examples/datasets', exist_ok=True)
    
    # Generate Iris dataset
    print("\n📊 Generating Iris dataset...")
    iris_df = create_iris_dataset()
    
    # Generate Titanic dataset
    print("\n🚢 Generating Titanic dataset...")
    titanic_df = create_titanic_dataset()
    
    # Generate Housing dataset
    print("\n🏠 Generating Housing dataset...")
    housing_df = create_housing_dataset()
    
    print("\n" + "=" * 60)
    print("✅ All datasets generated successfully!")
    print("\n📁 Generated files:")
    print("   - examples/datasets/iris.csv")
    print("   - examples/datasets/titanic.csv")
    print("   - examples/datasets/housing.csv")
    
    print("\n🎯 Dataset Summary:")
    print(f"   Iris: {iris_df.shape[0]} samples, {iris_df.shape[1]} features (Classification)")
    print(f"   Titanic: {titanic_df.shape[0]} samples, {titanic_df.shape[1]} features (Classification)")
    print(f"   Housing: {housing_df.shape[0]} samples, {housing_df.shape[1]} features (Regression)")
    
    print("\n💡 Usage:")
    print("   You can now use these datasets to test the Autonomous ML Agent!")
    print("   Upload any of these CSV files through the web interface.")

if __name__ == "__main__":
    generate_all_datasets()
