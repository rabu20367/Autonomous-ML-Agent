"""
Basic usage example for the Autonomous ML Agent
"""

import pandas as pd
import numpy as np
from autonomous_ml import AutonomousMLAgent
from autonomous_ml.config import Config

def create_sample_data():
    """Create sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    # Add some correlation to make it more realistic
    data['target'] = (
        (data['credit_score'] > 700).astype(int) * 0.3 +
        (data['debt_ratio'] < 0.3).astype(int) * 0.2 +
        (data['income'] > 60000).astype(int) * 0.2 +
        np.random.random(n_samples) * 0.3
    ).round().astype(int)
    
    return pd.DataFrame(data)

def main():
    """Main example function"""
    print("🚀 Autonomous ML Agent - Basic Usage Example")
    print("=" * 60)
    
    # Create sample data
    print("\n📊 Creating sample dataset...")
    df = create_sample_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Target distribution: {df['target'].value_counts(normalize=True).round(3).to_dict()}")
    
    # Save sample data
    df.to_csv('sample_data.csv', index=False)
    print("   Sample data saved to: sample_data.csv")
    
    # Initialize configuration
    print("\n⚙️ Initializing configuration...")
    config = Config()
    
    # Set up LLM configuration (you'll need to set your API key)
    config.LLM.provider = 'openai'
    config.LLM.model = 'gpt-4'
    config.LLM.api_key = 'your_api_key_here'  # Replace with your actual API key
    
    # Initialize agent
    print("\n🤖 Initializing Autonomous ML Agent...")
    agent = AutonomousMLAgent(config, strategy='adaptive')
    
    try:
        # Run the complete pipeline
        print("\n🧪 Running autonomous ML pipeline...")
        results = agent.run_pipeline(
            data_path='sample_data.csv',
            target_column='target',
            test_size=0.2,
            max_experiments=5  # Reduced for demo
        )
        
        # Display results
        print("\n📋 Results Summary:")
        print("-" * 40)
        
        if results['leaderboard']:
            print("🏆 Top 3 Models:")
            for i, model in enumerate(results['leaderboard'][:3], 1):
                print(f"   {i}. {model['model']}: {model['test_accuracy']:.4f} accuracy")
        
        if results['best_model']['result']:
            best = results['best_model']
            print(f"\n🎯 Best Model: {best['result']['plan']['model_selection']['primary_model']}")
            print(f"   Test Accuracy: {best['performance']['test_accuracy']:.4f}")
            print(f"   Test F1-Score: {best['performance']['test_f1']:.4f}")
        
        # Show insights
        if 'insights' in results:
            print(f"\n💡 Key Insights:")
            for insight in results['insights']['recommendations']:
                print(f"   • {insight}")
        
        # Generate deployment package
        print(f"\n🚀 Generating deployment package...")
        try:
            deployment_path = agent.generate_deployment_package()
            print(f"   ✅ Deployment package created: {deployment_path}")
        except Exception as e:
            print(f"   ⚠️ Could not generate deployment package: {str(e)}")
        
        print(f"\n✅ Example completed successfully!")
        print(f"   Results saved to: {config.RESULTS_DIR}")
        
    except Exception as e:
        print(f"❌ Error running pipeline: {str(e)}")
        print("   Make sure to set your LLM API key in the configuration")

if __name__ == "__main__":
    main()
