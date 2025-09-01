"""
Advanced usage example showing custom configuration and strategies
"""

import pandas as pd
import numpy as np
from autonomous_ml import AutonomousMLAgent
from autonomous_ml.config import Config
from autonomous_ml.data_pipeline import DataPipeline

def create_complex_dataset():
    """Create a more complex dataset with various data types"""
    np.random.seed(42)
    n_samples = 2000
    
    # Generate complex features
    data = {
        # Numerical features
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples),
        
        # Categorical features
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
        
        # Datetime features
        'application_date': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        
        # High cardinality categorical
        'city': np.random.choice([f'City_{i}' for i in range(50)], n_samples),
        
        # Missing values
        'previous_loans': np.random.poisson(2, n_samples),
    }
    
    # Add missing values
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    data['income'][missing_indices] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    data['credit_score'][missing_indices] = np.nan
    
    # Create target with complex relationships
    target = (
        (data['credit_score'] > 700).astype(int) * 0.3 +
        (data['debt_ratio'] < 0.3).astype(int) * 0.2 +
        (data['income'] > np.percentile(data['income'], 70)).astype(int) * 0.2 +
        (data['education'].isin(['Master', 'PhD'])).astype(int) * 0.1 +
        np.random.random(n_samples) * 0.2
    ).round().astype(int)
    
    data['loan_approved'] = target
    
    return pd.DataFrame(data)

def compare_strategies():
    """Compare different strategic approaches"""
    print("🔬 Comparing Strategic Approaches")
    print("=" * 60)
    
    # Create complex dataset
    df = create_complex_dataset()
    df.to_csv('complex_data.csv', index=False)
    print(f"📊 Created complex dataset: {df.shape}")
    
    strategies = ['adaptive', 'exploratory', 'conservative']
    results_comparison = {}
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.upper()} strategy...")
        
        # Configure for this strategy
        config = Config()
        config.LLM.provider = 'openai'
        config.LLM.model = 'gpt-4'
        config.LLM.api_key = 'your_api_key_here'  # Replace with your actual API key
        
        # Adjust configuration based on strategy
        if strategy == 'exploratory':
            config.OPTIMIZATION['n_trials'] = 30  # More trials for exploration
        elif strategy == 'conservative':
            config.OPTIMIZATION['n_trials'] = 20  # Fewer trials, more reliable
        
        # Initialize agent
        agent = AutonomousMLAgent(config, strategy=strategy)
        
        try:
            # Run pipeline
            results = agent.run_pipeline(
                data_path='complex_data.csv',
                target_column='loan_approved',
                test_size=0.2,
                max_experiments=3  # Reduced for demo
            )
            
            # Store results
            if results['leaderboard']:
                best_accuracy = results['leaderboard'][0]['test_accuracy']
                results_comparison[strategy] = {
                    'best_accuracy': best_accuracy,
                    'best_model': results['leaderboard'][0]['model'],
                    'total_experiments': results['pipeline_metadata']['total_experiments'],
                    'successful_experiments': results['pipeline_metadata']['successful_experiments']
                }
                print(f"   ✅ Best accuracy: {best_accuracy:.4f} ({results['leaderboard'][0]['model']})")
            else:
                results_comparison[strategy] = {'best_accuracy': 0, 'error': 'No successful experiments'}
                print(f"   ❌ No successful experiments")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results_comparison[strategy] = {'best_accuracy': 0, 'error': str(e)}
    
    # Compare results
    print(f"\n📊 Strategy Comparison Results:")
    print("-" * 50)
    for strategy, result in results_comparison.items():
        if 'error' in result:
            print(f"{strategy.upper():12}: ERROR - {result['error']}")
        else:
            print(f"{strategy.upper():12}: {result['best_accuracy']:.4f} accuracy ({result['best_model']})")
    
    return results_comparison

def demonstrate_data_analysis():
    """Demonstrate advanced data analysis capabilities"""
    print("\n🔍 Advanced Data Analysis Demo")
    print("=" * 60)
    
    # Create dataset
    df = create_complex_dataset()
    
    # Initialize data pipeline
    pipeline = DataPipeline('loan_approved')
    
    # Analyze data
    analysis = pipeline.analyze_data(df)
    
    print("📈 Dataset Analysis Results:")
    print(f"   Shape: {analysis['shape']}")
    print(f"   Target type: {analysis['target_type']}")
    print(f"   Complexity score: {analysis['complexity_score']:.3f}")
    print(f"   Domain hints: {', '.join(analysis['domain_hints'])}")
    
    print(f"\n📊 Target Distribution:")
    for class_name, pct in analysis['target_distribution'].items():
        print(f"   Class {class_name}: {pct:.3f}")
    
    print(f"\n🔧 Feature Types:")
    feature_type_counts = {}
    for feature, ftype in analysis['feature_types'].items():
        feature_type_counts[ftype] = feature_type_counts.get(ftype, 0) + 1
    for ftype, count in feature_type_counts.items():
        print(f"   {ftype}: {count} features")
    
    print(f"\n❓ Missing Value Patterns:")
    for feature, pct in analysis['missing_patterns'].items():
        if pct > 0:
            print(f"   {feature}: {pct:.3f} missing")
    
    return analysis

def demonstrate_experience_memory():
    """Demonstrate experience memory capabilities"""
    print("\n🧠 Experience Memory Demo")
    print("=" * 60)
    
    from autonomous_ml.experience_memory import ExperienceMemory
    
    # Initialize memory
    memory = ExperienceMemory()
    
    # Create mock experiment data
    mock_experiments = [
        {
            'experiment_plan': {
                'model_selection': {'primary_model': 'logistic_regression'},
                'preprocessing_strategy': {'missing_handling': 'median'}
            },
            'results': {
                'success': True,
                'performance_metrics': {'test_accuracy': 0.85}
            }
        },
        {
            'experiment_plan': {
                'model_selection': {'primary_model': 'random_forest'},
                'preprocessing_strategy': {'missing_handling': 'knn'}
            },
            'results': {
                'success': True,
                'performance_metrics': {'test_accuracy': 0.82}
            }
        }
    ]
    
    # Store experiments
    for i, exp in enumerate(mock_experiments):
        exp['experiment_plan']['experiment_id'] = f'demo_exp_{i}'
        memory.store_experiment(exp['experiment_plan'], exp['results'])
    
    print("✅ Stored mock experiments in memory")
    
    # Get global insights
    insights = memory.get_global_insights()
    print(f"\n📊 Global Insights:")
    print(f"   Total experiments: {insights['total_experiments']}")
    print(f"   Average performance: {insights['average_performance']:.4f}")
    print(f"   Unique datasets: {insights['unique_datasets']}")
    
    if insights['top_models']:
        print(f"\n🏆 Top Models:")
        for model in insights['top_models'][:3]:
            print(f"   {model['model']}: {model['avg_score']:.4f} (used {model['count']} times)")

def main():
    """Main advanced example function"""
    print("🚀 Autonomous ML Agent - Advanced Usage Example")
    print("=" * 60)
    
    # Demonstrate data analysis
    analysis = demonstrate_data_analysis()
    
    # Demonstrate experience memory
    demonstrate_experience_memory()
    
    # Compare strategies (commented out to avoid API calls in demo)
    print(f"\n⚠️ Strategy comparison requires LLM API key")
    print(f"   Uncomment the compare_strategies() call to test different strategies")
    # results_comparison = compare_strategies()
    
    print(f"\n✅ Advanced example completed!")
    print(f"   This demonstrates the full capabilities of the Autonomous ML Agent")

if __name__ == "__main__":
    main()
