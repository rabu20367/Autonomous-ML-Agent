"""
Command-line interface for the Autonomous ML Agent
"""

import click
import pandas as pd
import json
from pathlib import Path
from typing import Optional

from .core import AutonomousMLAgent
from .config import Config

@click.group()
def main():
    """Autonomous ML Agent - Strategic ML automation with LLM-driven planning"""
    pass

@main.command()
@click.option('--data-path', '-d', required=True, help='Path to the dataset file')
@click.option('--target-column', '-t', required=True, help='Name of the target column')
@click.option('--test-size', default=0.2, help='Test set size (default: 0.2)')
@click.option('--max-experiments', default=10, help='Maximum number of experiments (default: 10)')
@click.option('--strategy', default='adaptive', type=click.Choice(['adaptive', 'exploratory', 'conservative']), 
              help='Strategy for experiment planning (default: adaptive)')
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--llm-provider', default='openai', type=click.Choice(['openai', 'anthropic']), 
              help='LLM provider (default: openai)')
@click.option('--llm-model', help='LLM model name')
@click.option('--llm-api-key', help='LLM API key')
def train(data_path: str, target_column: str, test_size: float, max_experiments: int,
          strategy: str, output_dir: Optional[str], llm_provider: str, 
          llm_model: Optional[str], llm_api_key: Optional[str]):
    """Train models on a dataset using autonomous ML agent"""
    
    # Setup configuration
    config = Config()
    if llm_provider:
        config.LLM.provider = llm_provider
    if llm_model:
        config.LLM.model = llm_model
    if llm_api_key:
        config.LLM.api_key = llm_api_key
    if output_dir:
        config.RESULTS_DIR = Path(output_dir)
        config.MODELS_DIR = Path(output_dir) / 'models'
    
    # Initialize agent
    agent = AutonomousMLAgent(config, strategy=strategy)
    
    try:
        # Run pipeline
        results = agent.run_pipeline(data_path, target_column, test_size, max_experiments)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("🎯 PIPELINE RESULTS")
        click.echo("="*60)
        
        # Show leaderboard
        click.echo("\n📊 MODEL LEADERBOARD:")
        click.echo("-" * 40)
        for i, model in enumerate(results['leaderboard'], 1):
            click.echo(f"{i}. {model['model']}: {model['test_accuracy']:.4f} accuracy")
        
        # Show best model
        if results['best_model']['result']:
            best_model = results['best_model']
            click.echo(f"\n🏆 BEST MODEL: {best_model['result']['plan']['model_selection']['primary_model']}")
            click.echo(f"   Test Accuracy: {best_model['performance']['test_accuracy']:.4f}")
            click.echo(f"   Test F1-Score: {best_model['performance']['test_f1']:.4f}")
        
        # Show insights
        if 'insights' in results:
            click.echo(f"\n💡 KEY INSIGHTS:")
            click.echo("-" * 40)
            for insight in results['insights']['recommendations']:
                click.echo(f"   • {insight}")
        
        # Show ensemble strategy
        if results['ensemble_strategy']['strategy'] != 'no_ensemble':
            click.echo(f"\n🎯 ENSEMBLE STRATEGY: {results['ensemble_strategy']['strategy']}")
        
        click.echo(f"\n📁 Results saved to: {config.RESULTS_DIR}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()

@main.command()
@click.option('--results-dir', '-r', required=True, help='Results directory from training')
def deploy(results_dir: str):
    """Generate deployment package for the best model"""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        click.echo(f"❌ Results directory not found: {results_dir}", err=True)
        raise click.Abort()
    
    # Load results
    results_files = list(results_path.glob("results_*.json"))
    if not results_files:
        click.echo(f"❌ No results files found in {results_dir}", err=True)
        raise click.Abort()
    
    # Use the most recent results file
    latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_results, 'r') as f:
        results = json.load(f)
    
    if not results.get('best_model', {}).get('result'):
        click.echo("❌ No trained model found in results", err=True)
        raise click.Abort()
    
    # Generate deployment package
    config = Config()
    config.RESULTS_DIR = results_path
    
    agent = AutonomousMLAgent(config)
    agent.results = results
    
    try:
        deployment_path = agent.generate_deployment_package()
        click.echo(f"✅ Deployment package generated: {deployment_path}")
        
    except Exception as e:
        click.echo(f"❌ Error generating deployment package: {str(e)}", err=True)
        raise click.Abort()

@main.command()
@click.option('--data-path', '-d', required=True, help='Path to the dataset file')
@click.option('--target-column', '-t', required=True, help='Name of the target column')
def analyze(data_path: str, target_column: str):
    """Analyze dataset characteristics without training"""
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        click.echo(f"📊 Dataset loaded: {df.shape}")
        
        # Analyze data
        from .data_pipeline import DataPipeline
        pipeline = DataPipeline(target_column)
        analysis = pipeline.analyze_data(df)
        
        # Display analysis
        click.echo("\n🔍 DATASET ANALYSIS:")
        click.echo("-" * 40)
        click.echo(f"Shape: {analysis['shape']}")
        click.echo(f"Target type: {analysis['target_type']}")
        click.echo(f"Complexity score: {analysis['complexity_score']:.3f}")
        click.echo(f"Domain hints: {', '.join(analysis['domain_hints'])}")
        
        click.echo(f"\n📈 Target distribution:")
        for class_name, pct in analysis['target_distribution'].items():
            click.echo(f"   {class_name}: {pct:.3f}")
        
        click.echo(f"\n🔧 Feature types:")
        for feature, ftype in analysis['feature_types'].items():
            click.echo(f"   {feature}: {ftype}")
        
        click.echo(f"\n❓ Missing values:")
        for feature, pct in analysis['missing_patterns'].items():
            if pct > 0:
                click.echo(f"   {feature}: {pct:.3f}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()

@main.command()
def insights():
    """Show global insights from experience memory"""
    
    try:
        from .experience_memory import ExperienceMemory
        memory = ExperienceMemory()
        insights = memory.get_global_insights()
        
        click.echo("🧠 GLOBAL INSIGHTS FROM EXPERIENCE MEMORY:")
        click.echo("-" * 50)
        click.echo(f"Total experiments: {insights['total_experiments']}")
        click.echo(f"Average performance: {insights['average_performance']:.4f}")
        click.echo(f"Unique datasets: {insights['unique_datasets']}")
        
        if insights['top_models']:
            click.echo(f"\n🏆 TOP PERFORMING MODELS:")
            for model in insights['top_models'][:5]:
                click.echo(f"   {model['model']}: {model['avg_score']:.4f} (used {model['count']} times)")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    main()
