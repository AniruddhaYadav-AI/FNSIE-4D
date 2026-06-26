"""
Model Comparison Script - Compare multiple models on the same dataset

Usage:
1. Modify the model configuration below
2. Run: python compare_models.py
"""
import sys
import io
import json
import pandas as pd
from typing import List, Dict

# Set stdout to UTF-8 encoding (Windows compatibility)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_loader import DataLoader
from model_adapters import create_adapter
from inference_engine import InferenceEngine


# Configure models to compare
MODELS_TO_COMPARE = [
    {
        "name": "Mock Model",
        "adapter_type": "mock",
        "model_name": "mock-model"
    },
    # Uncomment to test real models
    # {
    #     "name": "GPT-3.5-Turbo",
    #     "adapter_type": "openai",
    #     "model_name": "gpt-3.5-turbo",
    #     "api_key": "your-openai-api-key"
    # },
    # {
    #     "name": "GPT-4",
    #     "adapter_type": "openai",
    #     "model_name": "gpt-4",
    #     "api_key": "your-openai-api-key"
    # },
    # {
    #     "name": "Claude-3-Sonnet",
    #     "adapter_type": "claude",
    #     "model_name": "claude-3-sonnet-20240229",
    #     "api_key": "your-anthropic-api-key"
    # },
]


def run_comparison(models: List[Dict], dataset_path: str = "dataset.json"):
    """
    Run model comparison evaluation
    
    Args:
        models: List of model configurations
        dataset_path: Dataset path
    """
    print("="*70)
    print("🏆 Multi-Model Comparison Evaluation")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading dataset: {dataset_path}")
    loader = DataLoader(dataset_path)
    loader.load_data()
    
    # Store all model results
    all_results = []
    
    # Evaluate each model
    for i, model_config in enumerate(models, 1):
        print(f"\n{'='*70}")
        print(f"📊 [{i}/{len(models)}] Evaluating model: {model_config['name']}")
        print(f"{'='*70}")
        
        try:
            # Create adapter
            adapter = create_adapter(
                adapter_type=model_config['adapter_type'],
                model_name=model_config['model_name'],
                api_key=model_config.get('api_key')
            )
            
            # Create inference engine
            engine = InferenceEngine(adapter, loader)
            
            # Run evaluation
            results = engine.run_evaluation(verbose=False, sleep_time=0.5)
            
            # Save results
            output_file = f"results_{model_config['adapter_type']}_{model_config['model_name'].replace('/', '_')}.json"
            engine.save_results(output_file)
            
            # Extract key metrics
            summary = results['summary']
            model_stats = results['model_stats']
            
            all_results.append({
                "Model Name": model_config['name'],
                "Adapter": model_config['adapter_type'],
                "Model ID": model_config['model_name'],
                "Total Items": summary['total_items'],
                "Correct": summary['correct'],
                "Wrong": summary['wrong'],
                "Accuracy": summary['overall_accuracy'],
                "Requests": model_stats['request_count'],
                "Total Tokens": model_stats.get('total_tokens', 'N/A'),
                "Result File": output_file
            })
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            all_results.append({
                "Model Name": model_config['name'],
                "Adapter": model_config['adapter_type'],
                "Model ID": model_config['model_name'],
                "Accuracy": "Evaluation failed",
                "Error": str(e)
            })
    
    # Print comparison table
    print("\n" + "="*70)
    print("📈 Evaluation Results Comparison")
    print("="*70 + "\n")
    
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))
    
    # Save comparison results
    comparison_file = "model_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Comparison results saved to: {comparison_file}")
    
    # Find best model
    try:
        valid_results = [r for r in all_results if isinstance(r.get('Correct'), int)]
        if valid_results:
            best_model = max(valid_results, key=lambda x: x['Correct'])
            print(f"\n🏆 Best Model: {best_model['Model Name']} - Accuracy: {best_model['Accuracy']}")
    except:
        pass
    
    print("\n" + "="*70)
    print("✅ Comparison evaluation completed!")
    print("="*70)


def main():
    """Main function"""
    if len(MODELS_TO_COMPARE) == 0:
        print("⚠️ Warning: No models configured for comparison")
        print("Please edit compare_models.py and add model configurations to MODELS_TO_COMPARE list")
        return
    
    print("Preparing to compare the following models:")
    for i, model in enumerate(MODELS_TO_COMPARE, 1):
        print(f"  {i}. {model['name']} ({model['model_name']})")
    
    # Confirm start
    print("\nEvaluation will start soon...")
    
    try:
        run_comparison(MODELS_TO_COMPARE)
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
