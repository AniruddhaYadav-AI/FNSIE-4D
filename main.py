"""
LLM Inference Evaluation Engine - Main Entry Point
"""
import os
import sys
import io
import argparse
import yaml
from dotenv import load_dotenv

# Set stdout to UTF-8 encoding (Windows compatibility)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_loader import DataLoader
from model_adapters import create_adapter
from inference_engine import InferenceEngine


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM Inference Evaluation Engine")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--adapter", type=str, help="Adapter type (openai/claude/custom/mock)")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--dataset", type=str, help="Dataset path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--output", type=str, help="Results output path")
    parser.add_argument("--subset", type=str, help="Only evaluate specified IDs (comma-separated)")
    parser.add_argument("--task-type", type=str, help="Only evaluate specified task type")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"⚠️ Config file not found: {args.config}, using defaults")
        config = {
            "data": {"dataset_path": "dataset.json"},
            "model": {"adapter_type": "mock", "model_name": "mock-model"},
            "evaluation": {"verbose": False, "sleep_time": 0.5},
            "output": {"results_path": "evaluation_results.json", "save_results": True}
        }
    
    # Command line arguments have higher priority than config file
    adapter_type = args.adapter or config['model']['adapter_type']
    model_name = args.model or config['model'].get('model_name')
    api_key = args.api_key or config['model'].get('api_key') or None
    dataset_path = args.dataset or config['data']['dataset_path']
    verbose = args.verbose or config['evaluation']['verbose']
    output_path = args.output or config['output']['results_path']
    
    print("🎯 LLM Inference Evaluation Engine")
    print("="*60)
    
    # 1. Load data
    print("\n📂 Loading dataset...")
    try:
        data_loader = DataLoader(dataset_path)
        data_loader.load_data()
        
        # Print dataset statistics
        stats = data_loader.get_statistics()
        print(f"\nDataset statistics:")
        for task_type, count in stats.items():
            print(f"  - {task_type}: {count} items")
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found - {dataset_path}")
        sys.exit(1)
    
    # 2. Create model adapter
    print(f"\n🤖 Initializing model adapter...")
    print(f"Adapter type: {adapter_type}")
    print(f"Model name: {model_name}")
    
    try:
        # Get additional parameters
        adapter_kwargs = {}
        if adapter_type == "custom":
            adapter_kwargs["api_url"] = config['model'].get('api_url')
            adapter_kwargs["request_format"] = config['model'].get('request_format', 'openai')
        
        model_adapter = create_adapter(
            adapter_type=adapter_type,
            model_name=model_name,
            api_key=api_key,
            **adapter_kwargs
        )
        print("✓ Model adapter initialized successfully")
    except Exception as e:
        print(f"❌ Error: Model adapter initialization failed - {str(e)}")
        sys.exit(1)
    
    # 3. Create inference engine
    print("\n⚙️ Initializing inference engine...")
    engine = InferenceEngine(model_adapter, data_loader)
    print("✓ Inference engine initialized successfully")
    
    # 4. Run evaluation
    try:
        if args.subset:
            # Subset evaluation
            item_ids = [x.strip() for x in args.subset.split(",")]
            results = engine.run_subset(item_ids=item_ids, verbose=verbose)
        elif args.task_type:
            # Evaluation by task type
            results = engine.run_subset(task_type=args.task_type, verbose=verbose)
        else:
            # Full evaluation
            results = engine.run_evaluation(
                verbose=verbose,
                sleep_time=config['evaluation']['sleep_time']
            )
        
        # 5. Save results
        if config['output']['save_results']:
            engine.save_results(output_path)
        
        print("\n✅ Evaluation completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
