"""
Simple Test Script - Verify inference engine functionality
"""
import sys
import io

# Set stdout to UTF-8 encoding (Windows compatibility)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data_loader import DataLoader
from model_adapters import MockAdapter
from inference_engine import InferenceEngine


def test_basic_functionality():
    """Test basic functionality"""
    print("🧪 Testing LLM Inference Evaluation Engine\n")
    
    # 1. Test data loading
    print("1️⃣ Testing data loading...")
    loader = DataLoader("dataset.json")
    data = loader.load_data()
    assert len(data) == 100, "Data loading failed"
    print("✓ Data loading successful\n")
    
    # 2. Test data statistics
    print("2️⃣ Testing data statistics...")
    stats = loader.get_statistics()
    print(f"Dataset statistics: {stats}")
    assert len(stats) > 0, "Statistics empty"
    print("✓ Data statistics normal\n")
    
    # 3. Test model adapter
    print("3️⃣ Testing model adapter...")
    adapter = MockAdapter()
    response = adapter.generate("Test prompt")
    assert response is not None, "Adapter response empty"
    print(f"Mock response: {response}")
    print("✓ Model adapter normal\n")
    
    # 4. Test inference engine (first 5 items only)
    print("4️⃣ Testing inference engine (first 5 items)...")
    engine = InferenceEngine(adapter, loader)
    results = engine.run_subset(item_ids=["E-0001", "E-0002", "E-0003", "E-0004", "E-0005"], verbose=False)
    assert len(results['results']) == 5, "Inference result count incorrect"
    print("✓ Inference engine normal\n")
    
    # 5. Test result saving
    print("5️⃣ Testing result saving...")
    engine.save_results("test_results.json")
    import os
    assert os.path.exists("test_results.json"), "Result file not saved"
    print("✓ Result saving successful\n")
    
    print("="*60)
    print("✅ All tests passed! System is functioning properly.")
    print("="*60)
    print("\nTip: You can now run full evaluation with:")
    print("  python main.py --adapter mock  # Using mock mode")
    print("  python main.py --adapter openai --model gpt-3.5-turbo  # Using OpenAI")


if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
