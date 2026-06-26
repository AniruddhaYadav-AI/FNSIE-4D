# LLM Inference Evaluation Engine

A simple yet powerful evaluation tool for Large Language Model APIs, supporting OpenAI, Claude, custom APIs, and mock mode for testing.

## ✨ Features

- ✅ **Multiple Model Support**: OpenAI (GPT series), Claude, custom APIs, mock mode
- ✅ **100 Evaluation Items**: Comprehensive dataset covering 8 task types
- ✅ **Complete Evaluation Pipeline**: Data loading → Model inference → Result analysis → Report generation
- ✅ **Detailed Analytics**: Overall accuracy, task-wise accuracy, error case analysis
- ✅ **Easy to Extend**: Modular design for adding new models and tasks
- ✅ **Cross-platform**: Works on Windows, Linux, and macOS

## 📊 Dataset

The evaluation dataset contains **100 items** across **8 task types**:

| Task Type | Count | Description |
|-----------|-------|-------------|
| Topic Classification | 18 | Sports, Finance, Politics |
| Sentiment Analysis | 15 | Positive, Neutral, Negative |
| Entity Type Recognition | 13 | Person, Organization, Location |
| Domain Classification | 13 | Medical, Legal, Technical |
| Yes/No Questions | 10 | Binary classification |
| Intent Classification | 10 | Information Request, Command, Chitchat |
| Polarity Detection | 10 | Question vs Statement |
| Robustness Check | 11 | Math problems and typo handling |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Evaluation

```bash
# Using mock mode (no API key needed)
python main.py --adapter mock

# Using OpenAI
python main.py --adapter openai --model gpt-3.5-turbo --api-key your-key

# Using Claude
set ANTHROPIC_API_KEY=your-key
python main.py --adapter claude --model claude-3-sonnet-20240229
```

### 3. View Results

After evaluation completes:
- Console displays accuracy and error analysis
- Results saved to `evaluation_results.json`

## 📂 Project Structure

```
ToyInferenceEngine/
├── Core Modules
│   ├── main.py              - Main entry point ⭐
│   ├── data_loader.py       - Data loader
│   ├── model_adapters.py    - Model API adapters
│   ├── inference_engine.py  - Inference engine core
│   └── metrics.py           - Evaluation metrics
│
├── Data & Config
│   ├── dataset.json         - Evaluation dataset (100 items)
│   └── config.yaml          - Configuration file
│
├── Tools
│   ├── test_engine.py       - Test script
│   └── compare_models.py    - Model comparison tool
│
├── Documentation
│   ├── README.md            - English documentation (this file)
│   ├── README_CN.md         - Chinese documentation
│   ├── 使用指南.md         - Detailed user guide (Chinese)
│   └── 快速开始.md         - Quick start guide (Chinese)
│
└── requirements.txt         - Dependencies
```

## 💡 Usage Examples

### Basic Usage

```bash
# View help
python main.py --help

# Show detailed evaluation process
python main.py --adapter mock --verbose

# Evaluate first 5 items only
python main.py --subset E-0001,E-0002,E-0003,E-0004,E-0005

# Evaluate sentiment analysis tasks only
python main.py --task-type sentiment_analysis
```

### Compare Multiple Models

Edit `compare_models.py` to configure models, then run:

```bash
python compare_models.py
```

Automatically generates comparison table and best model recommendation.

### Python API Usage

```python
from data_loader import DataLoader
from model_adapters import create_adapter
from inference_engine import InferenceEngine

# Load data
loader = DataLoader("dataset.json")
loader.load_data()

# Create model adapter
adapter = create_adapter("openai", "gpt-3.5-turbo", api_key="your-key")

# Run evaluation
engine = InferenceEngine(adapter, loader)
results = engine.run_evaluation()

print(f"Accuracy: {results['summary']['overall_accuracy']}")
```

## 🎨 Example Output

```
🎯 LLM Inference Evaluation Engine
============================================================

📂 Loading dataset...
✓ Successfully loaded 100 evaluation items

Dataset statistics:
  - topic_classification: 18 items
  - sentiment_analysis: 15 items
  - entity_type: 13 items
  ...

🤖 Initializing model adapter...
Adapter type: openai
Model name: gpt-3.5-turbo
✓ Model adapter initialized successfully

🚀 Starting evaluation on 100 items
Evaluation Progress: 100%|████████████| 100/100 [00:50<00:00]

============================================================
📊 Evaluation Summary
============================================================
Total Items: 100
Correct: 92
Wrong: 8
Overall Accuracy: 92.00%

Task-wise Accuracy:
  - topic_classification: 100.00%
  - sentiment_analysis: 93.33%
  - entity_type: 92.31%
  ...
============================================================

💾 Evaluation results saved to: evaluation_results.json
✅ Evaluation completed!
```

## 🛠️ Advanced Features

### 1. Add Custom Model

Edit `model_adapters.py`, extend `BaseModelAdapter`:

```python
class MyModelAdapter(BaseModelAdapter):
    def generate(self, prompt: str, temperature: float = 0.0, 
                 max_tokens: int = 500) -> str:
        # Implement your API call
        return response
```

### 2. Add Custom Data

Edit `dataset.json`, add new items:

```json
{
  "id": "E-0101",
  "task_type": "your_task_type",
  "input": "your input text",
  "classes": ["class1", "class2"],
  "gold_label": {"label": "class1"}
}
```

### 3. Modify Prompts

Edit the `build_prompt` method in `inference_engine.py`.

## ⚙️ Configuration

Edit `config.yaml` for configuration:

```yaml
# Data configuration
data:
  dataset_path: "dataset.json"

# Model configuration
model:
  adapter_type: "mock"  # openai/claude/custom/mock
  model_name: "gpt-3.5-turbo"

# Evaluation configuration
evaluation:
  verbose: false
  sleep_time: 0.5  # API call interval
  temperature: 0.0
  max_tokens: 500

# Output configuration
output:
  results_path: "evaluation_results.json"
  save_results: true
```

## 💡 Tips

1. **Save API Costs**: Test with `--subset` first
2. **Avoid Rate Limits**: Adjust `sleep_time` in `config.yaml`
3. **Debug Issues**: Use `--verbose` for detailed logs
4. **Compare Models**: Use `compare_models.py` for automatic comparison

## ❓ FAQ

**Q: Don't have API key?**  
A: Use mock mode: `python main.py --adapter mock`

**Q: How to add custom data?**  
A: Edit `dataset.json` file

**Q: How to modify evaluation logic?**  
A: Edit `compare_labels` function in `metrics.py`

**Q: Garbled characters on Windows?**  
A: Already handled with UTF-8 encoding in code

## 🧪 Run Tests

```bash
# Run basic tests
python test_engine.py

# Run full evaluation (mock mode)
python main.py --adapter mock

# Run model comparison
python compare_models.py
```

## 📄 License

MIT License

## 🙏 Acknowledgments

Thank you for using this tool for LLM evaluation!

---

**For detailed documentation, see `README_CN.md` (Chinese) or `使用指南.md` (Detailed User Guide).** 📖

**Happy Evaluating!** 🎉
