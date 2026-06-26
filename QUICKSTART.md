# Quick Start Guide ⚡

Get started with the LLM Inference Evaluation Engine in 3 simple steps.

## Step 1: Install Dependencies 📦

```bash
pip install -r requirements.txt
```

## Step 2: Run Test 🧪

```bash
python test_engine.py
```

If you see "✅ All tests passed!", the system is working properly.

## Step 3: Start Evaluation 🚀

### Option A: No API Key Needed (Mock Mode)

```bash
python main.py --adapter mock
```

### Option B: Using OpenAI API

```bash
python main.py --adapter openai --model gpt-3.5-turbo --api-key your-api-key
```

### Option C: Using Claude API

```bash
set ANTHROPIC_API_KEY=your-api-key
python main.py --adapter claude --model claude-3-sonnet-20240229
```

## Common Commands 📋

```bash
# View detailed evaluation process
python main.py --adapter mock --verbose

# Evaluate first 5 items only
python main.py --adapter mock --subset E-0001,E-0002,E-0003,E-0004,E-0005

# Evaluate sentiment analysis tasks only
python main.py --adapter mock --task-type sentiment_analysis

# Compare multiple models (edit compare_models.py first)
python compare_models.py

# View help
python main.py --help
```

## File Overview 📂

| File | Description | Importance |
|------|-------------|------------|
| `main.py` | Main program, run evaluation | ⭐⭐⭐ |
| `dataset.json` | Evaluation data (100 items) | ⭐⭐⭐ |
| `config.yaml` | Configuration file | ⭐⭐ |
| `compare_models.py` | Model comparison script | ⭐⭐ |
| `test_engine.py` | Test script | ⭐ |
| `README.md` | Project documentation (English) | ⭐⭐ |
| `README_CN.md` | Project documentation (Chinese) | ⭐⭐ |

## Evaluation Results 📊

After evaluation completes, you will see:

```
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
  ...
============================================================
```

Results are saved to `evaluation_results.json`

## Customize Dataset ✏️

Edit `dataset.json` to add your own evaluation data:

```json
{
  "id": "E-0101",
  "task_type": "sentiment_analysis",
  "input": "This product is amazing!",
  "classes": ["positive", "neutral", "negative"],
  "gold_label": {"label": "positive"}
}
```

## Next Steps 🎯

1. ✅ Start with mock mode to familiarize yourself
2. ✅ Add your own evaluation data
3. ✅ Use real APIs for evaluation
4. ✅ Compare different model performances
5. ✅ Analyze error cases and optimize prompts

## Need Help? ❓

- Basic usage: See this file
- Detailed guide: See `README.md` or `README_CN.md`
- Technical details: Check code comments
- Troubleshooting: Run `python main.py --help`

---

**Start your first evaluation now!** 🎉

```bash
python main.py --adapter mock
```

