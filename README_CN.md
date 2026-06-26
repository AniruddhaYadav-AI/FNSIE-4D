# 大模型推理评测引擎 🚀

一个功能完整的大模型API评测工具，支持OpenAI、Claude等多种模型，用于评测分类任务的性能。

## ✨ 特性

- ✅ **多模型支持**: OpenAI (GPT系列)、Claude、自定义API、模拟模式
- ✅ **8种任务类型**: 主题分类、情感分析、实体识别、领域分类、意图识别等
- ✅ **完整评测流程**: 数据加载 → 模型推理 → 结果分析 → 报告生成
- ✅ **详细分析**: 总体准确率、分任务准确率、错误案例分析
- ✅ **易于扩展**: 模块化设计，可轻松添加新模型和新任务
- ✅ **Windows友好**: 已解决中文和emoji显示问题

## 🎯 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行评测

```bash
# 使用模拟模式（无需API密钥）
python main.py --adapter mock

# 使用OpenAI（需要API密钥）
python main.py --adapter openai --model gpt-3.5-turbo --api-key your-key

# 使用Claude（需要API密钥）
python main.py --adapter claude --model claude-3-sonnet-20240229
```

### 3. 查看结果

评测完成后：
- 控制台显示准确率和错误分析
- 结果保存到 `evaluation_results.json`

## 📂 项目结构

```
ToyInferenceEngine/
├── main.py                  # 主程序入口 ⭐
├── config.yaml              # 配置文件
├── dataset.json             # 评测数据集（20个样本）
│
├── data_loader.py           # 数据加载器
├── model_adapters.py        # 模型API适配器
├── inference_engine.py      # 推理引擎核心
├── metrics.py               # 评测指标计算
│
├── test_engine.py           # 测试脚本
├── compare_models.py        # 模型对比脚本 ⭐
│
├── requirements.txt         # 依赖列表
├── README.md                # 英文文档
├── README_CN.md             # 中文文档（本文件）
└── 使用指南.md             # 详细使用说明 ⭐⭐⭐
```

## 📊 评测数据集

当前包含 **20个样本**，涵盖 **8种任务**：

| 任务类型 | 数量 | 示例 |
|---------|------|------|
| 主题分类 | 3 | "股市收盘上涨2%" → finance |
| 情感分析 | 3 | "我非常喜欢这款手机" → positive |
| 实体类型 | 3 | "巴黎是法国的首都" → location |
| 领域分类 | 3 | "患者被诊断为肺炎" → medical |
| Yes/No | 2 | "文本描述了数学公式" → yes |
| 意图识别 | 3 | "明天天气如何" → information_request |
| 极性检测 | 2 | "外面在下雨吗？" → question |
| 鲁棒性 | 1 | "Wat is 2 + 2?" → 4 |

## 🔧 使用示例

### 基础用法

```bash
# 查看帮助
python main.py --help

# 显示详细信息
python main.py --adapter mock --verbose

# 只评测前5条
python main.py --subset 1,2,3,4,5

# 只评测情感分析
python main.py --task-type sentiment_analysis
```

### 对比多个模型

编辑 `compare_models.py` 配置要对比的模型，然后运行：

```bash
python compare_models.py
```

自动生成对比表格和最佳模型推荐。

### Python编程方式

```python
from data_loader import DataLoader
from model_adapters import create_adapter
from inference_engine import InferenceEngine

# 加载数据
loader = DataLoader("dataset.json")
loader.load_data()

# 创建模型
adapter = create_adapter("openai", "gpt-3.5-turbo", api_key="your-key")

# 运行评测
engine = InferenceEngine(adapter, loader)
results = engine.run_evaluation()

print(f"准确率: {results['summary']['总体准确率']}")
```

## 🎨 示例输出

```
🎯 大模型推理评测引擎
============================================================

📂 正在加载数据集...
✓ 成功加载 20 条评测数据

数据集统计:
  - topic_classification: 3 条
  - sentiment_analysis: 3 条
  - entity_type: 3 条
  ...

🤖 正在初始化模型适配器...
适配器类型: openai
模型名称: gpt-3.5-turbo
✓ 模型适配器初始化成功

🚀 开始评测，共 20 个数据项
评测进度: 100%|████████████████████| 20/20 [00:15<00:00]

============================================================
📊 评测结果总结
============================================================
总数据量: 20
正确数: 18
错误数: 2
总体准确率: 90.00%

各任务类型准确率:
  - topic_classification: 100.00%
  - sentiment_analysis: 100.00%
  - entity_type: 66.67%
  ...
============================================================

💾 评测结果已保存到: evaluation_results.json
✅ 评测完成！
```

## 🛠️ 高级功能

### 1. 添加自定义模型

编辑 `model_adapters.py`，继承 `BaseModelAdapter`：

```python
class MyModelAdapter(BaseModelAdapter):
    def generate(self, prompt: str, temperature: float = 0.0, 
                 max_tokens: int = 500) -> str:
        # 实现你的API调用
        return response
```

### 2. 添加自定义数据

编辑 `dataset.json`，添加新数据项：

```json
{
  "id": 21,
  "task_type": "your_task_type",
  "input": "your input text",
  "classes": ["class1", "class2"],
  "gold_label": {"label": "class1"}
}
```

### 3. 修改提示词

编辑 `inference_engine.py` 中的 `build_prompt` 方法。

## 📖 详细文档

- **快速上手**: 见本文件
- **详细使用**: 见 `使用指南.md` ⭐⭐⭐
- **英文文档**: 见 `README.md`
- **代码注释**: 见各个 `.py` 文件

## ⚙️ 配置说明

编辑 `config.yaml` 进行配置：

```yaml
# 数据配置
data:
  dataset_path: "dataset.json"

# 模型配置
model:
  adapter_type: "mock"  # openai/claude/custom/mock
  model_name: "gpt-3.5-turbo"

# 评测配置
evaluation:
  verbose: false
  sleep_time: 0.5  # API调用间隔
  temperature: 0.0
  max_tokens: 500

# 输出配置
output:
  results_path: "evaluation_results.json"
  save_results: true
```

## 💡 提示

1. **节省API费用**: 先用 `--subset` 测试少量数据
2. **避免速率限制**: 调整 `config.yaml` 中的 `sleep_time`
3. **调试问题**: 使用 `--verbose` 查看详细日志
4. **对比模型**: 使用 `compare_models.py` 自动对比

## ❓ 常见问题

**Q: 没有API密钥？**  
A: 使用模拟模式：`python main.py --adapter mock`

**Q: 如何添加新数据？**  
A: 编辑 `dataset.json` 文件

**Q: 如何修改评测逻辑？**  
A: 编辑 `metrics.py` 中的 `compare_labels` 函数

**Q: Windows显示乱码？**  
A: 已在代码中处理UTF-8编码，应该不会有问题

## 🧪 运行测试

```bash
# 运行基础测试
python test_engine.py

# 运行完整评测（模拟模式）
python main.py --adapter mock

# 运行模型对比
python compare_models.py
```

## 📝 开发计划

- [ ] 添加批量API调用支持
- [ ] 添加更多评测指标（F1、Precision、Recall等）
- [ ] 添加可视化图表生成
- [ ] 支持更多模型API
- [ ] 添加Web界面

## 📄 许可证

MIT License

## 🙏 致谢

感谢使用本工具进行大模型评测！

---

**如有问题，请查看 `使用指南.md` 获取更详细的说明。** 📖

**祝评测顺利！** 🎉

