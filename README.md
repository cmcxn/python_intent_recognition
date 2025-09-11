# Chinese RoBERTa Intent Recognition for Office Domain

This module provides a complete implementation of Chinese intent recognition for office domain tasks using Chinese RoBERTa (hfl/chinese-roberta-wwm-ext).

## Supported Intents (支持的意图)

The model can identify the following Chinese office domain intents:

1. **CHECK_PAYSLIP** - 查询工资单相关问题
2. **BOOK_MEETING_ROOM** - 会议室预订请求
3. **REQUEST_LEAVE** - 请假申请
4. **CHECK_BENEFITS** - 福利查询
5. **IT_TICKET** - IT支持工单
6. **EXPENSE_REIMBURSE** - 费用报销

## Model Features

- **Chinese RoBERTa**: Uses `hfl/chinese-roberta-wwm-ext` for optimal Chinese text processing
- **GPU Acceleration**: Automatically detects and uses GPU when available
- **Optimized for Chinese**: Designed specifically for Chinese office domain conversations
- **High Performance**: Fast inference with confidence scores

## Project Structure

```
chinese_intent_recognition/
├── README.md                 # This file
├── requirements.txt          # Python dependencies  
├── train_intent.py          # Chinese RoBERTa training script
├── intent_infer.py          # Chinese inference script
├── intent_api.py            # FastAPI web service
├── test_api.sh              # API test cases (curl commands)
├── test_chinese_system.py   # Comprehensive system test
├── dataset/
│   ├── __init__.py
│   └── dataset_creator.py    # Chinese dataset generation
├── data/                     # Generated training data
│   ├── train.csv
│   ├── test.csv
│   ├── train.json
│   ├── test.json
│   └── label_mapping.json
├── models/
│   ├── __init__.py
│   └── roberta_classifier.py # Chinese RoBERTa model implementation
├── training/
│   ├── __init__.py
│   └── train.py             # Alternative training script
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py          # Evaluation module
├── inference/
│   ├── __init__.py
│   └── predict.py           # Alternative inference script
└── examples/
    ├── __init__.py
    └── demo.py              # Complete pipeline demonstration
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate Chinese dataset:
   ```bash
   python -m dataset.dataset_creator
   ```

3. Train the Chinese RoBERTa model:
   ```bash
   python train_intent.py
   ```

4. Test inference with Chinese text:
   ```bash
   python intent_infer.py "想订明天两点的会议室，10个人"
   ```

5. Run comprehensive system test:
   ```bash
   python test_chinese_system.py
   ```

## Usage Examples

### Training (训练)
```python
# 训练中文意图识别模型
python train_intent.py
```

### Inference (推理)
```python
from intent_infer import IntentClassifier

# 初始化分类器
clf = IntentClassifier()

# 预测意图
result = clf.predict("想订明天两点的会议室，10个人")
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"All probabilities: {result['probs']}")
```

### Batch Processing (批量处理)
```python
texts = [
    "我想查看这个月的工资单",
    "需要请假三天",
    "电脑开不了机，需要IT支持"
]

results = clf.predict_batch(texts)
for result in results:
    print(f"{result['text']} -> {result['intent']}")
```

### FastAPI Web Service (FastAPI网络服务)

#### Starting the Service (启动服务)
```bash
# 启动FastAPI服务
python intent_api.py --host 0.0.0.0 --port 8000

# 或使用uvicorn直接启动
uvicorn intent_api:app --host 0.0.0.0 --port 8000
```

#### API Endpoints (API端点)

1. **Health Check (健康检查)**
   ```bash
   curl -X GET "http://localhost:8000/health"
   ```

2. **Single Prediction (单个预测)**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "想订明天两点的会议室，10个人"}'
   ```

3. **Batch Prediction (批量预测)**
   ```bash
   curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["想订明天两点的会议室，10个人", "我想查看这个月的工资单"]}'
   ```

4. **Model Information (模型信息)**
   ```bash
   curl -X GET "http://localhost:8000/model/info"
   ```

#### API Documentation (API文档)
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### Test Cases (测试用例)
Run the comprehensive test suite:
```bash
./test_api.sh
```

## Implementation Details

### Chinese Dataset Creation
- Generates synthetic Chinese training data for each intent class
- Includes diverse Chinese phrasings and vocabulary for robust training
- Creates balanced dataset with equal samples per intent
- Supports office domain scenarios in Chinese context

### Chinese RoBERTa Model Architecture
- Uses pre-trained `hfl/chinese-roberta-wwm-ext` as the foundation
- Optimized for Chinese text processing and understanding
- Adds a classification head for intent prediction
- Implements proper Chinese tokenization and preprocessing

### Training Process
- Fine-tunes Chinese RoBERTa on the intent classification task
- GPU detection and automatic utilization when available
- Uses appropriate learning rate and training strategies for Chinese models
- Includes validation and early stopping

### GPU Acceleration
- Automatically detects CUDA availability
- Prioritizes GPU usage for training and inference
- Falls back to CPU if GPU is not available
- Optimized memory usage with FP16 training when possible

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion matrix for detailed analysis
- Per-class performance metrics for each Chinese intent

### Inference
- Simple API for predicting intents from Chinese text
- Confidence scores for predictions
- Batch processing capability
- Support for both single and multiple text inputs

## Educational Value

This implementation includes extensive comments and documentation to help understand:
- How to fine-tune pre-trained Chinese language models
- Chinese intent classification system design
- PyTorch/Transformers best practices for Chinese NLP
- Model evaluation and validation techniques for Chinese text
- GPU acceleration and optimization strategies
- Chinese text processing and tokenization methods