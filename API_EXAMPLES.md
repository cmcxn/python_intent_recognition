# FastAPI Intent Recognition Service Examples

## 启动服务 (Starting the Service)

```bash
# 启动FastAPI服务
python intent_api.py --host 0.0.0.0 --port 8000

# 或使用uvicorn直接启动
uvicorn intent_api:app --host 0.0.0.0 --port 8000 --reload
```

## API端点测试 (API Endpoint Testing)

### 1. 健康检查 (Health Check)
```bash
curl -X GET "http://localhost:8000/health" \
  -H "Content-Type: application/json"

# 预期响应 Expected Response:
# {"status":"healthy","model_loaded":false,"version":"1.0.0"}
```

### 2. 根端点 (Root Endpoint)
```bash
curl -X GET "http://localhost:8000/" \
  -H "Content-Type: application/json"

# 预期响应 Expected Response:
# {
#   "message": "Chinese Intent Recognition API",
#   "version": "1.0.0",
#   "endpoints": {
#     "predict": "/predict",
#     "predict_batch": "/predict/batch",
#     "model_info": "/model/info",
#     "health": "/health"
#   }
# }
```

### 3. 模型信息 (Model Information)
```bash
curl -X GET "http://localhost:8000/model/info" \
  -H "Content-Type: application/json"

# 预期响应 Expected Response:
# {
#   "model_path": "models/intent_roberta",
#   "device": "cpu",
#   "max_length": 64,
#   "num_labels": 6,
#   "intent_labels": ["CHECK_PAYSLIP", "BOOK_MEETING_ROOM", "REQUEST_LEAVE", "CHECK_BENEFITS", "IT_TICKET", "EXPENSE_REIMBURSE"],
#   "model_available": false
# }
```

### 4. 单个预测 (Single Prediction)

#### 会议室预订 (Meeting Room Booking)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "想订明天两点的会议室，10个人"}'

# 预期响应 Expected Response:
# {
#   "intent": "BOOK_MEETING_ROOM",
#   "confidence": 0.9181,
#   "probs": {
#     "CHECK_PAYSLIP": 0.0214,
#     "BOOK_MEETING_ROOM": 0.9181,
#     "REQUEST_LEAVE": 0.0121,
#     "CHECK_BENEFITS": 0.021,
#     "IT_TICKET": 0.0153,
#     "EXPENSE_REIMBURSE": 0.0121
#   }
# }
```

#### 工资单查询 (Payslip Check)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "我想查看这个月的工资单"}'
```

#### 请假申请 (Leave Request)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "我需要请假三天，下周一到周三"}'
```

#### 福利查询 (Benefits Check)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "公积金怎么查询？"}'
```

#### IT工单 (IT Ticket)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "我的电脑开不了机，需要IT支持"}'
```

#### 费用报销 (Expense Reimbursement)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "出差费用怎么报销？"}'
```

### 5. 批量预测 (Batch Prediction)
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "想订明天两点的会议室，10个人",
      "我想查看这个月的工资单",
      "我需要请假三天",
      "公积金怎么查询？",
      "电脑开不了机，需要IT支持",
      "出差费用怎么报销？"
    ]
  }'

# 预期响应 Expected Response:
# {
#   "results": [
#     {
#       "intent": "BOOK_MEETING_ROOM",
#       "confidence": 0.9152,
#       "probs": {...},
#       "text": "想订明天两点的会议室，10个人"
#     },
#     ...
#   ]
# }
```

## 错误处理测试 (Error Handling Tests)

### 空文本 (Empty Text)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'

# 预期响应 Expected Response:
# {
#   "detail": [
#     {
#       "type": "string_too_short",
#       "loc": ["body", "text"],
#       "msg": "String should have at least 1 character",
#       "input": "",
#       "ctx": {"min_length": 1}
#     }
#   ]
# }
```

### 无效JSON (Invalid JSON)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"invalid": json}'

# 预期响应 Expected Response: 422 Unprocessable Entity
```

## API文档访问 (API Documentation Access)

### Swagger UI
```bash
# 在浏览器中打开 (Open in browser):
http://localhost:8000/docs
```

### ReDoc
```bash
# 在浏览器中打开 (Open in browser):
http://localhost:8000/redoc
```

## Python客户端示例 (Python Client Example)

```python
import requests
import json

# 基础URL (Base URL)
BASE_URL = "http://localhost:8000"

# 单个预测 (Single Prediction)
def predict_intent(text):
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        json={"text": text}
    )
    return response.json()

# 批量预测 (Batch Prediction)
def predict_batch(texts):
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        headers={"Content-Type": "application/json"},
        json={"texts": texts}
    )
    return response.json()

# 使用示例 (Usage Example)
if __name__ == "__main__":
    # 单个预测
    result = predict_intent("想订明天两点的会议室，10个人")
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']}")
    
    # 批量预测
    texts = [
        "我想查看这个月的工资单",
        "需要请假三天",
        "电脑坏了需要修理"
    ]
    results = predict_batch(texts)
    for result in results['results']:
        print(f"{result['text']} -> {result['intent']}")
```

## 支持的意图类别 (Supported Intent Categories)

1. **CHECK_PAYSLIP** - 查询工资单相关问题
   - 示例: "我想查看这个月的工资单", "工资什么时候发"

2. **BOOK_MEETING_ROOM** - 会议室预订请求
   - 示例: "想订明天两点的会议室，10个人", "预订会议室"

3. **REQUEST_LEAVE** - 请假申请
   - 示例: "我需要请假三天", "想请年假"

4. **CHECK_BENEFITS** - 福利查询
   - 示例: "公积金怎么查询？", "保险如何报销"

5. **IT_TICKET** - IT支持工单
   - 示例: "我的电脑开不了机", "网络连不上"

6. **EXPENSE_REIMBURSE** - 费用报销
   - 示例: "出差费用怎么报销？", "发票如何提交"