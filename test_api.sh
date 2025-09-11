#!/bin/bash
# 
# Chinese Intent Recognition API Test Cases
# 中文意图识别API测试用例
#
# This script contains curl commands to test all endpoints of the FastAPI service
# 该脚本包含测试FastAPI服务所有端点的curl命令

API_BASE="http://localhost:8000"

echo "🧪 Testing Chinese Intent Recognition API"
echo "测试中文意图识别API"
echo "========================================"

# Test 1: Health Check
echo "📋 Test 1: Health Check (健康检查)"
curl -X GET "${API_BASE}/health" \
  -H "Content-Type: application/json" | jq '.'
echo -e "\n"

# Test 2: Root endpoint
echo "📋 Test 2: Root Endpoint (根端点)"
curl -X GET "${API_BASE}/" \
  -H "Content-Type: application/json" | jq '.'
echo -e "\n"

# Test 3: Model Info
echo "📋 Test 3: Model Info (模型信息)"
curl -X GET "${API_BASE}/model/info" \
  -H "Content-Type: application/json" | jq '.'
echo -e "\n"

# Test 4: Single Prediction - Meeting Room Booking
echo "📋 Test 4: Single Prediction - Meeting Room (单个预测 - 会议室预订)"
curl -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "想订明天两点的会议室，10个人"}' | jq '.'
echo -e "\n"

# Test 5: Single Prediction - Payslip Check
echo "📋 Test 5: Single Prediction - Payslip (单个预测 - 工资单查询)"
curl -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "我想查看这个月的工资单"}' | jq '.'
echo -e "\n"

# Test 6: Single Prediction - Leave Request
echo "📋 Test 6: Single Prediction - Leave Request (单个预测 - 请假申请)"
curl -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "我需要请假三天，下周一到周三"}' | jq '.'
echo -e "\n"

# Test 7: Single Prediction - Benefits Check
echo "📋 Test 7: Single Prediction - Benefits (单个预测 - 福利查询)"
curl -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "公积金怎么查询？"}' | jq '.'
echo -e "\n"

# Test 8: Single Prediction - IT Ticket
echo "📋 Test 8: Single Prediction - IT Ticket (单个预测 - IT工单)"
curl -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "我的电脑开不了机，需要IT支持"}' | jq '.'
echo -e "\n"

# Test 9: Single Prediction - Expense Reimbursement
echo "📋 Test 9: Single Prediction - Expense (单个预测 - 费用报销)"
curl -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "出差费用怎么报销？"}' | jq '.'
echo -e "\n"

# Test 10: Batch Prediction
echo "📋 Test 10: Batch Prediction (批量预测)"
curl -X POST "${API_BASE}/predict/batch" \
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
  }' | jq '.'
echo -e "\n"

# Test 11: Error Handling - Empty Text
echo "📋 Test 11: Error Handling - Empty Text (错误处理 - 空文本)"
curl -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": ""}' | jq '.'
echo -e "\n"

# Test 12: Error Handling - Invalid JSON
echo "📋 Test 12: Error Handling - Invalid JSON (错误处理 - 无效JSON)"
curl -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{"invalid": json}' 2>/dev/null || echo "❌ Expected error for invalid JSON"
echo -e "\n"

echo "✅ All tests completed!"
echo "所有测试完成！"