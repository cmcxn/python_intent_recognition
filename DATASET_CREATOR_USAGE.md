# 数据集生成器使用指南 / Dataset Creator Usage Guide

## 概述 / Overview

`dataset/dataset_creator.py` 是中文办公领域意图识别数据集生成器，现已支持选择性意图生成和数据集合并功能。

The `dataset/dataset_creator.py` is a Chinese office domain intent recognition dataset creator, now with selective intent generation and dataset merging capabilities.

## 新功能 / New Features

### 1. 选择性意图生成 / Selective Intent Generation
- 可以选择生成哪些意图的数据，而不需要生成所有8个意图
- Can generate data for specific intents instead of all 8 intents

### 2. 数据集合并 / Dataset Merging  
- 如果data文件夹下有原来生成的数据，会自动合并到新数据集中
- Automatically merges with existing dataset files in the data folder
- 自动去除重复数据 / Automatically removes duplicates

## 使用方法 / Usage

### 基本用法 / Basic Usage
```bash
# 生成所有意图的数据（默认行为）
# Generate data for all intents (default behavior)
python dataset/dataset_creator.py

# 生成指定意图的数据
# Generate data for specific intents
python dataset/dataset_creator.py --intents CHECK_PAYSLIP BOOK_MEETING_ROOM

# 设置每个意图的样本数
# Set number of samples per intent
python dataset/dataset_creator.py --samples-per-intent 50
```

### 高级选项 / Advanced Options
```bash
# 指定输出目录
# Specify output directory
python dataset/dataset_creator.py --output-dir my_dataset

# 不合并现有数据（创建全新数据集）
# Don't merge with existing data (create fresh dataset)
python dataset/dataset_creator.py --no-merge

# 交互模式选择意图
# Interactive mode for selecting intents
python dataset/dataset_creator.py --interactive
```

### 组合使用 / Combined Usage
```bash
# 生成指定意图的数据，每个意图30个样本，输出到test_data目录，不合并现有数据
# Generate specific intents with 30 samples each, output to test_data, no merging
python dataset/dataset_creator.py \
  --intents REQUEST_LEAVE IT_TICKET \
  --samples-per-intent 30 \
  --output-dir test_data \
  --no-merge
```

## 可用意图 / Available Intents

1. `CHECK_PAYSLIP` - 查询工资单相关问题
2. `BOOK_MEETING_ROOM` - 会议室预订请求
3. `REQUEST_LEAVE` - 请假申请
4. `CHECK_BENEFITS` - 福利查询
5. `IT_TICKET` - IT支持工单
6. `EXPENSE_REIMBURSE` - 费用报销
7. `COMPANY_LOOKUP` - 查公司相关信息
8. `USER_LOOKUP` - 查用户相关信息
9. `QUERY_RESPONSIBLE_PERSON` - 查询负责人相关信息

## 输出文件 / Output Files

生成的文件包括：
Generated files include:

- `train.csv` / `train.json` - 训练数据 / Training data
- `test.csv` / `test.json` - 测试数据 / Test data  
- `label_mapping.json` - 标签映射 / Label mapping

## 示例场景 / Example Scenarios

### 场景1：增量添加新意图数据
```bash
# 第一次：生成基础意图数据
python dataset/dataset_creator.py --intents CHECK_PAYSLIP BOOK_MEETING_ROOM --samples-per-intent 100

# 后来：添加更多意图数据（会自动合并）
python dataset/dataset_creator.py --intents REQUEST_LEAVE CHECK_BENEFITS --samples-per-intent 50
```

### 场景2：测试特定意图
```bash
# 创建测试数据集，只包含IT相关意图
python dataset/dataset_creator.py \
  --intents IT_TICKET \
  --samples-per-intent 20 \
  --output-dir test_it \
  --no-merge
```

### 场景3：交互式选择
```bash
# 使用交互模式选择需要的意图
python dataset/dataset_creator.py --interactive
```

## 技术细节 / Technical Details

- 合并时会自动去除重复的文本-标签对 / Automatically removes duplicate text-label pairs during merging
- 保持训练/测试集的平衡分布 / Maintains balanced train/test split
- 支持自定义随机种子以确保结果可重现 / Supports random seed for reproducible results
- 验证意图名称的有效性 / Validates intent names