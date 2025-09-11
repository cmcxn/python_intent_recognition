"""
Dataset Creator for Chinese Office Domain Intent Recognition

This module generates synthetic training data for Chinese office domain intent classification.
It creates diverse sample queries for each intent to train the Chinese RoBERTa model effectively.

The dataset includes 6 intent categories (aligned with reference implementation):
1. CHECK_PAYSLIP - 查询工资单相关问题
2. BOOK_MEETING_ROOM - 会议室预订请求
3. REQUEST_LEAVE - 请假申请
4. CHECK_BENEFITS - 福利查询
5. IT_TICKET - IT支持工单
6. EXPENSE_REIMBURSE - 费用报销
"""

import json
import pandas as pd
import random
from typing import List, Dict, Tuple
from pathlib import Path


class ChineseOfficeIntentDatasetCreator:
    """
    Creates a synthetic dataset for Chinese office domain intent recognition.
    
    This class generates diverse training examples for each intent category,
    ensuring the model can learn to recognize various ways users might express
    their intentions in a Chinese office environment.
    """
    
    def __init__(self, samples_per_intent: int = 100):
        """
        Initialize the dataset creator.
        
        Args:
            samples_per_intent: Number of training samples to generate per intent
        """
        self.samples_per_intent = samples_per_intent
        # Updated to match reference implementation with Chinese labels
        self.intent_labels = [
            "CHECK_PAYSLIP",
            "BOOK_MEETING_ROOM", 
            "REQUEST_LEAVE",
            "CHECK_BENEFITS",
            "IT_TICKET",
            "EXPENSE_REIMBURSE"
        ]
        
        # Define templates and vocabulary for each intent
        self._define_intent_templates()
    
    def _define_intent_templates(self):
        """
        Define template patterns and vocabulary for generating diverse Chinese queries.
        
        Each intent has multiple templates with different phrasings and structures
        to create realistic variations of how users might express their needs in Chinese.
        """
        
        # CHECK_PAYSLIP templates and vocabulary (查询工资单)
        self.payslip_templates = [
            "我想查看{period}的工资单",
            "帮我查询{period}{type}",
            "我需要看一下{period}的{type}",
            "{period}的工资是多少？",
            "查看我的{period}{type}",
            "能帮我查询{period}的{type}吗？",
            "我要看{period}的薪资明细",
            "请显示我的{period}{type}",
            "{period}发了多少工资？",
            "我需要{period}的{type}信息"
        ]
        
        self.payslip_vocab = {
            "period": ["这个月", "上个月", "本月", "上月", "当月", "这月", "月底", "本年度", "今年", "去年"],
            "type": ["工资单", "薪资", "工资", "薪水", "收入", "薪酬", "工资条", "薪资单", "收入明细"]
        }
        
        # BOOK_MEETING_ROOM templates (会议室预订)
        self.meeting_templates = [
            "我想预订{time}的{room_type}",
            "能帮我订一个{room_type}{time}用吗？",
            "我需要预约{time}的{room_type}",
            "{time}有{room_type}可以预订吗？",
            "想订{time}的{room_type}",
            "请帮我安排{time}的{room_type}",
            "我要预定{room_type}，{time}",
            "能预约{time}的{room_type}吗？",
            "帮忙订个{room_type}，{time}用",
            "{time}我需要用{room_type}"
        ]
        
        self.meeting_vocab = {
            "room_type": ["会议室", "大会议室", "小会议室", "培训室", "讨论室", "视频会议室", "多媒体室"],
            "time": ["明天上午", "下午两点", "明天下午", "后天", "周一", "周五下午", "明天十点", "下周", "今天下午"]
        }
        
        # REQUEST_LEAVE templates (请假申请)
        self.leave_templates = [
            "我想请{leave_type}{time}",
            "我需要{time}请{leave_type}",
            "能帮我申请{time}的{leave_type}吗？",
            "我要{time}请{leave_type}",
            "{time}我想请{leave_type}",
            "申请{time}的{leave_type}",
            "我想{time}休{leave_type}",
            "需要请{leave_type}，{time}",
            "{time}我要请假",
            "帮我申请{leave_type}，{time}"
        ]
        
        self.leave_vocab = {
            "leave_type": ["病假", "事假", "年假", "婚假", "产假", "调休", "假", "假期"],
            "time": ["明天", "下周", "这周五", "下个月", "这个月底", "后天", "周一到周三", "两天", "一周"]
        }
        
        # CHECK_BENEFITS templates (福利查询)
        self.benefits_templates = [
            "我想了解{benefit_type}",
            "能告诉我{benefit_type}的详情吗？",
            "查询{benefit_type}信息",
            "我的{benefit_type}有哪些？",
            "请介绍一下{benefit_type}",
            "{benefit_type}怎么申请？",
            "我想知道{benefit_type}政策",
            "查看{benefit_type}说明",
            "{benefit_type}的标准是什么？",
            "公司的{benefit_type}如何？"
        ]
        
        self.benefits_vocab = {
            "benefit_type": ["社保", "公积金", "保险", "福利待遇", "医疗保险", "年终奖", "餐补", "交通补贴", "住房补贴", "培训福利"]
        }
        
        # IT_TICKET templates (IT支持工单)
        self.it_templates = [
            "我的{device}有{problem}",
            "{device}{problem}了，需要帮助",
            "IT支持：{device}{problem}",
            "电脑问题：{problem}",
            "我需要IT帮助，{device}{problem}",
            "{device}出现{problem}，怎么办？",
            "技术支持：{problem}",
            "帮忙解决{device}的{problem}",
            "{device}有故障：{problem}",
            "IT工单：{device}{problem}"
        ]
        
        self.it_vocab = {
            "device": ["电脑", "笔记本", "打印机", "网络", "邮箱", "系统", "软件", "设备"],
            "problem": ["无法开机", "网络连接不上", "运行很慢", "死机", "无法打印", "登录不了", "崩溃", "出错"]
        }
        
        # EXPENSE_REIMBURSE templates (费用报销)
        self.expense_templates = [
            "我需要报销{expense_type}",
            "申请{expense_type}报销",
            "我想报销{expense_type}费用",
            "{expense_type}怎么报销？",
            "报销申请：{expense_type}",
            "我要提交{expense_type}的报销单",
            "能帮我报销{expense_type}吗？",
            "{expense_type}报销流程是什么？",
            "我有{expense_type}需要报销",
            "请帮我处理{expense_type}报销"
        ]
        
        self.expense_vocab = {
            "expense_type": ["差旅费", "交通费", "餐费", "住宿费", "培训费", "办公用品", "通讯费", "会议费", "招待费", "加班餐费"]
        }
    
    def _generate_samples_for_intent(self, intent: str, templates: List[str], vocab: Dict[str, List[str]]) -> List[str]:
        """
        Generate training samples for a specific intent.
        
        Args:
            intent: The intent label
            templates: List of template strings with placeholders
            vocab: Dictionary mapping placeholders to possible values
            
        Returns:
            List of generated text samples
        """
        samples = []
        
        for _ in range(self.samples_per_intent):
            # Randomly select a template
            template = random.choice(templates)
            
            # Fill in placeholders with random vocabulary
            sample = template
            for placeholder, options in vocab.items():
                if f"{{{placeholder}}}" in sample:
                    sample = sample.replace(f"{{{placeholder}}}", random.choice(options))
            
            samples.append(sample)
        
        return samples
    
    def generate_dataset(self) -> Tuple[List[str], List[str]]:
        """
        Generate the complete dataset for all intents.
        
        Returns:
            Tuple of (texts, labels) where texts are the input queries
            and labels are the corresponding intent labels
        """
        texts = []
        labels = []
        
        # Generate samples for each intent
        for intent in self.intent_labels:
            if intent == "CHECK_PAYSLIP":
                samples = self._generate_samples_for_intent(intent, self.payslip_templates, self.payslip_vocab)
            elif intent == "BOOK_MEETING_ROOM":
                samples = self._generate_samples_for_intent(intent, self.meeting_templates, self.meeting_vocab)
            elif intent == "REQUEST_LEAVE":
                samples = self._generate_samples_for_intent(intent, self.leave_templates, self.leave_vocab)
            elif intent == "CHECK_BENEFITS":
                samples = self._generate_samples_for_intent(intent, self.benefits_templates, self.benefits_vocab)
            elif intent == "IT_TICKET":
                samples = self._generate_samples_for_intent(intent, self.it_templates, self.it_vocab)
            elif intent == "EXPENSE_REIMBURSE":
                samples = self._generate_samples_for_intent(intent, self.expense_templates, self.expense_vocab)
            
            # Add samples and labels
            texts.extend(samples)
            labels.extend([intent] * len(samples))
        
        return texts, labels
    
    def create_train_test_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split of the dataset.
        
        Args:
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_df, test_df) DataFrames
        """
        texts, labels = self.generate_dataset()
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split by intent to ensure balanced representation
        train_dfs = []
        test_dfs = []
        
        for intent in self.intent_labels:
            intent_df = df[df['label'] == intent]
            split_idx = int(len(intent_df) * (1 - test_size))
            
            train_dfs.append(intent_df.iloc[:split_idx])
            test_dfs.append(intent_df.iloc[split_idx:])
        
        train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
        return train_df, test_df
    
    def save_dataset(self, output_dir: str = "data"):
        """
        Generate and save the dataset to files.
        
        Args:
            output_dir: Directory to save the dataset files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate train/test split
        train_df, test_df = self.create_train_test_split()
        
        # Save as CSV files
        train_df.to_csv(output_path / "train.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        # Save as JSON files for easier loading
        train_data = {
            'texts': train_df['text'].tolist(),
            'labels': train_df['label'].tolist()
        }
        test_data = {
            'texts': test_df['text'].tolist(),
            'labels': test_df['label'].tolist()
        }
        
        with open(output_path / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_path / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save label mapping
        label_mapping = {
            'intent_labels': self.intent_labels,
            'label_to_id': {label: i for i, label in enumerate(self.intent_labels)},
            'id_to_label': {i: label for i, label in enumerate(self.intent_labels)}
        }
        
        with open(output_path / "label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"Dataset saved to {output_path}")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Intents: {len(self.intent_labels)}")
        
        # Print sample distribution
        print("\nSample distribution:")
        print(train_df['label'].value_counts().sort_index())


def main():
    """Main function to create and save the Chinese dataset."""
    print("Creating Chinese Office Domain Intent Recognition Dataset...")
    print("创建中文办公领域意图识别数据集...")
    
    # Create dataset with 100 samples per intent (600 total samples)
    creator = ChineseOfficeIntentDatasetCreator(samples_per_intent=100)
    
    # Save dataset to data directory
    creator.save_dataset("data")
    
    print("\n数据集创建完成！Dataset creation completed!")
    print("创建的文件 Files created:")
    print("- data/train.csv")
    print("- data/test.csv") 
    print("- data/train.json")
    print("- data/test.json")
    print("- data/label_mapping.json")


if __name__ == "__main__":
    main()