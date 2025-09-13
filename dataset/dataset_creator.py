"""
Dataset Creator for Chinese Office Domain Intent Recognition

This module generates synthetic training data for Chinese office domain intent classification.
It creates diverse sample queries for each intent to train the Chinese RoBERTa model effectively.

The dataset includes 8 intent categories (aligned with reference implementation):
1. CHECK_PAYSLIP - 查询工资单相关问题
2. BOOK_MEETING_ROOM - 会议室预订请求
3. REQUEST_LEAVE - 请假申请
4. CHECK_BENEFITS - 福利查询
5. IT_TICKET - IT支持工单
6. EXPENSE_REIMBURSE - 费用报销
7. COMPANY_LOOKUP - 查公司相关信息
8. USER_LOOKUP - 查用户相关信息
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
            "EXPENSE_REIMBURSE",
            "COMPANY_LOOKUP",
            "USER_LOOKUP"
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
        
        # COMPANY_LOOKUP templates (查公司)
        self.company_templates = [
            "帮我介绍一下{company}",
            "帮我查询一下{company}",
            "我想了解{company}",
            "查询{company}的信息",
            "请介绍{company}",
            "{company}的详细信息",
            "我需要{company}的资料",
            "能告诉我{company}的情况吗？",
            "查看{company}的详情",
            "{company}是什么公司？"
        ]
        
        self.company_vocab = {
            "company": ["数科公司", "通用技术集团数字智能科技有限公司", "数字智能科技有限公司", "集团公司", "科技公司", "技术公司"]
        }
        
        # USER_LOOKUP templates (查用户) - organized by query type
        self.user_basic_templates = [
            "帮我查询一下{person}",
            "查询{person}的信息",
            "我想了解{person}",
            "请查找{person}",
            "帮我找{person}"
        ]
        
        self.user_clarified_templates = [
            "帮我查询一下{person}，{name_clarification}",
            "查询{person}，{name_clarification}",
            "我要找{person}，{name_clarification}"
        ]
        
        self.user_company_templates = [
            "查询{company}的{person}",
            "帮我查一下{company}的{person}",
            "我想了解{company}的{person}",
            "{company}有个{person}吗？"
        ]
        
        self.user_contact_templates = [
            "帮我查一下{company}{person}的{contact_type}",
            "查询{username}的{contact_type}",
            "{person}的{contact_type}是多少？",
            "我需要{person}的{contact_type}"
        ]
        
        self.user_department_templates = [
            "帮我查一下{company}的{person}目前在哪个{org_unit}？",
            "{person}在{company}的{job_aspect}是什么？",
            "帮我查一下{company}{department}的{job_title}是谁",
            "帮我查询一下{company}{department}的人员有哪些？"
        ]
        
        self.user_attribute_templates = [
            "查询一下{company}{department}的{gender}员工都有谁",
            "帮我查询一下办公地点在{location}的{person}",
            "帮我查询一下{gender}的{person}",
            "办公地点在{location}的员工有哪些？"
        ]
        
        self.user_reverse_templates = [
            "{phone_number}是谁的{contact_type}？",
            "帮我查下{contact_type}尾号{phone_suffix}的用户是谁？",
            "这个{contact_type}{phone_number}是谁的？"
        ]
        
        self.user_directory_templates = [
            "查询{company}的通讯录",
            "帮我看一下{company}的员工名单",
            "{company}的联系人列表",
            "我需要{company}的人员信息"
        ]
        
        self.user_vocab = {
            "person": ["张三", "孔文琦", "李四", "王五", "赵六", "陈七", "刘八", "马九"],
            "username": ["zhangsan1", "liwang2", "chenliu3", "maqian4", "user123", "test001"],
            "company": ["数科公司", "通用技术集团数字智能科技有限公司", "公司"],
            "contact_type": ["手机号", "办公电话", "邮箱号码", "座机号", "电话号码", "联系方式"],
            "name_clarification": ["张是弓长张", "琦是王字旁加奇怪的奇", "李是木子李", "王是三横一竖王"],
            "org_unit": ["部门", "科室", "事业部", "中心"],
            "job_aspect": ["职位", "岗位", "角色", "工作"],
            "phone_number": ["13282814679", "81168151", "13912345678", "010-88888888", "0571-12345678"],
            "phone_suffix": ["2345", "8888", "1234", "6789", "0000"],
            "department": ["管控数字化事业部", "综合办公室", "技术部", "市场部", "人事部", "财务部"],
            "job_title": ["总监", "总经理", "经理", "主任", "专员", "助理"],
            "gender": ["男性", "女性"],
            "location": ["北京", "上海", "深圳", "杭州", "广州", "成都"]
        }
    
    def _generate_user_lookup_samples(self) -> List[str]:
        """
        Generate training samples for USER_LOOKUP intent with multiple template groups.
        
        Returns:
            List of generated text samples
        """
        samples = []
        samples_per_group = max(1, self.samples_per_intent // 8)  # Distribute across 8 template groups
        
        # Generate samples for each template group
        template_groups = [
            (self.user_basic_templates, {"person": self.user_vocab["person"]}),
            (self.user_clarified_templates, {"person": self.user_vocab["person"], "name_clarification": self.user_vocab["name_clarification"]}),
            (self.user_company_templates, {"company": self.user_vocab["company"], "person": self.user_vocab["person"]}),
            (self.user_contact_templates, {"company": self.user_vocab["company"], "person": self.user_vocab["person"], "username": self.user_vocab["username"], "contact_type": self.user_vocab["contact_type"]}),
            (self.user_department_templates, {"company": self.user_vocab["company"], "person": self.user_vocab["person"], "org_unit": self.user_vocab["org_unit"], "job_aspect": self.user_vocab["job_aspect"], "department": self.user_vocab["department"], "job_title": self.user_vocab["job_title"]}),
            (self.user_attribute_templates, {"company": self.user_vocab["company"], "department": self.user_vocab["department"], "gender": self.user_vocab["gender"], "location": self.user_vocab["location"], "person": self.user_vocab["person"]}),
            (self.user_reverse_templates, {"phone_number": self.user_vocab["phone_number"], "contact_type": self.user_vocab["contact_type"], "phone_suffix": self.user_vocab["phone_suffix"]}),
            (self.user_directory_templates, {"company": self.user_vocab["company"]})
        ]
        
        for templates, vocab in template_groups:
            for _ in range(samples_per_group):
                # Randomly select a template
                template = random.choice(templates)
                
                # Fill in placeholders with random vocabulary
                sample = template
                for placeholder, options in vocab.items():
                    if f"{{{placeholder}}}" in sample:
                        sample = sample.replace(f"{{{placeholder}}}", random.choice(options))
                
                samples.append(sample)
        
        # Fill remaining samples if we haven't reached the target
        while len(samples) < self.samples_per_intent:
            # Pick a random template group
            templates, vocab = random.choice(template_groups)
            template = random.choice(templates)
            
            sample = template
            for placeholder, options in vocab.items():
                if f"{{{placeholder}}}" in sample:
                    sample = sample.replace(f"{{{placeholder}}}", random.choice(options))
            
            samples.append(sample)
        
        return samples[:self.samples_per_intent]
    
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
            elif intent == "COMPANY_LOOKUP":
                samples = self._generate_samples_for_intent(intent, self.company_templates, self.company_vocab)
            elif intent == "USER_LOOKUP":
                samples = self._generate_user_lookup_samples()
            
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
    
    # Create dataset with 100 samples per intent (800 total samples)
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