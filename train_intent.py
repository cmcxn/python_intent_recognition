#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese Intent Recognition Training Script (train_intent.py)

This script implements the Chinese RoBERTa-based intent classification training
pipeline as specified in the requirements. It uses the hfl/chinese-roberta-wwm-ext
model and supports GPU acceleration when available.

Features:
- Chinese RoBERTa (hfl/chinese-roberta-wwm-ext) model
- GPU detection and prioritized usage
- Training with validation and metrics
- Model saving for inference
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from sklearn.metrics import classification_report, confusion_matrix


print("gpu_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0=", torch.cuda.get_device_name(0))
else:
    print("gpu0= No GPU available")
# Chinese RoBERTa model configuration
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 中文RoBERTa-WWM-Ext

# Intent labels matching the reference implementation
LABEL_LIST = ["CHECK_PAYSLIP","BOOK_MEETING_ROOM","REQUEST_LEAVE",
              "CHECK_BENEFITS","IT_TICKET","EXPENSE_REIMBURSE","COMPANY_LOOKUP",
            "USER_LOOKUP","QUERY_RESPONSIBLE_PERSON"]
LABEL2ID = {l:i for i,l in enumerate(LABEL_LIST)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

def _normalize_and_map_labels(df: pd.DataFrame, path: str) -> pd.DataFrame:
    # 统一格式：去空格、大写
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    known = set(LABEL2ID.keys())
    unknown_labels = sorted(set(df["label"]) - known)
    if unknown_labels:
        # 直接报错，提示未知标签有哪些 & 数量
        counts = df[df["label"].isin(unknown_labels)]["label"].value_counts().to_dict()
        raise ValueError(f"{path} 存在未在 LABEL_LIST 中的标签: {unknown_labels}，计数: {counts}\n"
                         f"请修正 CSV 的 label 或更新 LABEL_LIST。")
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)
    return df

def load_csv(path: str) -> Dataset:
    df = pd.read_csv(path)
    # 兼容常见文本列名
    text_candidates = [c for c in ["text","question","utterance","sentence","content"] if c in df.columns]
    if not text_candidates:
        raise ValueError(f"{path} 缺少文本列（期望列名之一：text/question/utterance/sentence/content）")
    text_col = text_candidates[0]

    df = df[[text_col, "label"]].dropna()
    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str).str.strip()

    df = _normalize_and_map_labels(df, path)
    return Dataset.from_pandas(df[["text","label_id"]], preserve_index=False)


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of computed metrics
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

@dataclass
class DataCollator:
    """
    Data collator for Chinese text tokenization.
    """
    tokenizer: AutoTokenizer
    max_len: int = 64
    
    def __call__(self, features: List[Dict]):
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Tokenized batch ready for model input
        """
        texts = [f["text"] for f in features]
        # labels = [f["label_id"] for f in features]
        labels = [int(f["label_id"]) for f in features]
        enc = self.tokenizer(texts, truncation=True, padding=True, 
                           max_length=self.max_len, return_tensors="pt")
        
        # Handle the case where input_ids might be None due to tokenizer issues
        if enc["input_ids"] is None:
            raise ValueError("Tokenizer returned None for input_ids. Check tokenizer configuration.")
        
        # Use torch.tensor instead of input_ids.new_tensor to be more robust
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

def detect_device():
    """
    Detect and configure the best available device (GPU preferred).
    
    Returns:
        Device information and configuration
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        use_fp16 = True
    else:
        device = torch.device("cpu")
        print("! No GPU available, using CPU")
        use_fp16 = False
    
    return device, use_fp16

def main():
    """Main training function."""
    print("🚀 Chinese Intent Recognition Training")
    print("=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"Intent Categories: {len(LABEL_LIST)}")
    print(f"Labels: {LABEL_LIST}")
    print()
    
    # Device detection
    device, use_fp16 = detect_device()
    
    # Check if data files exist
    train_path = "data/train_strat.csv"
    dev_path = "data/dev_strat.csv"
    
    # If the exact files don't exist, try to use the generated ones
    if not os.path.exists(train_path):
        train_path = "data/train.csv"
        print(f"Using {train_path} instead of intent_train.csv")
    
    if not os.path.exists(dev_path):
        dev_path = "data/test.csv"
        print(f"Using {dev_path} instead of intent_dev.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        print("❌ Training data not found!")
        print("Please run the dataset creator first:")
        print("python -m dataset.dataset_creator")
        return
    
    try:
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Fix padding token issue for Chinese RoBERTa
        # 确保有 pad_token（BERT 类一般都有 [PAD]，此处兜底更稳）
        if tokenizer.pad_token is None:
            if tokenizer.sep_token is not None:
                tokenizer.pad_token = tokenizer.sep_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print(f"✓ Set padding token -> id={tokenizer.pad_token_id}")

        print("✓ Tokenizer loaded successfully")
        
        # Load datasets
        print("📊 Loading datasets...")
        train_ds = load_csv(train_path)
        dev_ds = load_csv(dev_path)
        print(f"✓ Training samples: {len(train_ds)}")
        print(f"✓ Validation samples: {len(dev_ds)}")
        
        # Load model
        print("🤖 Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL, 
            label2id=LABEL2ID
        )
        print("✓ Model loaded successfully")
        
        # Training arguments
        args = TrainingArguments(
            output_dir="ckpt/intent",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            num_train_epochs=4,  # 3~5 之间都可
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
            fp16=use_fp16,
            dataloader_pin_memory=torch.cuda.is_available(),
            logging_steps=10,
            logging_dir="logs/intent",
            seed=42,
            remove_unused_columns=False,
            weight_decay=0.01,  # ★ 新增
            warmup_ratio=0.1,  # ★ 新增
            lr_scheduler_type="cosine",  # ★ 可选
        )

        # Data collator
        collator = DataCollator(tokenizer, max_len=128)

        # Initialize trainer
        print("🏋️ Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        
        # Start training
        print("🚀 Starting training...")
        print("=" * 30)
        trainer.train()
        
        # Save model
        print("💾 Saving model...")
        os.makedirs("models/intent_roberta", exist_ok=True)
        trainer.save_model("models/intent_roberta")
        tokenizer.save_pretrained("models/intent_roberta")
        
        # Save label mapping
        import json
        label_mapping = {
            "LABEL_LIST": LABEL_LIST,
            "LABEL2ID": LABEL2ID,
            "ID2LABEL": ID2LABEL
        }
        with open("models/intent_roberta/label_mapping.json", "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
        print("✅ 训练完成，模型保存在 models/intent_roberta")
        print("✅ Training completed, model saved to models/intent_roberta")
        
        # Final evaluation
        print("\n📊 Final Evaluation:")
        results = trainer.evaluate()
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"F1 Score: {results['eval_f1']:.4f}")
        print(f"Precision: {results['eval_precision']:.4f}")
        print(f"Recall: {results['eval_recall']:.4f}")
        pred = trainer.predict(dev_ds)
        y_true = pred.label_ids
        y_pred = np.argmax(pred.predictions, axis=-1)

        # 分类报告（逐类 Precision/Recall/F1）
        print(classification_report(
            y_true, y_pred,
            target_names=[l for l in LABEL_LIST],
            digits=4
        ))

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()