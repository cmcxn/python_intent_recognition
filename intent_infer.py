#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese Intent Recognition Inference Script (intent_infer.py)

This script provides inference capabilities for the Chinese RoBERTa intent classifier.
It loads a trained model and provides prediction functionality for Chinese text input.

Features:
- Load trained Chinese RoBERTa model
- Fast inference with confidence scores
- Support for single and batch predictions
- GPU acceleration when available
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
from typing import Dict, List, Union

# Intent labels matching the training script
LABEL_LIST = ["CHECK_PAYSLIP","BOOK_MEETING_ROOM","REQUEST_LEAVE",
              "CHECK_BENEFITS","IT_TICKET","EXPENSE_REIMBURSE","COMPANY_LOOKUP",
            "USER_LOOKUP"]

class IntentClassifier:
    """
    Chinese Intent Classifier for office domain tasks.
    
    This classifier uses a fine-tuned Chinese RoBERTa model to predict
    intent categories from Chinese text input.
    """
    
    def __init__(self, model_path="models/intent_roberta", max_len=64):
        """
        Initialize the intent classifier.
        
        Args:
            model_path: Path to the trained model directory
            max_len: Maximum sequence length for tokenization
        """
        self.model_path = model_path
        self.max_len = max_len
        
        # Detect device (prioritize GPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("! Using CPU")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        try:
            print(f"📦 Loading model from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("✓ Tokenizer loaded")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✓ Model loaded and moved to device")
            
            # Load label mapping if available
            label_mapping_path = os.path.join(self.model_path, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, "r", encoding="utf-8") as f:
                    label_mapping = json.load(f)
                    if "LABEL_LIST" in label_mapping:
                        global LABEL_LIST
                        LABEL_LIST = label_mapping["LABEL_LIST"]
                        print(f"✓ Loaded {len(LABEL_LIST)} intent labels")
            
            print("🎯 Intent classifier ready!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Make sure the model is trained and saved properly.")
            raise
    
    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """
        Predict intent for a single text input.
        
        Args:
            text: Chinese text input
            
        Returns:
            Dictionary with prediction results including intent and probabilities
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=self.max_len
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        
        # Get prediction
        pred_idx = int(torch.argmax(logits, dim=-1))
        predicted_intent = LABEL_LIST[pred_idx]
        confidence = probs[pred_idx]
        
        # Create probability dictionary
        prob_dict = {label: round(prob, 4) for label, prob in zip(LABEL_LIST, probs)}
        
        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "probs": prob_dict
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict intents for multiple text inputs.
        
        Args:
            texts: List of Chinese text inputs
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict(text)
            result["text"] = text
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "max_length": self.max_len,
            "num_labels": len(LABEL_LIST),
            "intent_labels": LABEL_LIST,
            "model_size": sum(p.numel() for p in self.model.parameters())
        }

def demo():
    """Demonstration of the intent classifier."""
    print("🎯 Chinese Intent Recognition Demo")
    print("=" * 40)
    
    # Initialize classifier
    try:
        clf = IntentClassifier()
    except Exception as e:
        print(f"❌ Could not initialize classifier: {e}")
        print("Please train the model first using: python train_intent.py")
        return
    
    # Show model info
    info = clf.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"Device: {info['device']}")
    print(f"Labels: {info['num_labels']}")
    print(f"Model Size: {info['model_size']:,} parameters")
    
    # Demo examples
    demo_texts = [
        "想订明天两点的会议室，10个人",
        "我想查看这个月的工资单",
        "我需要请假三天，下周一到周三",
        "公积金怎么查询？",
        "我的电脑开不了机，需要IT支持",
        "出差费用怎么报销？"
    ]
    
    print(f"\n🔍 Demo Predictions:")
    print("-" * 50)
    
    for i, text in enumerate(demo_texts, 1):
        result = clf.predict(text)
        print(f"{i}. Text: \"{text}\"")
        print(f"   Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        
        # Show top 2 probabilities
        sorted_probs = sorted(result['probs'].items(), key=lambda x: x[1], reverse=True)
        print(f"   Top probabilities:")
        for label, prob in sorted_probs[:2]:
            print(f"     {label}: {prob:.3f}")
        print()

def interactive_mode():
    """Interactive mode for testing custom inputs."""
    print("🔄 Interactive Mode")
    print("Enter Chinese text to classify (or 'quit' to exit)")
    print("-" * 50)
    
    try:
        clf = IntentClassifier()
    except Exception as e:
        print(f"❌ Could not initialize classifier: {e}")
        return
    
    while True:
        try:
            text = input("\n输入文本 (Enter text): ").strip()
            
            if text.lower() in ['quit', 'exit', 'q', '退出']:
                print("Goodbye! 再见！")
                break
            
            if not text:
                continue
            
            result = clf.predict(text)
            print(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
            
            # Show all probabilities
            print("All probabilities:")
            for label, prob in result['probs'].items():
                print(f"  {label}: {prob:.3f}")
        
        except KeyboardInterrupt:
            print("\nGoodbye! 再见！")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            # Single prediction mode
            text = " ".join(sys.argv[1:])
            try:
                clf = IntentClassifier()
                result = clf.predict(text)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Demo mode
        demo()