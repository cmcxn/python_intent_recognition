#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese Intent Recognition System Test

This script tests the complete Chinese intent recognition system including:
1. Chinese dataset generation
2. Model initialization with GPU detection
3. Training script preparation
4. Inference capabilities

The system uses hfl/chinese-roberta-wwm-ext for Chinese text processing.
"""

import os
import sys
from pathlib import Path

def test_chinese_dataset_generation():
    """Test Chinese dataset generation."""
    print("🔍 Testing Chinese Dataset Generation")
    print("-" * 40)
    
    try:
        from dataset.dataset_creator import ChineseOfficeIntentDatasetCreator
        
        # Create small dataset for testing
        creator = ChineseOfficeIntentDatasetCreator(samples_per_intent=5)
        texts, labels = creator.generate_dataset()
        
        print(f"✓ Generated {len(texts)} Chinese samples")
        print(f"✓ Found {len(set(labels))} unique intents")
        print(f"✓ Intent labels: {creator.intent_labels}")
        
        # Show sample Chinese texts
        print("\n📝 Sample Chinese texts:")
        for intent in creator.intent_labels[:3]:
            intent_examples = [text for text, label in zip(texts, labels) if label == intent]
            if intent_examples:
                print(f"  {intent}: \"{intent_examples[0]}\"")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        return False

def test_model_initialization():
    """Test Chinese RoBERTa model initialization."""
    print("\n🤖 Testing Model Initialization")
    print("-" * 40)
    
    try:
        from models.roberta_classifier import create_intent_classifier
        
        # Create classifier with Chinese configuration
        clf = create_intent_classifier()
        
        print(f"✓ Model: {clf.model_name}")
        print(f"✓ Device: {clf.device}")
        print(f"✓ Max length: {clf.max_length}")
        print(f"✓ Number of labels: {clf.num_labels}")
        print(f"✓ Intent labels: {list(clf.label_to_id.keys())}")
        
        # Test prediction with Chinese text
        test_texts = [
            "我想查看这个月的工资单",
            "想订明天两点的会议室", 
            "我需要请假三天"
        ]
        
        results = clf.predict(test_texts)
        
        print("\n🔮 Sample predictions:")
        for result in results:
            print(f"  Text: \"{result['text']}\"")
            print(f"  Intent: {result['predicted_intent']} (confidence: {result['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def test_gpu_detection():
    """Test GPU detection functionality."""
    print("\n🔧 Testing GPU Detection")
    print("-" * 40)
    
    try:
        import torch
        from train_intent import detect_device
        
        device, use_fp16 = detect_device()
        
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name()}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Selected device: {device}")
        print(f"✓ FP16 enabled: {use_fp16}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

def test_training_components():
    """Test training script components."""
    print("\n🏋️ Testing Training Components")
    print("-" * 40)
    
    try:
        from train_intent import LABEL_LIST, LABEL2ID, ID2LABEL
        
        print(f"✓ Label list: {LABEL_LIST}")
        print(f"✓ Number of labels: {len(LABEL_LIST)}")
        print(f"✓ Label mapping: {LABEL2ID}")
        
        # Check if data files exist
        data_files = ['data/train.csv', 'data/test.csv', 'data/intent_train.csv', 'data/intent_dev.csv']
        existing_files = [f for f in data_files if os.path.exists(f)]
        
        print(f"✓ Available data files: {existing_files}")
        
        if len(existing_files) >= 2:
            print("✓ Sufficient data files for training")
        else:
            print("⚠️  Some data files missing (run dataset creator)")
        
        return True
        
    except Exception as e:
        print(f"❌ Training component test failed: {e}")
        return False

def test_inference_components():
    """Test inference script components."""
    print("\n🎯 Testing Inference Components")
    print("-" * 40)
    
    try:
        from intent_infer import LABEL_LIST
        
        print(f"✓ Inference labels: {LABEL_LIST}")
        print(f"✓ Number of labels: {len(LABEL_LIST)}")
        
        # Test that we can import the classifier class
        from intent_infer import IntentClassifier
        print("✓ IntentClassifier class available")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference component test failed: {e}")
        return False

def test_file_structure():
    """Test project file structure."""
    print("\n📁 Testing File Structure")
    print("-" * 40)
    
    required_files = [
        'train_intent.py',
        'intent_infer.py',
        'dataset/dataset_creator.py',
        'models/roberta_classifier.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if not missing_files:
        print("✓ All required files present")
        return True
    else:
        print(f"❌ Missing files: {missing_files}")
        return False

def main():
    """Run comprehensive Chinese intent recognition tests."""
    print("🇨🇳 Chinese Intent Recognition System Test")
    print("=" * 50)
    print("Testing hfl/chinese-roberta-wwm-ext implementation")
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Chinese Dataset Generation", test_chinese_dataset_generation),
        ("Model Initialization", test_model_initialization),
        ("GPU Detection", test_gpu_detection),
        ("Training Components", test_training_components),
        ("Inference Components", test_inference_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
        print()
    
    print("=" * 50)
    print("🔍 TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed!")
        print("\n🚀 Next Steps:")
        print("1. Install full ML dependencies: pip install -r requirements.txt")
        print("2. Generate full dataset: python -m dataset.dataset_creator")
        print("3. Train model: python train_intent.py")
        print("4. Test inference: python intent_infer.py \"想订明天两点的会议室\"")
        
        print("\n📊 System Features:")
        print("✓ Chinese RoBERTa model (hfl/chinese-roberta-wwm-ext)")
        print("✓ GPU detection and prioritized usage")
        print("✓ 6 Chinese office domain intents")
        print("✓ Complete training and inference pipeline")
        print("✓ Chinese text processing and tokenization")
    else:
        print("❌ Some tests failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)