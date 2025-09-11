"""
Complete Demo Script for RoBERTa Intent Recognition System

This script demonstrates the complete pipeline of the intent recognition system:

1. Dataset creation with sample office domain queries
2. Model training with RoBERTa fine-tuning
3. Model evaluation with comprehensive metrics
4. Inference demonstrations with example queries

This serves as both a complete example and a way to verify that
all components of the system work together correctly.
"""

import os
import sys
from pathlib import Path
import time
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from dataset.dataset_creator import OfficeIntentDatasetCreator
from models.roberta_classifier import RoBERTaIntentClassifier
from training.train import IntentDataset, IntentTrainer, load_data
from evaluation.evaluate import IntentEvaluator
from inference.predict import IntentPredictor


class CompletePipelineDemo:
    """
    Demonstrates the complete intent recognition pipeline.
    
    This class orchestrates all components of the system to show
    how they work together from data creation to inference.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the demo with configuration.
        
        Args:
            config: Configuration dictionary for the demo
        """
        # Default configuration
        self.config = config or {
            'samples_per_intent': 50,  # Smaller dataset for demo
            'model_name': 'roberta-base',
            'num_labels': 7,
            'max_length': 128,
            'batch_size': 8,  # Smaller batch size for demo
            'learning_rate': 2e-5,
            'num_epochs': 2,  # Fewer epochs for demo
            'data_dir': 'demo_data',
            'model_dir': 'demo_models',
            'results_dir': 'demo_results'
        }
        
        # Create directories
        for dir_name in ['data_dir', 'model_dir', 'results_dir']:
            Path(self.config[dir_name]).mkdir(exist_ok=True)
        
        print("Complete Intent Recognition Pipeline Demo")
        print("=" * 50)
        print("Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print()
    
    def step_1_create_dataset(self):
        """Step 1: Create synthetic dataset for training."""
        print("STEP 1: Creating Dataset")
        print("-" * 30)
        
        # Create dataset creator
        creator = OfficeIntentDatasetCreator(
            samples_per_intent=self.config['samples_per_intent']
        )
        
        # Generate and save dataset
        creator.save_dataset(self.config['data_dir'])
        
        # Show sample data
        import pandas as pd
        train_df = pd.read_csv(f"{self.config['data_dir']}/train.csv")
        
        print(f"\nDataset created successfully!")
        print(f"Training samples: {len(train_df)}")
        print(f"Sample distribution:")
        print(train_df['label'].value_counts())
        
        print(f"\nSample training examples:")
        for intent in train_df['label'].unique()[:3]:
            sample = train_df[train_df['label'] == intent].iloc[0]
            print(f"  {intent}: \"{sample['text']}\"")
        
        return True
    
    def step_2_train_model(self):
        """Step 2: Train RoBERTa model on the dataset."""
        print(f"\nSTEP 2: Training Model")
        print("-" * 30)
        
        try:
            # Load data
            train_texts, train_labels, val_texts, val_labels = load_data(self.config['data_dir'])
            
            print(f"Loaded {len(train_texts)} training and {len(val_texts)} validation samples")
            
            # Initialize classifier
            classifier = RoBERTaIntentClassifier(
                model_name=self.config['model_name'],
                num_labels=self.config['num_labels'],
                max_length=self.config['max_length']
            )
            
            # Load label mappings
            classifier.load_label_mapping(f"{self.config['data_dir']}/label_mapping.json")
            
            # Create datasets
            print("Creating datasets...")
            train_dataset = IntentDataset(train_texts, train_labels, classifier)
            val_dataset = IntentDataset(val_texts, val_labels, classifier)
            
            # Initialize trainer
            print("Starting training...")
            trainer = IntentTrainer(
                classifier=classifier,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=self.config['batch_size'],
                learning_rate=self.config['learning_rate'],
                num_epochs=self.config['num_epochs'],
                early_stopping_patience=2
            )
            
            # Train model
            model_path = f"{self.config['model_dir']}/trained_roberta"
            history = trainer.train(model_path)
            
            print(f"Training completed!")
            print(f"Final validation accuracy: {history['val_accuracy'][-1]:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def step_3_evaluate_model(self):
        """Step 3: Evaluate the trained model."""
        print(f"\nSTEP 3: Evaluating Model")
        print("-" * 30)
        
        try:
            # Load trained model
            model_path = f"{self.config['model_dir']}/trained_roberta"
            classifier = RoBERTaIntentClassifier.load_model(model_path)
            
            # Load test data
            import pandas as pd
            test_df = pd.read_csv(f"{self.config['data_dir']}/test.csv")
            texts = test_df['text'].tolist()
            labels = test_df['label'].tolist()
            
            print(f"Evaluating on {len(texts)} test samples...")
            
            # Initialize evaluator and run evaluation
            evaluator = IntentEvaluator(classifier)
            results = evaluator.evaluate_predictions(texts, labels)
            
            # Print summary
            evaluator.print_evaluation_summary()
            
            # Save results
            evaluator.save_detailed_results(self.config['results_dir'])
            
            print(f"Evaluation completed!")
            print(f"Results saved to {self.config['results_dir']}")
            
            return True
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return False
    
    def step_4_demonstrate_inference(self):
        """Step 4: Demonstrate inference with example queries."""
        print(f"\nSTEP 4: Demonstrating Inference")
        print("-" * 30)
        
        try:
            # Load trained model
            model_path = f"{self.config['model_dir']}/trained_roberta"
            predictor = IntentPredictor(model_path)
            
            # Example queries for each intent
            example_queries = [
                ("What is my annual salary?", "salary_inquiry"),
                ("Book a conference room for Monday", "meeting_room_booking"),
                ("I need to take vacation next week", "leave_request"),
                ("Find Sarah Johnson's phone number", "directory_search"),
                ("What is the company address?", "company_info"),
                ("Who is John's manager?", "employee_info"),
                ("Search for Smith at TechCorp", "employee_search")
            ]
            
            print(f"Testing {len(example_queries)} example queries:")
            print()
            
            correct_predictions = 0
            
            for i, (query, expected_intent) in enumerate(example_queries, 1):
                result = predictor.predict_single(query, return_probabilities=False)
                
                predicted_intent = result['predicted_intent']
                confidence = result['confidence']
                is_correct = predicted_intent == expected_intent
                
                if is_correct:
                    correct_predictions += 1
                
                status = "✓" if is_correct else "✗"
                print(f"{i}. {status} \"{query}\"")
                print(f"   Expected: {expected_intent}")
                print(f"   Predicted: {predicted_intent} (confidence: {confidence:.3f})")
                print()
            
            accuracy = correct_predictions / len(example_queries)
            print(f"Demo accuracy: {correct_predictions}/{len(example_queries)} ({accuracy:.1%})")
            
            return True
            
        except Exception as e:
            print(f"Error during inference demonstration: {e}")
            return False
    
    def run_complete_demo(self):
        """Run the complete demonstration pipeline."""
        start_time = time.time()
        
        steps = [
            ("Creating Dataset", self.step_1_create_dataset),
            ("Training Model", self.step_2_train_model),
            ("Evaluating Model", self.step_3_evaluate_model),
            ("Demonstrating Inference", self.step_4_demonstrate_inference)
        ]
        
        completed_steps = 0
        
        for step_name, step_function in steps:
            try:
                print(f"\n{'='*60}")
                success = step_function()
                if success:
                    completed_steps += 1
                    print(f"✓ {step_name} completed successfully")
                else:
                    print(f"✗ {step_name} failed")
                    break
            except Exception as e:
                print(f"✗ {step_name} failed with error: {e}")
                break
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("DEMO SUMMARY")
        print(f"{'='*60}")
        print(f"Completed steps: {completed_steps}/{len(steps)}")
        print(f"Total time: {total_time:.1f} seconds")
        
        if completed_steps == len(steps):
            print("🎉 Complete demo finished successfully!")
            print(f"\nGenerated files:")
            print(f"  Data: {self.config['data_dir']}/")
            print(f"  Model: {self.config['model_dir']}/")
            print(f"  Results: {self.config['results_dir']}/")
            
            print(f"\nNext steps:")
            print(f"  - Explore evaluation results in {self.config['results_dir']}/")
            print(f"  - Try interactive prediction: python inference/predict.py --interactive --model-path {self.config['model_dir']}/trained_roberta")
            print(f"  - Use the model in your applications")
        else:
            print("❌ Demo incomplete. Check error messages above.")
        
        return completed_steps == len(steps)


def create_quick_demo_config():
    """Create a configuration for a quick demo (smaller dataset, fewer epochs)."""
    return {
        'samples_per_intent': 30,  # Very small for quick demo
        'model_name': 'roberta-base',
        'num_labels': 7,
        'max_length': 128,
        'batch_size': 4,
        'learning_rate': 2e-5,
        'num_epochs': 1,  # Just 1 epoch for quick test
        'data_dir': 'quick_demo_data',
        'model_dir': 'quick_demo_models',
        'results_dir': 'quick_demo_results'
    }


def create_full_demo_config():
    """Create a configuration for a full demo (larger dataset, proper training)."""
    return {
        'samples_per_intent': 100,
        'model_name': 'roberta-base',
        'num_labels': 7,
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'data_dir': 'data',
        'model_dir': 'models',
        'results_dir': 'evaluation_results'
    }


def main():
    """Main function for running the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete RoBERTa Intent Recognition Demo")
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick demo with smaller dataset and fewer epochs'
    )
    parser.add_argument(
        '--config',
        help='Path to custom configuration JSON file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    elif args.quick:
        config = create_quick_demo_config()
        print("Running QUICK DEMO (small dataset, 1 epoch)")
    else:
        config = create_full_demo_config()
        print("Running FULL DEMO (complete dataset, proper training)")
    
    # Run demo
    demo = CompletePipelineDemo(config)
    success = demo.run_complete_demo()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()