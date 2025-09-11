"""
Inference Script for RoBERTa Intent Classification

This module provides a simple interface for using the trained intent
classification model to predict intents from text inputs. It includes:

1. Single text prediction
2. Batch text prediction
3. Confidence thresholding
4. Interactive command-line interface
5. JSON output formatting
6. Model loading and caching

This script demonstrates how to use the trained model in production
scenarios and provides examples of integrating it into applications.
"""

import torch
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
import sys

# Import our custom modules
sys.path.append(str(Path(__file__).parent.parent))
from models.roberta_classifier import RoBERTaIntentClassifier


class IntentPredictor:
    """
    Simple interface for intent prediction using trained RoBERTa model.
    
    This class provides an easy-to-use interface for making predictions
    with confidence scores and proper formatting.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = model_path
        self.classifier = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            print(f"Loading model from {self.model_path}...")
            self.classifier = RoBERTaIntentClassifier.load_model(self.model_path)
            print("Model loaded successfully!")
            
            # Print model info
            info = self.classifier.get_model_info()
            print(f"Model supports {info['num_labels']} intents:")
            for intent in info['intent_labels']:
                print(f"  - {intent}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_single(self, 
                      text: str, 
                      return_probabilities: bool = True,
                      confidence_threshold: float = 0.0) -> Dict:
        """
        Predict intent for a single text input.
        
        Args:
            text: Input text string
            return_probabilities: Whether to return class probabilities
            confidence_threshold: Minimum confidence threshold for prediction
            
        Returns:
            Prediction dictionary with intent, confidence, and probabilities
        """
        if not self.classifier:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get prediction
        result = self.classifier.predict_single(text, return_probabilities)
        
        # Check confidence threshold
        if result['confidence'] < confidence_threshold:
            result['predicted_intent'] = 'unknown'
            result['below_threshold'] = True
            result['threshold_used'] = confidence_threshold
        else:
            result['below_threshold'] = False
        
        return result
    
    def predict_batch(self, 
                     texts: List[str],
                     return_probabilities: bool = True,
                     confidence_threshold: float = 0.0) -> List[Dict]:
        """
        Predict intents for multiple text inputs.
        
        Args:
            texts: List of input text strings
            return_probabilities: Whether to return class probabilities
            confidence_threshold: Minimum confidence threshold for prediction
            
        Returns:
            List of prediction dictionaries
        """
        if not self.classifier:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get predictions
        results = self.classifier.predict(texts, return_probabilities)
        
        # Apply confidence threshold
        for result in results:
            if result['confidence'] < confidence_threshold:
                result['predicted_intent'] = 'unknown'
                result['below_threshold'] = True
                result['threshold_used'] = confidence_threshold
            else:
                result['below_threshold'] = False
        
        return results
    
    def interactive_mode(self):
        """
        Run interactive prediction mode.
        
        Allows users to input text and get real-time predictions.
        """
        print("\n" + "=" * 60)
        print("INTERACTIVE INTENT PREDICTION MODE")
        print("=" * 60)
        print("Enter text to predict intent (type 'quit' to exit)")
        print("Commands:")
        print("  'quit' or 'exit' - Exit interactive mode")
        print("  'help' - Show this help message")
        print("  'intents' - Show supported intents")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                text = input("\nEnter text: ").strip()
                
                # Handle commands
                if text.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                elif text.lower() == 'help':
                    print("\nCommands:")
                    print("  'quit' or 'exit' - Exit interactive mode")
                    print("  'help' - Show this help message")
                    print("  'intents' - Show supported intents")
                    continue
                elif text.lower() == 'intents':
                    print("\nSupported intents:")
                    for intent in self.classifier.label_to_id.keys():
                        print(f"  - {intent}")
                    continue
                elif not text:
                    print("Please enter some text.")
                    continue
                
                # Make prediction
                result = self.predict_single(text, return_probabilities=True)
                
                # Display result
                print(f"\n{'='*50}")
                print(f"Text: \"{text}\"")
                print(f"Predicted Intent: {result['predicted_intent']}")
                print(f"Confidence: {result['confidence']:.3f}")
                
                if 'probabilities' in result:
                    print(f"\nAll probabilities:")
                    sorted_probs = sorted(
                        result['probabilities'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    for intent, prob in sorted_probs:
                        print(f"  {intent}: {prob:.3f}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def predict_from_file(self, 
                         input_file: str, 
                         output_file: Optional[str] = None,
                         format_type: str = 'json') -> List[Dict]:
        """
        Predict intents for texts from a file.
        
        Args:
            input_file: Path to input file (one text per line)
            output_file: Path to output file (optional)
            format_type: Output format ('json' or 'csv')
            
        Returns:
            List of prediction results
        """
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Read {len(texts)} texts from {input_file}")
        
        # Make predictions
        results = self.predict_batch(texts, return_probabilities=True)
        
        # Save results if output file specified
        if output_file:
            if format_type.lower() == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            elif format_type.lower() == 'csv':
                import pandas as pd
                df_data = []
                for result in results:
                    row = {
                        'text': result['text'],
                        'predicted_intent': result['predicted_intent'],
                        'confidence': result['confidence']
                    }
                    # Add probability columns
                    if 'probabilities' in result:
                        for intent, prob in result['probabilities'].items():
                            row[f'prob_{intent}'] = prob
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_file, index=False)
            
            print(f"Results saved to {output_file}")
        
        return results


def create_example_queries() -> List[str]:
    """Create example queries for demonstration."""
    return [
        "What is my monthly salary?",
        "I need to book a meeting room for tomorrow",
        "Can I request vacation leave next week?",
        "Find John Smith's contact information",
        "What is the company policy on remote work?",
        "What department does Sarah work in?",
        "Search for employees named Johnson at TechCorp",
        "How much do I earn annually?",
        "Reserve a conference room for Friday afternoon",
        "I want to take sick leave tomorrow",
        "Look up the HR director's email",
        "What is our company mission statement?",
        "Who is the manager of the engineering team?",
        "Find staff named Brown at ABC Corp"
    ]


def demo_predictions(predictor: IntentPredictor):
    """Run a demonstration with example queries."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION - EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    examples = create_example_queries()
    
    print(f"Running predictions on {len(examples)} example queries...\n")
    
    results = predictor.predict_batch(examples, return_probabilities=False)
    
    # Display results in a nice format
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. Text: \"{result['text']}\"")
        print(f"    Intent: {result['predicted_intent']} (confidence: {result['confidence']:.3f})")
        if i % 5 == 0 and i < len(results):  # Add spacing every 5 items
            print()
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    intent_counts = {}
    confidence_scores = []
    
    for result in results:
        intent = result['predicted_intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        confidence_scores.append(result['confidence'])
    
    print(f"Total predictions: {len(results)}")
    print(f"Average confidence: {sum(confidence_scores) / len(confidence_scores):.3f}")
    print(f"Min confidence: {min(confidence_scores):.3f}")
    print(f"Max confidence: {max(confidence_scores):.3f}")
    
    print(f"\nIntent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="RoBERTa Intent Classification Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py "I want to book a meeting room"
  python predict.py --interactive
  python predict.py --demo
  python predict.py --file input.txt --output results.json
        """
    )
    
    # Add arguments
    parser.add_argument(
        'text',
        nargs='?',
        help='Text to predict intent for'
    )
    
    parser.add_argument(
        '--model-path',
        default='models/trained_roberta',
        help='Path to trained model directory (default: models/trained_roberta)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demonstration with example queries'
    )
    
    parser.add_argument(
        '--file', '-f',
        help='Input file with texts (one per line)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file for batch predictions'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Output format for batch predictions (default: json)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.0,
        help='Confidence threshold for predictions (default: 0.0)'
    )
    
    parser.add_argument(
        '--no-probabilities',
        action='store_true',
        help='Don\'t return class probabilities'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = IntentPredictor(args.model_path)
        
        # Handle different modes
        if args.interactive:
            predictor.interactive_mode()
        
        elif args.demo:
            demo_predictions(predictor)
        
        elif args.file:
            results = predictor.predict_from_file(
                args.file, 
                args.output, 
                args.format
            )
            
            # Print summary
            print(f"\nProcessed {len(results)} texts")
            if not args.output:
                print("\nFirst 5 predictions:")
                for i, result in enumerate(results[:5]):
                    print(f"{i+1}. \"{result['text']}\" -> {result['predicted_intent']} ({result['confidence']:.3f})")
        
        elif args.text:
            # Single prediction
            result = predictor.predict_single(
                args.text,
                return_probabilities=not args.no_probabilities,
                confidence_threshold=args.threshold
            )
            
            # Print result
            print(f"\nText: \"{result['text']}\"")
            print(f"Predicted Intent: {result['predicted_intent']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if 'below_threshold' in result and result['below_threshold']:
                print(f"Warning: Confidence below threshold ({args.threshold})")
            
            if 'probabilities' in result and not args.no_probabilities:
                print(f"\nAll probabilities:")
                sorted_probs = sorted(
                    result['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for intent, prob in sorted_probs:
                    print(f"  {intent}: {prob:.3f}")
        
        else:
            # No specific mode - show help and run demo
            print("No input provided. Running demonstration...")
            demo_predictions(predictor)
            print("\nFor interactive mode, use: python predict.py --interactive")
            print("For help, use: python predict.py --help")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()