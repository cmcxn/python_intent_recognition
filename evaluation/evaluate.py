"""
Evaluation Module for RoBERTa Intent Classification

This module provides comprehensive evaluation capabilities for the trained
intent classification model. It includes:

1. Standard classification metrics (accuracy, precision, recall, F1)
2. Per-class performance analysis
3. Confusion matrix visualization
4. Detailed prediction analysis
5. Error analysis and insights
6. Model performance reporting

The evaluation helps understand model strengths and weaknesses,
providing insights for model improvement and deployment decisions.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.roberta_classifier import RoBERTaIntentClassifier


class IntentEvaluator:
    """
    Comprehensive evaluator for intent classification models.
    
    This class provides detailed analysis of model performance including
    standard metrics, confusion matrices, and error analysis.
    """
    
    def __init__(self, classifier: RoBERTaIntentClassifier):
        """
        Initialize the evaluator.
        
        Args:
            classifier: Trained RoBERTaIntentClassifier instance
        """
        self.classifier = classifier
        self.results = {}
        
    def evaluate_predictions(self, 
                           texts: List[str], 
                           true_labels: List[str],
                           batch_size: int = 32) -> Dict:
        """
        Evaluate model predictions on a dataset.
        
        Args:
            texts: List of input texts
            true_labels: List of true intent labels
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary containing evaluation results
        """
        print("Evaluating model predictions...")
        
        # Get predictions in batches to manage memory
        all_predictions = []
        all_probabilities = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = self.classifier.predict(batch_texts, return_probabilities=True)
            
            for pred in batch_predictions:
                all_predictions.append(pred['predicted_intent'])
                all_probabilities.append(pred['probabilities'])
        
        # Calculate standard metrics
        accuracy = accuracy_score(true_labels, all_predictions)
        precision = precision_score(true_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, all_predictions, average='weighted', zero_division=0)
        
        # Get detailed classification report
        class_report = classification_report(
            true_labels, all_predictions, 
            target_names=self.classifier.label_to_id.keys(),
            output_dict=True,
            zero_division=0
        )
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(
            true_labels, all_predictions,
            labels=list(self.classifier.label_to_id.keys())
        )
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'true_labels': true_labels,
            'predicted_labels': all_predictions,
            'probabilities': all_probabilities,
            'texts': texts
        }
        
        return self.results
    
    def print_evaluation_summary(self):
        """Print a summary of evaluation results."""
        if not self.results:
            print("No evaluation results available. Run evaluate_predictions first.")
            return
        
        print("=" * 60)
        print("INTENT CLASSIFICATION EVALUATION RESULTS")
        print("=" * 60)
        
        # Overall metrics
        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {self.results['accuracy']:.4f}")
        print(f"  Precision: {self.results['precision']:.4f}")
        print(f"  Recall:    {self.results['recall']:.4f}")
        print(f"  F1-Score:  {self.results['f1_score']:.4f}")
        
        # Per-class metrics
        print(f"\nPer-Class Performance:")
        print("-" * 80)
        print(f"{'Intent':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 80)
        
        for intent in self.classifier.label_to_id.keys():
            if intent in self.results['classification_report']:
                metrics = self.results['classification_report'][intent]
                print(f"{intent:<20} {metrics['precision']:<10.3f} "
                      f"{metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} "
                      f"{int(metrics['support']):<10}")
        
        # Overall averages
        print("-" * 80)
        weighted_avg = self.results['classification_report']['weighted avg']
        print(f"{'Weighted Avg':<20} {weighted_avg['precision']:<10.3f} "
              f"{weighted_avg['recall']:<10.3f} {weighted_avg['f1-score']:<10.3f} "
              f"{int(weighted_avg['support']):<10}")
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot confusion matrix with nice formatting.
        
        Args:
            save_path: Path to save the plot (optional)
            figsize: Figure size for the plot
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_predictions first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Create confusion matrix heatmap
        labels = list(self.classifier.label_to_id.keys())
        
        # Normalize confusion matrix for better visualization
        cm_normalized = self.results['confusion_matrix'].astype('float') / \
                       self.results['confusion_matrix'].sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        
        plt.title('Confusion Matrix - Intent Classification\n(Normalized by True Label)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Intent', fontsize=12)
        plt.ylabel('True Intent', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot per-class performance metrics.
        
        Args:
            save_path: Path to save the plot (optional)
            figsize: Figure size for the plot
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_predictions first.")
            return
        
        # Extract per-class metrics
        intents = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for intent in self.classifier.label_to_id.keys():
            if intent in self.results['classification_report']:
                metrics = self.results['classification_report'][intent]
                intents.append(intent.replace('_', ' ').title())
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1-score'])
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Precision and Recall
        x = np.arange(len(intents))
        width = 0.35
        
        ax1.bar(x - width/2, precisions, width, label='Precision', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, recalls, width, label='Recall', alpha=0.8, color='lightcoral')
        
        ax1.set_title('Precision and Recall by Intent', fontweight='bold')
        ax1.set_xlabel('Intent')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(intents, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: F1-Score
        bars = ax2.bar(intents, f1_scores, alpha=0.8, color='lightgreen')
        ax2.set_title('F1-Score by Intent', fontweight='bold')
        ax2.set_xlabel('Intent')
        ax2.set_ylabel('F1-Score')
        ax2.set_xticklabels(intents, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to {save_path}")
        
        plt.show()
    
    def analyze_errors(self, top_k: int = 10) -> Dict:
        """
        Analyze prediction errors to understand model weaknesses.
        
        Args:
            top_k: Number of top errors to display
            
        Returns:
            Dictionary containing error analysis results
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_predictions first.")
            return {}
        
        # Find incorrectly predicted samples
        errors = []
        for i, (true_label, pred_label, text, probs) in enumerate(zip(
            self.results['true_labels'],
            self.results['predicted_labels'],
            self.results['texts'],
            self.results['probabilities']
        )):
            if true_label != pred_label:
                confidence = probs[pred_label]
                true_confidence = probs[true_label]
                
                errors.append({
                    'index': i,
                    'text': text,
                    'true_intent': true_label,
                    'predicted_intent': pred_label,
                    'predicted_confidence': confidence,
                    'true_confidence': true_confidence,
                    'confidence_diff': confidence - true_confidence
                })
        
        # Sort by confidence (highest confidence errors first)
        errors.sort(key=lambda x: x['predicted_confidence'], reverse=True)
        
        print(f"\nError Analysis: {len(errors)} misclassified samples")
        print("=" * 80)
        print(f"Top {min(top_k, len(errors))} High-Confidence Errors:")
        print("-" * 80)
        
        for i, error in enumerate(errors[:top_k]):
            print(f"\n{i+1}. Text: \"{error['text']}\"")
            print(f"   True Intent: {error['true_intent']}")
            print(f"   Predicted: {error['predicted_intent']} (confidence: {error['predicted_confidence']:.3f})")
            print(f"   True confidence: {error['true_confidence']:.3f}")
        
        # Error pattern analysis
        error_patterns = {}
        for error in errors:
            pattern = f"{error['true_intent']} -> {error['predicted_intent']}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        print(f"\nMost Common Error Patterns:")
        print("-" * 40)
        for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pattern}: {count} times")
        
        return {
            'errors': errors,
            'error_patterns': error_patterns,
            'total_errors': len(errors),
            'error_rate': len(errors) / len(self.results['true_labels'])
        }
    
    def save_detailed_results(self, save_path: str):
        """
        Save detailed evaluation results to files.
        
        Args:
            save_path: Directory path to save results
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_predictions first.")
            return
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation metrics
        metrics = {
            'overall_metrics': {
                'accuracy': float(self.results['accuracy']),
                'precision': float(self.results['precision']),
                'recall': float(self.results['recall']),
                'f1_score': float(self.results['f1_score'])
            },
            'per_class_metrics': {}
        }
        
        # Extract per-class metrics
        for intent in self.classifier.label_to_id.keys():
            if intent in self.results['classification_report']:
                metrics['per_class_metrics'][intent] = self.results['classification_report'][intent]
        
        # Save metrics as JSON
        with open(save_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'text': self.results['texts'],
            'true_label': self.results['true_labels'],
            'predicted_label': self.results['predicted_labels'],
            'correct': [t == p for t, p in zip(self.results['true_labels'], self.results['predicted_labels'])]
        })
        
        # Add confidence scores
        for intent in self.classifier.label_to_id.keys():
            predictions_df[f'prob_{intent}'] = [probs[intent] for probs in self.results['probabilities']]
        
        predictions_df.to_csv(save_dir / 'detailed_predictions.csv', index=False)
        
        # Save confusion matrix
        conf_matrix_df = pd.DataFrame(
            self.results['confusion_matrix'],
            index=list(self.classifier.label_to_id.keys()),
            columns=list(self.classifier.label_to_id.keys())
        )
        conf_matrix_df.to_csv(save_dir / 'confusion_matrix.csv')
        
        print(f"Detailed results saved to {save_dir}")


def evaluate_model(model_path: str, test_data_path: str, output_path: str = "evaluation_results"):
    """
    Complete evaluation pipeline for a trained model.
    
    Args:
        model_path: Path to trained model directory
        test_data_path: Path to test data CSV file
        output_path: Path to save evaluation results
    """
    print("Loading model and data...")
    
    # Load trained model
    classifier = RoBERTaIntentClassifier.load_model(model_path)
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    texts = test_df['text'].tolist()
    labels = test_df['label'].tolist()
    
    print(f"Loaded {len(texts)} test samples")
    
    # Initialize evaluator
    evaluator = IntentEvaluator(classifier)
    
    # Run evaluation
    results = evaluator.evaluate_predictions(texts, labels)
    
    # Print summary
    evaluator.print_evaluation_summary()
    
    # Generate plots
    evaluator.plot_confusion_matrix(f"{output_path}/confusion_matrix.png")
    evaluator.plot_per_class_metrics(f"{output_path}/per_class_metrics.png")
    
    # Analyze errors
    evaluator.analyze_errors()
    
    # Save detailed results
    evaluator.save_detailed_results(output_path)
    
    return evaluator


def main():
    """Main evaluation function."""
    print("RoBERTa Intent Classification Evaluation")
    print("=" * 45)
    
    # Configuration
    config = {
        'model_path': 'models/trained_roberta',
        'test_data_path': 'data/test.csv',
        'output_path': 'evaluation_results'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Run complete evaluation
        evaluator = evaluate_model(
            model_path=config['model_path'],
            test_data_path=config['test_data_path'],
            output_path=config['output_path']
        )
        
        print("\n" + "=" * 45)
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()