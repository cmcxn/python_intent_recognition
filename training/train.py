"""
Training Script for RoBERTa Intent Classification

This module implements the training pipeline for fine-tuning RoBERTa
on office domain intent classification. It includes:

1. Data loading and preprocessing
2. Model configuration and initialization  
3. Training loop with validation
4. Learning rate scheduling
5. Early stopping
6. Model checkpointing
7. Performance monitoring

The training process follows best practices for fine-tuning transformer models
on classification tasks, with proper regularization and optimization strategies.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.roberta_classifier import RoBERTaIntentClassifier


class IntentDataset(Dataset):
    """
    PyTorch Dataset for intent classification.
    
    This dataset class handles loading and preprocessing of intent data
    for training and evaluation.
    """
    
    def __init__(self, texts: List[str], labels: List[str], classifier: RoBERTaIntentClassifier):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input text strings
            labels: List of intent labels
            classifier: RoBERTaIntentClassifier instance for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.classifier = classifier
        
        # Pre-tokenize all texts for efficiency
        print("Tokenizing texts...")
        self.encoded_data = self.classifier.tokenize_texts(texts)
        self.label_ids = self.classifier.encode_labels(labels)
        
        print(f"Dataset initialized with {len(texts)} samples")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encoded_data['input_ids'][idx],
            'attention_mask': self.encoded_data['attention_mask'][idx],
            'labels': self.label_ids[idx]
        }


class IntentTrainer:
    """
    Trainer class for RoBERTa intent classification.
    
    This class handles the complete training pipeline including:
    - Model optimization
    - Learning rate scheduling  
    - Validation monitoring
    - Early stopping
    - Model checkpointing
    """
    
    def __init__(self, 
                 classifier: RoBERTaIntentClassifier,
                 train_dataset: IntentDataset,
                 val_dataset: IntentDataset,
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 3,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.01,
                 early_stopping_patience: int = 3):
        """
        Initialize the trainer.
        
        Args:
            classifier: RoBERTaIntentClassifier instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate schedule
            weight_decay: Weight decay for regularization
            early_stopping_patience: Patience for early stopping
        """
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.classifier.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Calculate total training steps
        self.total_steps = len(self.train_loader) * num_epochs
        
        # Initialize learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        print(f"Trainer initialized:")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Total training steps: {self.total_steps}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Validation batches: {len(self.val_loader)}")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.classifier.model.train()
        total_loss = 0.0
        
        # Progress bar for training
        train_pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in train_pbar:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.classifier.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.classifier.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.classifier.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            val_pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for batch in val_pbar:
                # Forward pass
                outputs = self.classifier.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # Calculate loss
                loss = outputs.loss
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += batch['labels'].size(0)
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, save_path: str = "models/trained_roberta"):
        """
        Execute the complete training pipeline.
        
        Args:
            save_path: Path to save the best model
        """
        print("Starting training...")
        print("=" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 30)
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model state
                self.best_model_state = self.classifier.model.state_dict().copy()
                print("✓ New best model!")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                if self.patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered!")
                    break
        
        # Load best model state
        if self.best_model_state is not None:
            self.classifier.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights")
        
        # Save final model
        self.classifier.save_model(save_path)
        
        # Training completed
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training curves
        self.plot_training_curves(save_path)
        
        return self.history
    
    def plot_training_curves(self, save_path: str):
        """
        Plot and save training curves.
        
        Args:
            save_path: Directory to save plots
        """
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Training Curves', fontsize=16)
            
            epochs = range(1, len(self.history['train_loss']) + 1)
            
            # Plot training and validation loss
            axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
            axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot validation accuracy
            axes[0, 1].plot(epochs, self.history['val_accuracy'], 'g-', label='Validation Accuracy')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot learning rate
            axes[1, 0].plot(epochs, self.history['learning_rates'], 'm-', label='Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            axes[1, 0].set_yscale('log')
            
            # Plot loss comparison
            axes[1, 1].plot(epochs, self.history['train_loss'], 'b-', label='Train')
            axes[1, 1].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
            axes[1, 1].set_title('Loss Comparison')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Training curves saved to {save_dir / 'training_curves.png'}")
            
        except Exception as e:
            print(f"Could not save training curves: {e}")


def load_data(data_dir: str = "data") -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load training and validation data.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels)
    """
    data_path = Path(data_dir)
    
    # Load training data
    train_df = pd.read_csv(data_path / "train.csv")
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    
    # Load test data as validation
    test_df = pd.read_csv(data_path / "test.csv")
    val_texts = test_df['text'].tolist()
    val_labels = test_df['label'].tolist()
    
    print(f"Loaded {len(train_texts)} training samples")
    print(f"Loaded {len(val_texts)} validation samples")
    
    return train_texts, train_labels, val_texts, val_labels


def main():
    """Main training function."""
    print("RoBERTa Intent Classification Training")
    print("=" * 40)
    
    # Configuration
    config = {
        'data_dir': 'data',
        'model_name': 'roberta-base',
        'num_labels': 7,
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'early_stopping_patience': 3,
        'save_path': 'models/trained_roberta'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Load data
        print("\nLoading data...")
        train_texts, train_labels, val_texts, val_labels = load_data(config['data_dir'])
        
        # Initialize classifier
        print("\nInitializing model...")
        classifier = RoBERTaIntentClassifier(
            model_name=config['model_name'],
            num_labels=config['num_labels'],
            max_length=config['max_length']
        )
        
        # Load label mappings
        classifier.load_label_mapping(f"{config['data_dir']}/label_mapping.json")
        
        # Create datasets
        print("\nCreating datasets...")
        train_dataset = IntentDataset(train_texts, train_labels, classifier)
        val_dataset = IntentDataset(val_texts, val_labels, classifier)
        
        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = IntentTrainer(
            classifier=classifier,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_epochs=config['num_epochs'],
            warmup_steps=config['warmup_steps'],
            weight_decay=config['weight_decay'],
            early_stopping_patience=config['early_stopping_patience']
        )
        
        # Start training
        history = trainer.train(config['save_path'])
        
        print("\n" + "=" * 40)
        print("Training completed successfully!")
        print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        print(f"Best validation loss: {min(history['val_loss']):.4f}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()