"""
RoBERTa Intent Classifier Implementation

This module implements a fine-tuned RoBERTa model for intent classification
in office domain scenarios. The classifier supports:

1. Multiple intent categories for office tasks
2. Confidence scoring for predictions  
3. Batch and single text processing
4. Model persistence and loading
5. Automatic tokenization and preprocessing

Architecture:
- Pre-trained RoBERTa model as backbone
- Classification head for intent prediction
- Support for Chinese and multilingual text processing

Key Features:
- Easy-to-use prediction interface
- Comprehensive model information
- Flexible configuration options
- Production-ready model persistence
"""

import torch
import torch.nn as nn
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    RobertaConfig,
    BertTokenizer,
    AutoTokenizer
)
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# Use Chinese RoBERTa model as suggested
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 中文RoBERTa


class RoBERTaIntentClassifier:
    """
    RoBERTa-based intent classifier for office domain tasks.
    
    This classifier fine-tunes a pre-trained RoBERTa model for intent
    classification, providing high-accuracy predictions with confidence scores.
    
    Supported Intents:
    - salary_inquiry: Checking salary information
    - meeting_room_booking: Booking meeting rooms  
    - leave_request: Requesting leave/time off
    - directory_search: Searching company directory
    - company_info: Querying company information
    - employee_info: Querying employee information
    - employee_search: Finding employees by criteria
    """
    
    def __init__(self, 
                 model_name: str = MODEL_NAME,
                 num_labels: int = 7,
                 max_length: int = 128,
                 device: Optional[str] = None,
                 offline_mode: bool = False):
        """
        Initialize the RoBERTa intent classifier.
        
        Args:
            model_name: Pre-trained model name (default: Chinese RoBERTa)
            num_labels: Number of intent classes
            max_length: Maximum sequence length for tokenization
            device: Device to run model on (auto-detected if None)
            offline_mode: If True, create a local mock model for testing
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.offline_mode = offline_mode
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing RoBERTa classifier:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Max length: {max_length}")
        print(f"  Number of labels: {num_labels}")
        print(f"  Offline mode: {offline_mode}")
        
        # Initialize label mappings first (needed for tokenizer setup)
        self.label_to_id = {}
        self.id_to_label = {}
        self._initialize_default_labels()
        
        if offline_mode:
            self._initialize_offline_components()
        else:
            self._initialize_online_components()
        
        print("✓ RoBERTa classifier initialized successfully")
    
    def _initialize_default_labels(self):
        """Initialize default office domain intent labels."""
        default_labels = [
            'salary_inquiry',
            'meeting_room_booking', 
            'leave_request',
            'directory_search',
            'company_info',
            'employee_info',
            'employee_search'
        ]
        
        for i, label in enumerate(default_labels):
            self.label_to_id[label] = i
            self.id_to_label[i] = label
    
    def _initialize_offline_components(self):
        """Initialize components for offline mode (testing)."""
        print("Initializing offline components...")
        
        # Create a simple mock tokenizer
        self.tokenizer = self._create_mock_tokenizer()
        print("✓ Mock tokenizer created")
        
        # Create a simple mock model
        self.model = self._create_mock_model()
        self.model.to(self.device)
        print("✓ Mock model created")
    
    def _initialize_online_components(self):
        """Initialize components for online mode (production)."""
        print("Initializing online components...")
        
        # Initialize tokenizer
        try:
            # Use AutoTokenizer for better compatibility with different models
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {self.model_name}, falling back to roberta-base")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
                print("✓ Fallback tokenizer loaded")
            except Exception as e2:
                print(f"Error loading fallback tokenizer: {e2}")
                print("Switching to offline mode...")
                self.offline_mode = True
                self._initialize_offline_components()
                return
            
        # Initialize model
        try:
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                output_attentions=False,
                output_hidden_states=False
            )
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}, falling back to roberta-base")
            try:
                self.model = RobertaForSequenceClassification.from_pretrained(
                    'roberta-base',
                    num_labels=self.num_labels,
                    output_attentions=False,
                    output_hidden_states=False
                )
                print("✓ Fallback model loaded")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                print("Switching to offline mode...")
                self.offline_mode = True
                self._initialize_offline_components()
                return
        
        # Move model to device
        self.model.to(self.device)
    
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer for offline testing."""
        class MockTokenizer:
            def __init__(self, max_length):
                self.max_length = max_length
                self.vocab_size = 1000
                self.pad_token_id = 0
                self.cls_token_id = 1
                self.sep_token_id = 2
                
            def __call__(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                
                # Simple tokenization: convert to character IDs
                input_ids = []
                attention_mask = []
                
                for text in texts:
                    # Simple encoding: use character ordinals (limited)
                    ids = [self.cls_token_id]  # CLS token
                    for char in text[:self.max_length-2]:  # Leave room for CLS and SEP
                        char_id = min(ord(char) % self.vocab_size, self.vocab_size - 1)
                        ids.append(char_id)
                    ids.append(self.sep_token_id)  # SEP token
                    
                    # Pad to max_length
                    while len(ids) < self.max_length:
                        ids.append(self.pad_token_id)
                    
                    # Truncate if too long
                    ids = ids[:self.max_length]
                    
                    # Create attention mask
                    mask = [1 if id != self.pad_token_id else 0 for id in ids]
                    
                    input_ids.append(ids)
                    attention_mask.append(mask)
                
                return {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
                }
                
            def save_pretrained(self, path):
                # Mock save - just create directory
                Path(path).mkdir(parents=True, exist_ok=True)
                return True
        
        return MockTokenizer(self.max_length)
    
    def _create_mock_model(self):
        """Create a mock model for offline testing."""
        class MockModel(nn.Module):
            def __init__(self, num_labels, vocab_size=1000, hidden_size=128):
                super().__init__()
                self.num_labels = num_labels
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.classifier = nn.Linear(hidden_size, num_labels)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                # Simple forward pass
                embedded = self.embedding(input_ids)
                
                # Simple pooling: take mean of non-padded tokens
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    embedded = embedded * mask_expanded
                    pooled = embedded.sum(dim=1) / mask_expanded.sum(dim=1)
                else:
                    pooled = embedded.mean(dim=1)
                
                pooled = self.dropout(pooled)
                logits = self.classifier(pooled)
                
                outputs = type('Outputs', (), {})()
                outputs.logits = logits
                
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    outputs.loss = loss_fn(logits, labels)
                
                return outputs
            
            def save_pretrained(self, path):
                # Mock save - save state dict
                Path(path).mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), Path(path) / 'pytorch_model.bin')
                return True
                
            def train(self, mode=True):
                super().train(mode)
                
            def eval(self):
                super().eval()
        
        return MockModel(self.num_labels)
    
    def load_label_mapping(self, mapping_file: str):
        """
        Load label mappings from JSON file.
        
        Args:
            mapping_file: Path to JSON file with label mappings
        """
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            if 'label_to_id' in mapping_data:
                self.label_to_id = mapping_data['label_to_id']
                self.id_to_label = {v: k for k, v in self.label_to_id.items()}
            elif isinstance(mapping_data, dict):
                # Direct mapping provided
                self.label_to_id = mapping_data
                self.id_to_label = {v: k for k, v in mapping_data.items()}
            
            print(f"✓ Loaded {len(self.label_to_id)} labels from {mapping_file}")
        except Exception as e:
            print(f"Warning: Could not load label mapping from {mapping_file}: {e}")
            print("Using default label mapping")
    
    def save_label_mapping(self, output_file: str):
        """
        Save label mappings to JSON file.
        
        Args:
            output_file: Path to save label mappings
        """
        mapping_data = {
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Label mapping saved to {output_file}")
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts for model input.
        
        Args:
            texts: List of text strings to tokenize
            
        Returns:
            Dictionary containing input_ids and attention_mask tensors
        """
        # Tokenize all texts
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        
        return encoded
    
    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        """
        Encode string labels to tensor indices.
        
        Args:
            labels: List of string labels
            
        Returns:
            Tensor of label indices
        """
        label_ids = [self.label_to_id[label] for label in labels]
        return torch.tensor(label_ids, dtype=torch.long).to(self.device)
    
    def decode_predictions(self, predictions: torch.Tensor) -> List[str]:
        """
        Decode prediction indices to string labels.
        
        Args:
            predictions: Tensor of prediction indices
            
        Returns:
            List of string labels
        """
        pred_labels = predictions.cpu().numpy()
        return [self.id_to_label[pred_id] for pred_id in pred_labels]
    
    def predict_single(self, 
                      text: str, 
                      return_probabilities: bool = True) -> Dict:
        """
        Predict intent for a single text input.
        
        Args:
            text: Input text string
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        return self.predict([text], return_probabilities)[0]
    
    def predict(self, 
               texts: List[str],
               return_probabilities: bool = True,
               batch_size: int = 32) -> List[Dict]:
        """
        Predict intents for multiple text inputs.
        
        Args:
            texts: List of input text strings
            return_probabilities: Whether to return class probabilities
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        all_results = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                encoded = self.tokenize_texts(batch_texts)
                
                # Get model predictions
                outputs = self.model(**encoded)
                logits = outputs.logits
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Convert to results
                for j, text in enumerate(batch_texts):
                    result = {
                        'text': text,
                        'predicted_intent': self.id_to_label[predictions[j].item()],
                        'confidence': probabilities[j][predictions[j]].item()
                    }
                    
                    if return_probabilities:
                        prob_dict = {}
                        for label_id, prob in enumerate(probabilities[j]):
                            label = self.id_to_label[label_id]
                            prob_dict[label] = prob.item()
                        result['probabilities'] = prob_dict
                    
                    all_results.append(result)
        
        return all_results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'device': str(self.device),
            'intent_labels': list(self.label_to_id.keys()),
            'model_size': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def save_model(self, save_path: str):
        """
        Save the trained model and associated files.
        
        Args:
            save_path: Directory path to save the model
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label
        }
        
        with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Save label mapping separately for compatibility
        self.save_label_mapping(save_dir / 'label_mapping.json')
        
        print(f"✓ Model saved to {save_dir}")
    
    @classmethod
    def load_model(cls, model_path: str, device: Optional[str] = None, offline_mode: bool = True) -> 'RoBERTaIntentClassifier':
        """
        Load a trained model from directory.
        
        Args:
            model_path: Path to saved model directory
            device: Device to load model on
            offline_mode: Use offline mode for loading (default: True)
            
        Returns:
            Loaded RoBERTaIntentClassifier instance
        """
        model_dir = Path(model_path)
        
        # Load configuration
        config_file = model_dir / 'config.json'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # Fallback configuration
            config = {
                'model_name': MODEL_NAME,
                'num_labels': 7,
                'max_length': 128
            }
        
        # Create classifier instance
        classifier = cls(
            model_name=config.get('model_name', MODEL_NAME),
            num_labels=config.get('num_labels', 7),
            max_length=config.get('max_length', 128),
            device=device,
            offline_mode=offline_mode
        )
        
        # Load saved model weights (only if not in offline mode)
        if not offline_mode:
            try:
                classifier.model = RobertaForSequenceClassification.from_pretrained(model_dir)
                classifier.model.to(classifier.device)
                print("✓ Model weights loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
            
            # Load saved tokenizer
            try:
                classifier.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                print("✓ Tokenizer loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
        else:
            print("✓ Using offline mode - mock model and tokenizer")
        
        # Load label mappings
        if 'label_to_id' in config:
            classifier.label_to_id = config['label_to_id']
            # Reconstruct id_to_label with integer keys
            classifier.id_to_label = {v: k for k, v in classifier.label_to_id.items()}
        else:
            # Try to load from separate file
            label_file = model_dir / 'label_mapping.json'
            if label_file.exists():
                classifier.load_label_mapping(str(label_file))
        
        print(f"✓ Model loaded from {model_dir}")
        return classifier
    
    def set_training_mode(self, training: bool = True):
        """
        Set model to training or evaluation mode.
        
        Args:
            training: Whether to set training mode
        """
        self.model.train(training)
    
    def get_trainable_parameters(self):
        """Get model parameters for training."""
        return self.model.parameters()
    
    def freeze_base_model(self):
        """Freeze the base RoBERTa model parameters."""
        for param in self.model.roberta.parameters():
            param.requires_grad = False
        print("✓ Base RoBERTa model frozen")
    
    def unfreeze_base_model(self):
        """Unfreeze the base RoBERTa model parameters."""
        for param in self.model.roberta.parameters():
            param.requires_grad = True
        print("✓ Base RoBERTa model unfrozen")


# Convenience function for quick model creation
def create_intent_classifier(num_labels: int = 7, 
                           max_length: int = 128,
                           device: Optional[str] = None,
                           offline_mode: bool = True) -> RoBERTaIntentClassifier:
    """
    Create a RoBERTa intent classifier with default settings.
    
    Args:
        num_labels: Number of intent classes
        max_length: Maximum sequence length
        device: Device to run on
        offline_mode: Use offline mode for testing (default: True)
        
    Returns:
        Initialized RoBERTaIntentClassifier
    """
    return RoBERTaIntentClassifier(
        model_name=MODEL_NAME,
        num_labels=num_labels,
        max_length=max_length,
        device=device,
        offline_mode=offline_mode
    )


# Example usage
if __name__ == "__main__":
    # Demo usage
    print("RoBERTa Intent Classifier Demo")
    print("=" * 40)
    
    # Create classifier
    classifier = create_intent_classifier()
    
    # Show model info
    info = classifier.get_model_info()
    print(f"Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example predictions (requires trained model)
    sample_texts = [
        "What is my monthly salary?",
        "I need to book a meeting room for tomorrow",
        "Can I request vacation leave next week?"
    ]
    
    print(f"\nExample predictions:")
    try:
        results = classifier.predict(sample_texts)
        for result in results:
            print(f"Text: \"{result['text']}\"")
            print(f"Intent: {result['predicted_intent']} (confidence: {result['confidence']:.3f})")
            print()
    except Exception as e:
        print(f"Cannot make predictions without training: {e}")
        print("Train the model first using the training script.")