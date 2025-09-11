# RoBERTa Intent Recognition for Office Domain

This module provides a complete implementation of intent recognition for office domain tasks using RoBERTa (Robustly Optimized BERT Pretraining Approach).

## Supported Intents

The model can identify the following office domain intents:

1. **salary_inquiry** - Checking salary information
2. **meeting_room_booking** - Booking meeting rooms
3. **leave_request** - Requesting leave/time off
4. **directory_search** - Searching company directory
5. **company_info** - Querying company information
6. **employee_info** - Querying employee information
7. **employee_search** - Finding employees by company abbreviation and name

## Project Structure

```
python_intent_recognition/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── dataset/
│   ├── __init__.py
│   └── dataset_creator.py    # Sample dataset generation
├── models/
│   ├── __init__.py
│   └── roberta_classifier.py # RoBERTa model implementation
├── training/
│   ├── __init__.py
│   └── train.py             # Training script
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py          # Evaluation module
├── inference/
│   ├── __init__.py
│   └── predict.py           # Inference script
└── examples/
    ├── __init__.py
    └── demo.py              # Complete pipeline demonstration
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate sample dataset:
   ```bash
   python -m dataset.dataset_creator
   ```

3. Train the model:
   ```bash
   python -m training.train
   ```

4. Evaluate the model:
   ```bash
   python -m evaluation.evaluate
   ```

5. Run inference:
   ```bash
   python -m inference.predict "I want to book a meeting room for tomorrow"
   ```

6. Run complete demo:
   ```bash
   python -m examples.demo
   ```

## Implementation Details

### Dataset Creation
- Generates synthetic training data for each intent class
- Includes diverse phrasings and vocabulary for robust training
- Creates balanced dataset with equal samples per intent

### Model Architecture
- Uses pre-trained RoBERTa-base as the foundation
- Adds a classification head for intent prediction
- Implements proper tokenization and preprocessing

### Training Process
- Fine-tunes RoBERTa on the intent classification task
- Uses appropriate learning rate and training strategies
- Includes validation and early stopping

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion matrix for detailed analysis
- Per-class performance metrics

### Inference
- Simple API for predicting intents from text
- Confidence scores for predictions
- Batch processing capability

## Educational Value

This implementation includes extensive comments and documentation to help understand:
- How to fine-tune pre-trained language models
- Intent classification system design
- PyTorch/Transformers best practices
- Model evaluation and validation techniques