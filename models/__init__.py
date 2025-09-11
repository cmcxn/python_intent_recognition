"""
Models package for intent recognition.

This package contains the neural network models used for intent classification,
including the RoBERTa-based classifier implementation.
"""

from .roberta_classifier import RoBERTaIntentClassifier

__all__ = ['RoBERTaIntentClassifier']