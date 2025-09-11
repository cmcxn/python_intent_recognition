#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Intent Recognition Service

This FastAPI service provides a REST API for Chinese intent recognition
based on the intent_infer.py script. It wraps the IntentClassifier
to provide HTTP endpoints for intent prediction.

Features:
- REST API for intent prediction
- Single text prediction
- Batch text prediction
- Model information endpoint
- Health check endpoint
- Automatic model loading with fallback
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
import json
import os
from contextlib import asynccontextmanager

# Try to import the IntentClassifier, handle gracefully if model not available
try:
    from intent_infer import IntentClassifier, LABEL_LIST
    CLASSIFIER_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import IntentClassifier: {e}")
    CLASSIFIER_AVAILABLE = False
    LABEL_LIST = ["CHECK_PAYSLIP","BOOK_MEETING_ROOM","REQUEST_LEAVE",
                  "CHECK_BENEFITS","IT_TICKET","EXPENSE_REIMBURSE"]

# Global classifier instance
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup"""
    global classifier
    
    if CLASSIFIER_AVAILABLE:
        try:
            print("🚀 Starting FastAPI Intent Recognition Service...")
            print("📦 Loading intent classifier...")
            classifier = IntentClassifier()
            print("✅ Intent classifier loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not load classifier: {e}")
            print("🔄 Service will run in demo mode")
            classifier = None
    else:
        print("🔄 Running in demo mode - classifier not available")
        classifier = None
    
    yield
    
    # Cleanup (if needed)
    print("🛑 Shutting down Intent Recognition Service")

# Create FastAPI app
app = FastAPI(
    title="Chinese Intent Recognition API",
    description="FastAPI service for Chinese intent recognition using RoBERTa model",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Chinese text for intent prediction", min_length=1)

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of Chinese texts for intent prediction")

class PredictionResponse(BaseModel):
    intent: str = Field(..., description="Predicted intent label")
    confidence: float = Field(..., description="Prediction confidence score", ge=0.0, le=1.0)
    probs: Dict[str, float] = Field(..., description="Probability scores for all intents")

class BatchPredictionResponse(BaseModel):
    results: List[Dict] = Field(..., description="List of prediction results")

class ModelInfoResponse(BaseModel):
    model_path: str
    device: str
    max_length: int
    num_labels: int
    intent_labels: List[str]
    model_available: bool

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

def create_demo_prediction(text: str) -> Dict:
    """Create a demo prediction when model is not available"""
    import random
    
    # Simple keyword-based demo logic
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['会议室', '预订', '订', '会议']):
        intent = "BOOK_MEETING_ROOM"
    elif any(word in text_lower for word in ['工资', '工资单', '薪资', '薪水']):
        intent = "CHECK_PAYSLIP"
    elif any(word in text_lower for word in ['请假', '休假', '假期']):
        intent = "REQUEST_LEAVE"
    elif any(word in text_lower for word in ['福利', '保险', '公积金']):
        intent = "CHECK_BENEFITS"
    elif any(word in text_lower for word in ['电脑', '系统', 'IT', '技术', '故障']):
        intent = "IT_TICKET"
    elif any(word in text_lower for word in ['报销', '费用', '发票']):
        intent = "EXPENSE_REIMBURSE"
    else:
        intent = random.choice(LABEL_LIST)
    
    # Generate demo probabilities
    probs = {}
    base_prob = 0.1 / (len(LABEL_LIST) - 1)
    
    for label in LABEL_LIST:
        if label == intent:
            probs[label] = round(random.uniform(0.7, 0.95), 4)
        else:
            probs[label] = round(random.uniform(0.01, base_prob), 4)
    
    # Normalize probabilities
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}
    
    return {
        "intent": intent,
        "confidence": probs[intent],
        "probs": probs
    }

@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with basic service information"""
    return {
        "message": "Chinese Intent Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "health": "/health"
        }
    }

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=classifier is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse, summary="Predict intent for single text")
async def predict_intent(request: PredictionRequest):
    """
    Predict intent for a single Chinese text input.
    
    Returns the predicted intent with confidence scores.
    """
    try:
        if classifier is not None:
            # Use real classifier
            result = classifier.predict(request.text)
        else:
            # Use demo mode
            result = create_demo_prediction(request.text)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Predict intent for multiple texts")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict intents for multiple Chinese text inputs.
    
    Returns a list of predictions with confidence scores.
    """
    try:
        results = []
        
        for text in request.texts:
            if classifier is not None:
                # Use real classifier
                result = classifier.predict(text)
                result["text"] = text
            else:
                # Use demo mode
                result = create_demo_prediction(text)
                result["text"] = text
            
            results.append(result)
        
        return BatchPredictionResponse(results=results)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info", response_model=ModelInfoResponse, summary="Get model information")
async def get_model_info():
    """
    Get information about the loaded model.
    """
    try:
        if classifier is not None:
            info = classifier.get_model_info()
            info["model_available"] = True
            return ModelInfoResponse(**info)
        else:
            # Return demo info when model not available
            return ModelInfoResponse(
                model_path="models/intent_roberta",
                device="cpu",
                max_length=64,
                num_labels=len(LABEL_LIST),
                intent_labels=LABEL_LIST,
                model_available=False
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not get model info: {str(e)}"
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chinese Intent Recognition FastAPI Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "intent_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )