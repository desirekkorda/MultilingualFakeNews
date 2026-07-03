from fastapi import APIRouter

from src.inference import predict_news
from app.api.schemas import (
    NewsRequest,
    PredictionResponse
)

router = APIRouter(
    prefix="/api/v1",
    tags=["Fake News Detection"]
)

# Health Endpoint
@router.get(
    "/health",
    summary="Health Check",
    description="Returns the current health status of the API."
)
def health():

    return {
        "status": "healthy",
        "model": "loaded"
    }

# Information Endpoint, useful when deploying to cloud services
@router.get("/info")
def info():

    return {
        "project": "Multilingual Fake News Detection",

        "api_version": "1.1.0",

        "model": "xlm-roberta-base",

        "model_version": "1.0.0",

        "supported_languages": [
            "English"
        ],

        "status": "online"
    }

# Prediction Endpoint
@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Fake News",
    description="Predict whether a news article is Fake or Legit."
)
def predict(request: NewsRequest):

    return predict_news(request.text)