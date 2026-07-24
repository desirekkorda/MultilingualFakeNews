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


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: NewsRequest):

    return predict_news(request.text)


@router.get("/")
def root():

    return {
        "message": "Multilingual Fake News Detection API"
    }


@router.get("/health")
def health():

    return {
        "status": "healthy"
    }


@router.get("/info")
def info():

    return {
        "model": "XLM-RoBERTa",
        "version": "1.1.0"
    }