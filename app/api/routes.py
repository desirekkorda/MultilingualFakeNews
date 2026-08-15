from fastapi import APIRouter

from src.inference import predict_news, load_model

from app.api.schemas import (
    NewsRequest,
    PredictionResponse,
)

router = APIRouter(
    prefix="/api/v1",
    tags=["Fake News Detection"],
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: NewsRequest):
    """
    Classify submitted news text as Legit or Fake.
    """
    return predict_news(request.text)


@router.get("/")
def root():
    """
    API root endpoint.
    """
    return {
        "message": "Multilingual Fake News Detection API",
        "version": "3.0.0",
    }


@router.get("/health", status_code=200)
def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "Multilingual Fake News Detection API",
    }


@router.get("/info")
def info():
    """
    Model and service information.
    """
    return {
        "model": "XLM-RoBERTa-base",
        "version": "3.0.0",
        "format": "Quantized ONNX",
        "quantization": "Dynamic INT8",
        "languages": [
            "English",
            "Hindi",
            "Indonesian",
            "Swahili",
            "Vietnamese",
        ],
    }


# ... existing router ...

@router.get("/debug/model-load")
def debug_model_load():
    """
    Explicitly test production model initialization.
    """
    load_model()

    return {
        "status": "loaded",
        "message": "Tokenizer and ONNX model loaded successfully."
    }


@router.get("/debug/model-load")
def debug_model_load():
    load_model()

    return {
        "status": "loaded",
        "message": "Tokenizer and ONNX model loaded successfully."
    }