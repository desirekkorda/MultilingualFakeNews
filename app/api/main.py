from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


app = FastAPI(
    title="Multilingual Fake News Detection API",
    version="3.0.0",
    description=(
        "REST API for multilingual fake news detection "
        "powered by a fine-tuned XLM-RoBERTa model "
        "and optimized for CPU inference using quantized ONNX."
    ),
    contact={
        "name": "Desire K. Korda",
        "url": "https://github.com/desirekkorda",
    },
    license_info={
        "name": "MIT License"
    }
)


# -------------------------------------------------------------------------
# CORS configuration
# -------------------------------------------------------------------------
#
# Development:
#   Use "*" temporarily while testing the PWA locally.
#
# Production:
#   Replace "*" with the actual PWA domain(s).
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# API routes
# -------------------------------------------------------------------------

app.include_router(router)