from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from src.inference import load_model, unload_model


@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "Starting application: loading production model...",
        flush=True
    )

    # This executes before the API begins accepting requests.
    load_model()

    print(
        "Production model loaded. API is ready.",
        flush=True
    )

    yield

    print(
        "Shutting down: releasing model resources...",
        flush=True
    )

    unload_model()


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
    },
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],
    allow_headers=["*"],
)


app.include_router(router)