"""
Inference module for the Multilingual Fake News Detection system.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import XLMRobertaTokenizer

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

HF_REPO = os.getenv(
    "HF_REPO",
    "desirekkorda/multilingual-fake-news-xlmr-v3"
)

MODEL_FILENAME = os.getenv(
    "MODEL_FILENAME",
    "model_quantized.onnx"
)

MAX_LEN = int(
    os.getenv("MAX_LEN", "384")
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "3.0.0"
)

ID2LABEL = {
    0: "Legit",
    1: "Fake",
}

LABEL2ID = {
    "Legit": 0,
    "Fake": 1,
}

SUPPORTED_LANGUAGES = [
    "English",
    "Hindi",
    "Indonesian",
    "Swahili",
    "Vietnamese",
]


# -------------------------------------------------------------------------
# Local tokenizer
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_DIR = (
    PROJECT_ROOT
    / "models"
    / "production_multilingual"
)


# -------------------------------------------------------------------------
# Cached objects
# -------------------------------------------------------------------------

_model = None
_tokenizer = None


# -------------------------------------------------------------------------
# Model initialization
# -------------------------------------------------------------------------

def load_model():
    """
    Load the tokenizer and quantized ONNX model once.
    """

    global _model
    global _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    print("=== MODEL LOAD START ===", flush=True)

    print(
        f"Tokenizer directory: {TOKENIZER_DIR}",
        flush=True
    )

    print(
        f"Tokenizer directory exists: "
        f"{TOKENIZER_DIR.exists()}",
        flush=True
    )

    required_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
    ]

    for filename in required_files:

        path = TOKENIZER_DIR / filename

        print(
            f"{filename}: {path.exists()}",
            flush=True
        )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    print(
        "Loading XLM-R fast tokenizer...",
        flush=True
    )

    _tokenizer = XLMRobertaTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        local_files_only=True,
    )

    print(
        "Tokenizer loaded successfully.",
        flush=True
    )

    # ---------------------------------------------------------
    # ONNX model
    # ---------------------------------------------------------

    print(
        "Loading quantized ONNX model...",
        flush=True
    )

    _model = ORTModelForSequenceClassification.from_pretrained(
        HF_REPO,
        file_name=MODEL_FILENAME,
    )

    print(
        "Quantized ONNX model loaded successfully.",
        flush=True
    )

    _model.config.id2label = ID2LABEL
    _model.config.label2id = LABEL2ID

    print(
        "=== MODEL LOAD COMPLETE ===",
        flush=True
    )

    return _model, _tokenizer


def unload_model():
    """
    Release cached model/tokenizer references.
    """

    global _model
    global _tokenizer

    _model = None
    _tokenizer = None


# -------------------------------------------------------------------------
# Softmax
# -------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:

    logits = np.asarray(
        logits,
        dtype=np.float32
    )

    shifted = (
        logits -
        np.max(
            logits,
            axis=-1,
            keepdims=True
        )
    )

    exp_values = np.exp(
        shifted
    )

    return (
        exp_values /
        np.sum(
            exp_values,
            axis=-1,
            keepdims=True
        )
    )


# -------------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------------

def predict_news(text: str) -> dict[str, Any]:

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "News text cannot be empty."
        )

    model, tokenizer = load_model()

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    outputs = model(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
    )

    probabilities = _softmax(
        outputs.logits
    )

    prediction = int(
        np.argmax(
            probabilities,
            axis=1
        )[0]
    )

    confidence = float(
        probabilities[0][prediction]
    )

    return {
        "prediction": ID2LABEL[prediction],
        "label_id": prediction,
        "confidence": round(
            confidence,
            4
        ),
        "is_confident": confidence >= 0.80,
        "probabilities": {
            "Legit": round(
                float(probabilities[0][0]),
                4
            ),
            "Fake": round(
                float(probabilities[0][1]),
                4
            ),
        },
        "model": "XLM-RoBERTa-base",
        "model_format": "Quantized ONNX",
        "quantization": "Dynamic INT8",
        "model_version": MODEL_VERSION,
        "max_length": MAX_LEN,
        "languages": SUPPORTED_LANGUAGES,
    }