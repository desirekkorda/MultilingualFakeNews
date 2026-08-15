"""
Inference module for the Multilingual Fake News Detection system.

Production model:
    XLM-RoBERTa-base fine-tuned on five TALLIP languages
    and deployed as a quantized ONNX model.
"""

import os
from typing import Any

import numpy as np
import pandas as pd

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


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

SUPPORTED_LANGUAGES = [
    "English",
    "Hindi",
    "Indonesian",
    "Swahili",
    "Vietnamese",
]

ID2LABEL = {
    0: "Legit",
    1: "Fake",
}

LABEL2ID = {
    "Legit": 0,
    "Fake": 1,
}


# -------------------------------------------------------------------------
# Cached model objects
# -------------------------------------------------------------------------

_model = None
_tokenizer = None


def load_model():
    """
    Load the tokenizer and quantized ONNX model once.
    """

    global _model
    global _tokenizer

    if _model is None or _tokenizer is None:

        print("=== MODEL LOAD START ===", flush=True)

        print(
            f"HF_REPO={HF_REPO}",
            flush=True
        )

        print(
            f"MODEL_FILENAME={MODEL_FILENAME}",
            flush=True
        )

        print(
            "Loading multilingual tokenizer...",
            flush=True
        )

        _tokenizer = AutoTokenizer.from_pretrained(
            HF_REPO
        )

        print(
            "Tokenizer loaded successfully.",
            flush=True
        )

        print(
            "Loading quantized ONNX model...",
            flush=True
        )

        _model = ORTModelForSequenceClassification.from_pretrained(
            HF_REPO,
            file_name=MODEL_FILENAME
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

# -------------------------------------------------------------------------
# Prediction helper
# -------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    """

    logits = np.asarray(logits, dtype=np.float32)

    shifted = logits - np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=-1,
        keepdims=True
    )


# -------------------------------------------------------------------------
# News prediction
# -------------------------------------------------------------------------

def predict_news(text: str) -> dict[str, Any]:
    """
    Predict whether a news article is Legit or Fake.

    Parameters
    ----------
    text : str
        News article text.

    Returns
    -------
    dict
        Prediction, confidence, probabilities,
        and model metadata.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("News text cannot be empty.")

    model, tokenizer = load_model()

    # IMPORTANT:
    # We do not call the previous clean_text() function here.
    # The multilingual model was trained directly on the standardized
    # TALLIP text representation, so inference should use the same
    # representation rather than introducing a new preprocessing step.

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

    probabilities = _softmax(outputs.logits)

    prediction = int(
        np.argmax(probabilities, axis=1)[0]
    )

    legit_probability = float(
        probabilities[0][0]
    )

    fake_probability = float(
        probabilities[0][1]
    )

    confidence = float(
        probabilities[0][prediction]
    )

    return {
        "prediction": ID2LABEL[prediction],
        "label_id": prediction,
        "confidence": round(confidence, 4),
        "is_confident": confidence >= 0.80,
        "probabilities": {
            "Legit": round(legit_probability, 4),
            "Fake": round(fake_probability, 4),
        },
        "model": "XLM-RoBERTa-base",
        "model_format": "Quantized ONNX",
        "quantization": "Dynamic INT8",
        "model_version": MODEL_VERSION,
        "max_length": MAX_LEN,
        "languages": SUPPORTED_LANGUAGES,
    }


# -------------------------------------------------------------------------
# Batch prediction
# -------------------------------------------------------------------------

def predict_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
) -> pd.DataFrame:
    """
    Predict an entire DataFrame of news articles.
    """

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in DataFrame."
        )

    result_df = df.copy()

    predictions = []
    confidences = []

    for text in result_df[text_column]:

        result = predict_news(
            str(text)
        )

        predictions.append(
            result["prediction"]
        )

        confidences.append(
            result["confidence"]
        )

    result_df["Prediction"] = predictions
    result_df["Confidence"] = confidences

    return result_df