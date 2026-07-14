"""
Inference module for the Multilingual Fake News Detection system.
"""

HF_REPO = "desirekkorda/multilingual-fake-news-xlmr"
MODEL_FILENAME = "best_xlmr.pt"

import os
from huggingface_hub import hf_hub_download

import torch
from transformers import AutoTokenizer

from src.model import XLMRClassifier
from src.preprocessing import clean_text
from src.config import *

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Cached objects
_model = None
_tokenizer = None


def load_model():
    """
    Load the tokenizer and trained model once.
    """

    global _model
    global _tokenizer

    if _model is None:

        print("Loading tokenizer...")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        print("Loading trained model...")

        _model = XLMRClassifier()

        # checkpoint = torch.load(
        #     f"{MODEL_DIR}/best_xlmr.pt",
        #     map_location=device
        # )

        local_model = os.path.join(
            MODEL_DIR,
            MODEL_FILENAME
        )

        if os.path.exists(local_model):

            model_path = local_model

        else:

            print("📥 Downloading model from Hugging Face Hub...")

            model_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=MODEL_FILENAME
            )

        checkpoint = torch.load(
            model_path,
            map_location="cpu"
        )

        _model.load_state_dict(checkpoint)

        _model.to(device)

        _model.eval()

        print("Model ready.")

        if os.path.exists(local_model):

            print("📂 Using local model.")

            model_path = local_model

        else:

            print("📥 Downloading model from Hugging Face Hub...")

            model_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=MODEL_FILENAME
            )

    return _model, _tokenizer
    
import torch.nn.functional as F

# News Prediction
def predict_news(text):
    """
    Predict whether a news article is Legit or Fake.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict
    """

    model, tokenizer = load_model()

    # Clean text
    text = clean_text(text)

    # Tokenize
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Prediction
    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0][prediction].item()

    return {
        "prediction": ID2LABEL[prediction],
        "label_id": prediction,
        "confidence": round(confidence, 4),
        "is_confident": confidence >= 0.80,
        "probabilities": {
            "Legit": round(probabilities[0][0].item(), 4),
            "Fake": round(probabilities[0][1].item(), 4)
        },
        "model": MODEL_NAME,
        "max_length": MAX_LEN,
        "model_version": "1.0.0"
    }
    
    
# Adding batch prediction

import pandas as pd

def predict_dataframe(df, text_column="text"):
    """
    Predict an entire DataFrame of news articles.

    Parameters
    ----------
    df : pandas.DataFrame
    text_column : str

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()

    predictions = []
    confidences = []

    for text in df[text_column]:

        result = predict_news(text)

        predictions.append(result["prediction"])
        confidences.append(result["confidence"])

    df["Prediction"] = predictions
    df["Confidence"] = confidences

    return df