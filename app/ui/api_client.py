import os
import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/api/v1/predict"
)


def predict_news(text: str):

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "text must be a non-empty string"
        )

    response = requests.post(
        API_URL,
        json={"text": text},
        timeout=60,
    )

    response.raise_for_status()

    return response.json()