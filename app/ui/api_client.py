import requests

API_URL = "http://127.0.0.1:8000/api/v1/predict"


def predict_news_api(text: str):

    response = requests.post(
        API_URL,
        json={"text": text},
        timeout=60
    )

    response.raise_for_status()

    return response.json()