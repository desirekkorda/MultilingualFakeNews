from pydantic import BaseModel


class NewsRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    label_id: int
    confidence: float
    is_confident: bool
    probabilities: dict
    model: str
    max_length: int
    model_version: str