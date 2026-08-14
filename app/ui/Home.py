import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGO = PROJECT_ROOT / "assets" / "images" / "app-logo.png"
ICON = PROJECT_ROOT / "assets" / "images" / "favicon.png"

st.set_page_config(
    page_title="Multilingual Fake News Detector",
    page_icon=str(ICON) if ICON.exists() else "📰",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Optional UI helpers
# -----------------------------------------------------------------------------

try:
    from app.ui.components import section_title
    from app.ui.utils import show_footer

except ImportError:

    def section_title(text):
        st.subheader(text)

    def show_footer():
        pass


# -----------------------------------------------------------------------------
# 2. Production Model Configuration
# -----------------------------------------------------------------------------

MODEL_ID = "desirekkorda/multilingual-fake-news-xlmr-v3"
MODEL_FILENAME = "model_quantized.onnx"

MAX_LEN = 384

ID2LABEL = {
    0: "Legit",
    1: "Fake",
}

SUPPORTED_LANGUAGES = [
    "English",
    "Hindi",
    "Indonesian",
    "Swahili",
    "Vietnamese",
]


# -----------------------------------------------------------------------------
# 3. Model Loading
# -----------------------------------------------------------------------------

@st.cache_resource
def load_onnx_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        file_name=MODEL_FILENAME,
    )

    model.config.id2label = ID2LABEL

    return tokenizer, model


with st.spinner(
    "Loading multilingual INT8 ONNX model into memory..."
):
    tokenizer, classifier = load_onnx_model()


# -----------------------------------------------------------------------------
# 4. Prediction Function
# -----------------------------------------------------------------------------

def softmax(logits):

    logits = np.asarray(
        logits,
        dtype=np.float32
    )

    shifted = logits - np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    exp_values = np.exp(
        shifted
    )

    return exp_values / np.sum(
        exp_values,
        axis=-1,
        keepdims=True
    )


def predict_text(text):

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    outputs = classifier(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
    )

    probabilities = softmax(
        outputs.logits
    )[0]

    prediction_id = int(
        np.argmax(probabilities)
    )

    legit_probability = float(
        probabilities[0]
    )

    fake_probability = float(
        probabilities[1]
    )

    return {
        "prediction": ID2LABEL[prediction_id],
        "confidence": float(
            probabilities[prediction_id]
        ),
        "probabilities": {
            "Legit": legit_probability,
            "Fake": fake_probability,
        },
        "label_id": prediction_id,
    }


# -----------------------------------------------------------------------------
# 5. Sidebar UI
# -----------------------------------------------------------------------------

with st.sidebar:

    if LOGO.exists():
        st.image(
            str(LOGO),
            width=200
        )

    st.title("Fake News Detector")
    st.markdown("---")

    st.subheader("📌 About")

    st.write(
        "This application detects whether a textual news article is more "
        "likely to be Legitimate or Fake using a fine-tuned multilingual "
        "XLM-RoBERTa model."
    )

    st.markdown("---")

    st.subheader("📊 Model Performance")

    st.metric("Accuracy", "92.02%")
    st.metric("Precision", "91.63%")
    st.metric("Recall", "92.44%")
    st.metric("F1 Score", "92.04%")

    st.caption(
        "Measured on the held-out five-language test set."
    )

    st.markdown("---")

    st.subheader("🌍 Supported Languages")

    for language in SUPPORTED_LANGUAGES:
        st.success(language)

    st.markdown("---")

    st.subheader("Try an Example")

    legit_example = (
        "NASA scientists confirmed new evidence of water beneath "
        "the Martian surface."
    )

    fake_example = (
        "Rebounding Revenge! Selena Gomez And Orlando Bloom Are "
        "Hooking Up To Make Miranda Kerr And Justin Bieber Jealous?!"
    )

    if st.button("Load Legit Example"):
        st.session_state.news_text = legit_example
        st.rerun()

    if st.button("Load Fake Example"):
        st.session_state.news_text = fake_example
        st.rerun()


# -----------------------------------------------------------------------------
# 6. Main Header
# -----------------------------------------------------------------------------

logo_col, title_col = st.columns(
    [1, 6]
)

with logo_col:

    if LOGO.exists():
        st.image(
            str(LOGO),
            width=120
        )

with title_col:

    st.title(
        "Multilingual Fake News Detector"
    )

    st.caption(
        "AI-powered multilingual fake news detection "
        "using XLM-RoBERTa"
    )

st.markdown("---")


# -----------------------------------------------------------------------------
# 7. Model Information Row
# -----------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🤖 Model",
        "XLM-R Base"
    )

with col2:
    st.metric(
        "🌍 Languages",
        "5"
    )

with col3:
    st.metric(
        "🚀 Version",
        "v3.0.0"
    )

st.markdown("---")


# -----------------------------------------------------------------------------
# 8. User Input & Inference
# -----------------------------------------------------------------------------

section_title(
    "Analyze Article"
)

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

news_text = st.text_area(
    "Paste a news article",
    height=200,
    key="news_text",
)

analyze = st.button(
    "🔍 Analyze Article",
    use_container_width=True,
    type="secondary",
)

if analyze:

    if not news_text.strip():

        st.warning(
            "Please enter some news text."
        )

    else:

        with st.spinner(
            "Analyzing text..."
        ):

            try:

                start_time = time.perf_counter()

                result = predict_text(
                    news_text
                )

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                st.session_state.result = {
                    **result,
                    "elapsed": elapsed,
                    "model": (
                        "XLM-RoBERTa "
                        "(INT8 ONNX)"
                    ),
                    "model_version": "3.0.0",
                }

            except Exception as e:

                st.error(
                    f"Inference Error: {e}"
                )


# -----------------------------------------------------------------------------
# 9. Display Prediction Results
# -----------------------------------------------------------------------------

if st.session_state.get(
    "result"
) is not None:

    res = st.session_state.result

    st.divider()

    section_title(
        "Analysis Result"
    )

    if res["prediction"] == "Legit":

        st.success(
            "🟢 Likely Legitimate News"
        )

        st.markdown(
            "The article appears consistent with patterns "
            "learned from legitimate news articles."
        )

    else:

        st.error(
            "🔴 Likely Fake News"
        )

        st.markdown(
            "The article contains patterns commonly "
            "associated with misinformation."
        )

    left, right = st.columns(
        [3, 2]
    )

    with left:

        st.subheader(
            "Class Probabilities"
        )

        st.write(
            f"**Legit:** "
            f"{res['probabilities']['Legit']:.2%}"
        )

        st.progress(
            res["probabilities"]["Legit"]
        )

        st.write(
            f"**Fake:** "
            f"{res['probabilities']['Fake']:.2%}"
        )

        st.progress(
            res["probabilities"]["Fake"]
        )

    with right:

        st.subheader(
            "Performance Metrics"
        )

        st.metric(
            "Model Confidence",
            f"{res['confidence']:.2%}"
        )

        st.metric(
            "Inference Time",
            f"{res['elapsed']:.2f} sec"
        )

    with st.expander(
        "Technical Metadata"
    ):

        st.json(res)


show_footer()