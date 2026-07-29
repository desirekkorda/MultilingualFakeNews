import sys
import time
from pathlib import Path

import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

# -----------------------------------------------------------------------------
# 1. Page Configuration (MUST BE FIRST STREAMLIT CALL)
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

# Optional UI helpers from your app folder
try:
    from app.ui.components import section_title
    from app.ui.utils import show_footer
except ImportError:

    def section_title(text):
        st.subheader(text)

    def show_footer():
        pass


# -----------------------------------------------------------------------------
# 2. Model Loading
# -----------------------------------------------------------------------------
MODEL_ID = "desirekkorda/multilingual-fake-news-xlmr-v2"


@st.cache_resource
def load_onnx_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_ID, file_name="model_quantized.onnx"
    )
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,  # Return all class probabilities
    )


with st.spinner("Loading lightweight ONNX model into memory..."):
    classifier = load_onnx_pipeline()


# -----------------------------------------------------------------------------
# 3. Sidebar UI
# -----------------------------------------------------------------------------
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=200)

    st.title("Fake News Detector")
    st.markdown("---")

    st.subheader("📌 About")
    st.write(
        "This application detects whether a news article is more likely to be "
        "Legitimate or Fake using a fine-tuned XLM-RoBERTa model."
    )

    st.markdown("---")
    st.subheader("📊 Model Performance")
    st.metric("Accuracy", "83.9%")
    st.metric("Precision", "89.8%")
    st.metric("Recall", "75.9%")
    st.metric("F1 Score", "82.2%")

    st.markdown("---")
    st.subheader("🌍 Languages")
    st.success("English")
    st.caption("Coming Soon")
    st.write("🚧 Swahili | 🚧 Hindi | 🚧 Indonesian | 🚧 Vietnamese")

    st.markdown("---")
    st.subheader("Try an Example")

    legit_example = "NASA scientists confirmed new evidence of water beneath the Martian surface."
    fake_example = (
        "Rebounding Revenge! Selena Gomez And Orlando Bloom Are Hooking Up "
        "To Make Miranda Kerr And Justin Bieber Jealous?!"
    )

    if st.button("Load Legit Example"):
        st.session_state.news_text = legit_example
        st.rerun()

    if st.button("Load Fake Example"):
        st.session_state.news_text = fake_example
        st.rerun()


# -----------------------------------------------------------------------------
# 4. Main Header
# -----------------------------------------------------------------------------
logo_col, title_col = st.columns([1, 6])

with logo_col:
    if LOGO.exists():
        st.image(str(LOGO), width=120)

with title_col:
    st.title("Multilingual Fake News Detector")
    st.caption("AI-powered multilingual fake news detection using XLM-RoBERTa")

# st.markdown(
#     """
#     Detect whether a news article is more likely to be **Legitimate** or **Fake**
#     using a fine-tuned multilingual transformer model.

#     Simply paste a news article below and click **Analyze Article**.
#     """
# )
st.markdown("---")

# Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🤖 Model", "XLM-R Base (ONNX)")
with col2:
    st.metric("🌍 Languages", "1")
with col3:
    st.metric("🚀 Version", "v1.2.0")

st.markdown("---")


# -----------------------------------------------------------------------------
# 5. User Input & Inference
# -----------------------------------------------------------------------------
section_title("Analyze Article")

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

news_text = st.text_area("Paste a news article", height=200, key="news_text")

analyze = st.button("🔍 Analyze Article", use_container_width=True, type="secondary")

if analyze:
    if not news_text.strip():
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                start_time = time.time()

                # Run inference via loaded ONNX model pipeline
                raw_outputs = classifier(news_text)[0]
                elapsed = time.time() - start_time

                # Map outputs (Assuming LABEL_0 = Legit, LABEL_1 = Fake or similar)
                probs = {item["label"]: item["score"] for item in raw_outputs}

                # Normalize label names if pipeline uses LABEL_0 / LABEL_1
                legit_prob = probs.get("Legit", probs.get("LABEL_0", 0.0))
                fake_prob = probs.get("Fake", probs.get("LABEL_1", 0.0))

                prediction = "Legit" if legit_prob >= fake_prob else "Fake"
                confidence = max(legit_prob, fake_prob)

                st.session_state.result = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": {"Legit": legit_prob, "Fake": fake_prob},
                    "elapsed": elapsed,
                    "model": "XLM-RoBERTa (INT8 ONNX)",
                    "model_version": "1.2.0",
                }

            except Exception as e:
                st.error(f"Inference Error: {e}")


# -----------------------------------------------------------------------------
# 6. Display Prediction Results
# -----------------------------------------------------------------------------
if st.session_state.get("result") is not None:
    res = st.session_state.result
    st.divider()
    section_title("Analysis Result")

    if res["prediction"] == "Legit":
        st.success("🟢 Likely Legitimate News")
        st.markdown(
            "The article appears consistent with patterns learned from legitimate news articles."
        )
    else:
        st.error("🔴 Likely Fake News")
        st.markdown(
            "The article contains patterns commonly associated with misinformation."
        )

    # Probabilities and Metrics Display
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Class Probabilities")
        st.write(f"**Legit:** {res['probabilities']['Legit']:.2%}")
        st.progress(res["probabilities"]["Legit"])

        st.write(f"**Fake:** {res['probabilities']['Fake']:.2%}")
        st.progress(res["probabilities"]["Fake"])

    with right:
        st.subheader("Performance Metrics")
        st.metric("Model Confidence", f"{res['confidence']:.2%}")
        st.metric("Inference Time", f"{res['elapsed']:.2f} sec")

    with st.expander("Technical Metadata"):
        st.json(res)

show_footer()