
import time
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import inference engeine
import streamlit as st

from src.inference import predict_news

from app.ui.utils import show_footer
from app.ui.components import section_title
from pathlib import Path

# Welcome

LOGO = PROJECT_ROOT  / "assets" / "images" / "app-logo.png"
ICON = PROJECT_ROOT  / "assets" / "images" / "favicon.png"

st.set_page_config(
    page_title="Multilingual Fake News Detection",
    page_icon=str(ICON),
    layout="wide"
)

# Sidebar
with st.sidebar:

    st.image(LOGO, width=200)

    st.title("Fake News Detector")

    st.markdown("---")

    st.subheader("📌 About")

    st.write(
        "This application detects whether a news article "
        "is more likely to be Legitimate or Fake using "
        "a fine-tuned XLM-RoBERTa model."
    )

    st.markdown("---")

    st.subheader("📊 Model Performance")

    st.metric("Accuracy", "83.9%")
    st.metric("Precision", "86.8%")
    st.metric("Recall", "79.3%")
    st.metric("F1 Score", "82.9%")

    st.markdown("---")

    st.subheader("🌍 Languages")

    st.success("English")

    st.caption("Coming Soon")

    st.write("🚧 Swahili")
    st.write("🚧 Hindi")
    st.write("🚧 Indonesian")
    st.write("🚧 Vietnamese")

    # Try example
    st.subheader("Try an Example")
    legit_example = (
        "NASA scientists confirmed new evidence of "
        "water beneath the Martian surface."
    )

    fake_example = (
        "Rebounding Revenge! Selena Gomez And Orlando Bloom Are Hooking Up "
        "To Make Miranda Kerr And Justin Bieber Jealous?! This weekend wasn't " 
        "the first time we saw Selena Gomez and Orlando Bloom together, "
        "so can you blame us for speculating???!"
    )

    if st.button("Load Legit Example"):
        st.session_state.news_text = legit_example

    if st.button("Load Fake Example"):
        st.session_state.news_text = fake_example

logo_col, title_col = st.columns([1, 6])

with logo_col:
    st.image(LOGO, width=250)

with title_col:
    st.title("Multilingual Fake News Detection")
    st.caption(
        "AI-powered multilingual fake news detection using XLM-RoBERTa"
    )

st.markdown(
    """
    Detect whether a news article is more likely to be **Legitimate** or **Fake**
    using a fine-tuned multilingual transformer model.

    Simply paste a news article below and click **Analyze Article**.
    """
)

st.markdown("---")

# Metrics Row
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🤖 Model", "XLM-R Base")

with col2:
    st.metric("🌍 Languages", "1")

with col3:
    st.metric("🚀 Version", "v1.2.0")

# Add text input
section_title("Analyze Article")

# news_text = st.text_area(
#     "Paste a news article",
#     height=250,
#     placeholder="Paste the news article here..."
# )


# Text box
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

news_text = st.text_area(
    "Paste a news article",
    height=250,
    key="news_text"
)

# Analyze button
if "result" not in st.session_state:
    st.session_state.result = None
# result = None
analyze = st.button(
    "🔍 Analyze Article",
    use_container_width=True
)

# Call the model
if analyze:

    if news_text.strip() == "":

        st.warning("Please enter some news text.")

    else:

        with st.spinner("Analyzing..."):

            start = time.time()

            st.session_state.result = predict_news(news_text)

            elapsed = time.time() - start

            st.session_state.elapsed = elapsed

# Display Prediction
if st.session_state.result is not None:

    result = st.session_state.result

    st.divider()

    section_title("Analysis Result")

# Result Card
    if result["prediction"] == "Legit":

        st.success("🟢 Analysis Complete")

        st.markdown(
            """
            ### Likely Legitimate News

            The article appears consistent with patterns learned
            from legitimate news articles.
            """
        )

    else:
        st.error("🔴 Analysis Complete")

        st.markdown(
            """
            ### Likely Fake News

            The article contains patterns commonly associated
            with misinformation.
            """
        )

    st.write(
    f"Model Confidence: "
    f"**{result['confidence']:.2%}**"
    )

#Confidencs Display
    left, right = st.columns([3, 2])
    
    with left:
        st.subheader("Class Probabilities")
        
        st.write(f"**Legit:** {result['probabilities']['Legit']:.2%}")
        st.progress(result["probabilities"]["Legit"])
        
        st.write(f"**Fake:** {result['probabilities']['Fake']:.2%}")
        st.progress(result["probabilities"]["Fake"])

    with right:
        st.subheader("Performance Metrics")
        st.metric("Model Confidence", f"{result['confidence']:.2%}")
        st.metric("Inference Time", f"{st.session_state.elapsed:.2f} sec")
        
    with st.expander("Technical Metadata"):
        st.write(f"**Model Type:** {result['model']}")
        st.write(f"**Engine Version:** {result['model_version']}")
        st.json(result)

show_footer()