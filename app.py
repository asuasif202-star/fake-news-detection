import os
import joblib
import streamlit as st

MODEL_PATH = os.path.join("models", "fake_news_model.pkl")

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection System")
st.caption("Machine Learning • TF-IDF + Logistic Regression")

st.info(
    "This app predicts whether text resembles FAKE or REAL examples in the training dataset. "
    "It is not a fact-checking service and cannot prove that an article is true or false."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found.")
    st.markdown(
        "### First-time setup\n"
        "1. Put `Fake.csv` and `True.csv` inside the `data` folder.\n"
        "2. Open the VS Code terminal in this project folder.\n"
        "3. Run: `python train_model.py`\n"
        "4. Restart this app."
    )
    st.stop()

model = load_model()

article = st.text_area(
    "Paste a news headline or article",
    height=220,
    placeholder="Enter at least five words from a news article..."
)

if st.button("Detect News", type="primary", use_container_width=True):
    cleaned = (article or "").strip()
    if len(cleaned.split()) < 5:
        st.warning("Please enter a longer headline or article (at least 5 words).")
    else:
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]
        classes = model.named_steps["classifier"].classes_
        probability_map = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }

        if prediction == "FAKE":
            st.error("Prediction: FAKE")
            st.write("The text resembles fake-news examples in the training dataset.")
        else:
            st.success("Prediction: REAL")
            st.write("The text resembles real-news examples in the training dataset.")

        st.subheader("Model confidence")
        st.bar_chart(probability_map)
        for label, probability in probability_map.items():
            st.write(f"**{label}:** {probability:.2%}")

        st.caption("Confidence is the model's estimated class probability, not proof of factual accuracy.")
