import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

try:
    STOPWORDS = stopwords.words("english")
except LookupError:
    st.error(
        "NLTK stopwords not found. "
        "Run: import nltk; nltk.download('stopwords')"
    )
    STOPWORDS = []


@st.cache_data(show_spinner=False)
def preprocess_text(text):
    """Clean, normalize and stem input text."""
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in STOPWORDS]
    return " ".join(text)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        svm_model = joblib.load("svm_model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
        return svm_model, vectorizer
    except FileNotFoundError:
        st.error(
            "Model files not found. "
            "Ensure 'svm_model.joblib' and 'vectorizer.joblib' exist."
        )
        return None, None


model, vectorizer = load_artifacts()

st.set_page_config(page_title="Spam Detector (SVM)", layout="centered")

st.title("✉️ Email Spam Detector")
st.markdown(
    "Enter a message below and click **Predict**"
)

user_input = st.text_area(
    "Message to classify:",
    height=150,
    placeholder="Example: Congratulations! You've won a free voucher."
)


if st.button("Predict"):
    if model is None or vectorizer is None:
        st.error("Model failed to load. Please check model files.")
    elif not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **NOT SPAM**.")


