import sys
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

try:
    STOPWORDS = stopwords.words("english")
except LookupError:
    print("Warning: NLTK stopwords not found. Proceeding without stopword removal.")
    STOPWORDS = []


def preprocess_text(text):
    """Clean, normalize and stem input message."""
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in STOPWORDS]
    return " ".join(text)

try:
    model = joblib.load("svm_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    print("✅ Linear SVM model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("❌ Error: svm_model.joblib or vectorizer.joblib not found.")
    print("Make sure the trained SVM model files exist in this directory.")
    sys.exit(1)


def classify_message(message):
    processed = preprocess_text(message)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return "SPAM" if prediction == 1 else "NOT SPAM"

def run_cli():
    print("\n--- AI Spam Detector (Linear SVM) ---")
    print("Type 'exit' or press Ctrl+C to quit.")

    while True:
        try:
            user_input = input("\nEnter message to classify: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye 👋")
                break

            if not user_input:
                continue

            result = classify_message(user_input)

            preview = user_input if len(user_input) <= 70 else user_input[:67] + "..."
            print("\nPrediction Result:")
            print(f"  → '{preview}'")
            print(f"  → Classified as: {result}")

        except KeyboardInterrupt:
            print("\nGoodbye 👋")
            break
        except Exception as err:
            print(f"Unexpected error: {err}")


if __name__ == "__main__":
    run_cli()
