import re
import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

from transformers import pipeline

nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
ps = PorterStemmer()

def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")

    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]]

    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [ps.stem(w) for w in words if w not in STOPWORDS]
    return " ".join(words)

def preprocess(df):
    df["clean"] = df["message"].apply(clean_text)
    return df

def tfidf_features(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    return X_train_vec, X_test_vec

def train_word2vec(sentences):
    tokenized = [s.split() for s in sentences]
    model = Word2Vec(tokenized, vector_size=100, window=5, min_count=2)
    return model

def get_w2v_features(model, sentences):
    vectors = []
    for sent in sentences:
        words = sent.split()
        vecs = [model.wv[w] for w in words if w in model.wv]
        if vecs:
            vectors.append(np.mean(vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)

def sentence_embeddings(train, test):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(train), model.encode(test)

def get_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="linear", probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "LightGBM": LGBMClassifier()
    }

def evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("ROC-AUC:", roc)
    print(classification_report(y_test, preds))

    return acc

def cross_validate(model, X, y):
    cv = StratifiedKFold(n_splits=5)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    return scores.mean()

def run_pipeline():
    df = load_data("spam.csv")
    df = preprocess(df)

    X = df["clean"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n===== TF-IDF =====")
    X_train_tfidf, X_test_tfidf = tfidf_features(X_train, X_test)

    models = get_models()

    results = {}

    for name, model in models.items():
        acc = evaluate(model, X_train_tfidf, X_test_tfidf, y_train, y_test, name)
        cv_score = cross_validate(model, X_train_tfidf, y_train)

        results[name] = (acc, cv_score)

    print("\n===== Word2Vec =====")
    w2v_model = train_word2vec(X_train)

    X_train_w2v = get_w2v_features(w2v_model, X_train)
    X_test_w2v = get_w2v_features(w2v_model, X_test)

    for name, model in models.items():
        evaluate(model, X_train_w2v, X_test_w2v, y_train, y_test, name + " (W2V)")

    print("\n===== Sentence Transformers =====")
    X_train_st, X_test_st = sentence_embeddings(X_train, X_test)

    for name, model in models.items():
        evaluate(model, X_train_st, X_test_st, y_train, y_test, name + " (ST)")

    print("\n===== BERT (Zero-shot) =====")
    classifier = pipeline("text-classification", model="distilbert-base-uncased")

    sample = X_test.iloc[:5]
    for text in sample:
        result = classifier(text)[0]
        print(text[:60], "->", result)

    print("\n===== FINAL RESULTS =====")
    for k, v in results.items():
        print(f"{k}: Accuracy={v[0]:.4f}, CV-F1={v[1]:.4f}")

def predict_live():
    print("\n--- LIVE SPAM DETECTOR ---")
    df = load_data("spam.csv")
    df = preprocess(df)

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["clean"])
    y = df["target"]

    model = SVC(kernel="linear", probability=True)
    model.fit(X, y)

    while True:
        text = input("\nEnter message (or 'exit'): ")

        if text.lower() == "exit":
            break

        cleaned = clean_text(text)
        vec = tfidf.transform([cleaned])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        print("SPAM" if pred == 1 else "NOT SPAM", f"(confidence: {prob:.2f})")

if __name__ == "__main__":
    run_pipeline()
    predict_live()