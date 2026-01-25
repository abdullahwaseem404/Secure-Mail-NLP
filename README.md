# ✉️ Secure-Mail-NLP
# Email Spam Detector

Working Demo : https://www.youtube.com/watch?v=n_zNlsyrzPc

A machine learning project to classify email/text messages as **SPAM** or **NOT SPAM** using multiple models (Naive Bayes, Logistic Regression, Linear SVM).  
This project includes both a **Streamlit web app** and a **CLI tool** for interactive spam detection.

---

## 🚀 Features
- **Streamlit Web App**: User-friendly interface to classify messages.
- **Command-Line Interface (CLI)**: Lightweight terminal-based spam detector.
- **Multiple Models Tested**:
  - Naive Bayes
  - Logistic Regression
  - Linear SVM
- **Preprocessing**:
  - Text normalization
  - Stopword removal (NLTK)
  - Stemming (Porter Stemmer)
- **Visualization**:
  - Model performance comparison using Seaborn & Matplotlib.

---

## 📊 Model Performance
| Model               | Accuracy |
|----------------------|----------|
| Linear SVM           | 0.9821 |
| Naive Bayes          | 0.9806   |
| Logistic Regression  | 0.9806   |

Linear SVM achieved the highest accuracy on the dataset.

## 📂 Project Structure
```
├── svm_model.joblib          # Trained SVM model
├── vectorizer.joblib         # Trained CountVectorizer
├── spam.csv                  # Dataset (SMS Spam Collection)
├── app.py                    # Streamlit web app
├── predict.py                # CLI tool
├── mail.ipynb                # Training & evaluation script
├── requirements.txt          # Install Dependencies
└── README.md                 # Project documentation
```

Dataset Link : https://www.kaggle.com/datasets/imgowthamg/email-spam-and-non-spam-datasets

---

## ▶️ Usage

### 1. Clone this repository
```bash
git clone https://github.com/abdullahwaseem404/Secure-Mail-NLP.git
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App
```bash
streamlit run app.py
```
