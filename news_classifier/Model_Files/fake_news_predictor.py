import os
import re
import joblib
import contractions
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from news_classifier.Model_Files import nltk_downloader  # ensures nltk data downloaded

# -------------------- Configuration --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add local nltk_data path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

# -------------------- Load Model and TF-IDF --------------------
model = joblib.load(os.path.join(BASE_DIR, "fake_news_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

# -------------------- Preprocessing Components --------------------
# Ensure stopwords are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
    stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# -------------------- Text Preprocessing --------------------
def preprocess_text(text):
    """
    Clean and normalize text before prediction.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = contractions.fix(text)
    words = [word for word in text.split() if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words if len(word) >= 3]
    return ' '.join(words)

# -------------------- Prediction Function --------------------
def predict_fake_news(text):
    """
    Predict whether a given news article is fake or real.
    """
    cleaned_text = preprocess_text(text)
    text_tfidf = tfidf.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]

    try:
        prob = model.predict_proba(text_tfidf)[0]
        fake_prob, real_prob = prob[1], prob[0]
        confidence = max(prob)
    except Exception:
        fake_prob = real_prob = confidence = "N/A"

    result_label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    is_fake = prediction == 1

    return {
        'prediction': int(prediction),
        'prediction_label': result_label,
        'is_fake': is_fake,
        'fake_probability': fake_prob,
        'real_probability': real_prob,
        'confidence': confidence,
    }

# -------------------- Test Section --------------------
if __name__ == "__main__":
    dummy_news = "Breaking: Scientists discovered a new planet in our solar system that could support life!"
    result = predict_fake_news(dummy_news)

    print("🔍 FAKE NEWS DETECTION RESULT")
    print("=" * 50)
    print(f"Prediction: {result['prediction_label']} ({result['confidence']})")
    print("🚨 FAKE" if result['is_fake'] else "✅ REAL")
