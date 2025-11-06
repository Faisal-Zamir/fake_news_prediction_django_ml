import os
import joblib
import re
import contractions
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------- Configuration --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and TF-IDF vectorizer
model = joblib.load(os.path.join(BASE_DIR, "fake_news_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

# Initialize preprocessing components
# stop_words = set(stopwords.words('english'))

# Download only if missing
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


lemmatizer = WordNetLemmatizer()

# -------------------- Text Preprocessing --------------------
def preprocess_text(text):
    """
    Preprocess news text exactly as done during training:
    1. Lowercase
    2. Remove punctuation/special characters
    3. Expand contractions
    4. Remove stopwords
    5. Remove short words (<3 chars)
    6. Lemmatization
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
    Predict whether a single news article is FAKE or REAL.
    
    Returns a dictionary suitable for Django views:
    {
        'prediction': 0 or 1,
        'prediction_label': 'REAL NEWS' or 'FAKE NEWS',
        'is_fake': True/False,
        'fake_probability': float,
        'real_probability': float,
        'confidence': float
    }
    """
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Transform with TF-IDF
    text_tfidf = tfidf.transform([cleaned_text])
    
    # Predict class
    prediction = model.predict(text_tfidf)[0]
    
    # Predict probabilities
    try:
        prob = model.predict_proba(text_tfidf)[0]
        fake_prob = prob[1]  # Probability of being fake
        real_prob = prob[0]  # Probability of being real
        confidence = max(prob)
    except:
        fake_prob = "N/A"
        real_prob = "N/A"
        confidence = "N/A"
    
    # Map prediction
    result_label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    is_fake = True if prediction == 1 else False
    
    return {
        'prediction': int(prediction),
        'prediction_label': result_label,
        'is_fake': is_fake,
        'fake_probability': fake_prob,
        'real_probability': real_prob,
        'confidence': confidence,
    }

# -------------------- Test Dummy News --------------------
if __name__ == "__main__":
    dummy_news = "Breaking: Scientists discovered a new planet in our solar system that could support life!"
    
    result = predict_fake_news(dummy_news)
    
    print("🔍 FAKE NEWS DETECTION RESULT")
    print("=" * 50)
    print(f"Input Text: {dummy_news}")
    print(f"Prediction: {result['prediction']} ({result['prediction_label']})")
    print(f"Is Fake: {result['is_fake']}")
    if result['confidence'] != "N/A":
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Fake Probability: {result['fake_probability']:.4f}")
        print(f"Real Probability: {result['real_probability']:.4f}")
    
    if result['is_fake']:
        print("🚨 WARNING: This appears to be FAKE NEWS!")
    else:
        print("✅ This appears to be REAL NEWS!")
