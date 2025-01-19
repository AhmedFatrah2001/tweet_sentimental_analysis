import joblib
from preprocessing_features import light_stemming, sentiment_features, compute_tfidf
from scipy.sparse import hstack
import pandas as pd

# File paths
model_file = 'models/sentiment_svm_model.pkl'
vectorizer_file = 'models/tfidf_vectorizer.pkl'

# Load the model and vectorizer
svm = joblib.load(model_file)
tfidf = joblib.load(vectorizer_file)

# Function to preprocess and extract features for a single text
def prepare_features(text):
    # Step 1: Apply Light Stemming
    stemmed_text = light_stemming(text)
    
    # Step 2: Extract TF-IDF Features
    tfidf_features = tfidf.transform([stemmed_text])
    
    # Step 3: Extract Sentiment Features
    sentiment_features_array = sentiment_features(pd.Series([text]))
    
    # Step 4: Combine Features (TF-IDF + Sentiment)
    combined_features = hstack([tfidf_features, sentiment_features_array])
    return combined_features

# Function to predict sentiment
def predict_sentiment(text):
    # Prepare features
    features = prepare_features(text)
    
    # Predict sentiment
    sentiment = svm.predict(features)
    sentiment_mapping = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    return sentiment_mapping[sentiment[0]]

# Test with example input
if __name__ == "__main__":
    test_text = input("Enter a tweet for sentiment analysis: ")
    result = predict_sentiment(test_text)
    print(f"Predicted Sentiment: {result}")
