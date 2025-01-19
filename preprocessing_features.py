import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from scipy.sparse import hstack

# Download NLTK resources
nltk.download('stopwords')

# Initialize PorterStemmer and Sentiment Analyzer
ps = PorterStemmer()
analyzer = SentimentIntensityAnalyzer()

# Function for Light Stemming
def light_stemming(text):
    tokens = text.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

# Function to apply Light Stemming to a dataset
def preprocess_with_stemming(data, text_column):
    data[text_column] = data[text_column].apply(light_stemming)
    return data

# Function to extract Sentiment Features
def sentiment_features(text_series):
    sentiment_scores = text_series.apply(lambda text: analyzer.polarity_scores(text))
    sentiment_features = np.array([
        [score['neg'], score['neu'], score['pos'], score['compound']]
        for score in sentiment_scores
    ])
    return sentiment_features

# Function to compute TF-IDF Features
def compute_tfidf(data, text_column, max_features=5000, ngram_range=(1, 3), fit=True, tfidf_vectorizer=None):
    if fit:
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X = tfidf_vectorizer.fit_transform(data[text_column])
    else:
        X = tfidf_vectorizer.transform(data[text_column])
    return X, tfidf_vectorizer

# Function to combine TF-IDF and Sentiment Features
def combine_features(tfidf_features, sentiment_features):
    return hstack([tfidf_features, sentiment_features])
