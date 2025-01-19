import pandas as pd
from sklearn.svm import SVC
import joblib
from preprocessing_features import preprocess_with_stemming, compute_tfidf, sentiment_features, combine_features

# File paths
cleaned_train_file = 'data/cleaned_twitter_training.csv'
cleaned_validation_file = 'data/cleaned_twitter_validation.csv'
model_dir = 'models'
model_file = f'{model_dir}/sentiment_svm_model.pkl'
vectorizer_file = f'{model_dir}/tfidf_vectorizer.pkl'

# Load cleaned datasets
train_data = pd.read_csv(cleaned_train_file)
validation_data = pd.read_csv(cleaned_validation_file)

# Preprocess data with Light Stemming
train_data = preprocess_with_stemming(train_data, 'Tweet content')
validation_data = preprocess_with_stemming(validation_data, 'Tweet content')

# Map sentiments to numerical values
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
y_train = train_data['sentiment'].map(sentiment_mapping)
y_val = validation_data['sentiment'].map(sentiment_mapping)

# Verify no NaN values
assert y_train.isnull().sum() == 0, "NaN values found in y_train!"
assert y_val.isnull().sum() == 0, "NaN values found in y_val!"

# Compute Sentiment Features
train_sentiment_features = sentiment_features(train_data['Tweet content'])
validation_sentiment_features = sentiment_features(validation_data['Tweet content'])

# Compute TF-IDF Features
X_train_tfidf, tfidf_vectorizer = compute_tfidf(train_data, 'Tweet content', fit=True)
X_val_tfidf, _ = compute_tfidf(validation_data, 'Tweet content', fit=False, tfidf_vectorizer=tfidf_vectorizer)

# Combine TF-IDF and Sentiment Features
X_train = combine_features(X_train_tfidf, train_sentiment_features)
X_val = combine_features(X_val_tfidf, validation_sentiment_features)

# Train the SVM model
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(svm, model_file)
joblib.dump(tfidf_vectorizer, vectorizer_file)
print(f"Model saved to {model_file}")
print(f"TF-IDF vectorizer saved to {vectorizer_file}")
