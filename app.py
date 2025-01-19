import joblib
from preprocessing_features import light_stemming, sentiment_features, compute_tfidf
from scipy.sparse import hstack
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import time

# Load the model and vectorizer
model_file = 'models/sentiment_svm_model.pkl'
vectorizer_file = 'models/tfidf_vectorizer.pkl'
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
    return combined_features, sentiment_features_array[0]  # Return both combined and raw sentiment scores

# Function to predict sentiment
def predict_sentiment(text):
    # Prepare features and extract sentiment scores
    _, sentiment_scores = prepare_features(text)

    # Find the sentiment with the highest score
    sentiment_mapping = {0: ('Negative', '😠'), 1: ('Neutral', '😐'), 2: ('Positive', '😊')}
    max_score_index = sentiment_scores.argmax()  # Get the index of the highest score

    # Return the sentiment corresponding to the highest score
    return sentiment_mapping[max_score_index], sentiment_scores




# Tkinter App
def analyze_sentiment():
    tweet = tweet_text.get("1.0", tk.END).strip()
    if not tweet:
        messagebox.showwarning("Input Error", "Please enter a tweet!")
        return

    # Clear previous results
    result_label.config(text="")
    sentiment_scores_label.config(text="")
    time_label.config(text="")

    # Process sentiment prediction
    start_time = time.time()
    overall_sentiment, sentiment_scores = predict_sentiment(tweet)
    elapsed_time = time.time() - start_time

    # Display overall sentiment with highlight and emoji
    sentiment_text, emoji = overall_sentiment
    result_label.config(
        text=f"{emoji} {sentiment_text}",
        font=("Helvetica", 16, "bold"),
        fg="green" if sentiment_text == "Positive" else "orange" if sentiment_text == "Neutral" else "red"
    )
    
    # Display individual sentiment scores
    scores_text = (
        f"Negative: {sentiment_scores[0]:.2f}\n"
        f"Neutral: {sentiment_scores[1]:.2f}\n"
        f"Positive: {sentiment_scores[2]:.2f}\n"
        f"Compound: {sentiment_scores[3]:.2f}"
    )
    sentiment_scores_label.config(text=scores_text)

    # Display response time
    time_label.config(text=f"Response Time: {elapsed_time:.2f} seconds")

# Create the main Tkinter window
app = tk.Tk()
app.title("Tweet Sentiment Analyzer")
app.geometry("500x500")
app.configure(bg="#f4f4f9")

# Input Section
tk.Label(app, text="Enter your tweet:", font=("Helvetica", 14, "bold"), bg="#f4f4f9").pack(pady=10)
tweet_text = tk.Text(app, font=("Helvetica", 12), height=5, width=50, wrap=tk.WORD, relief=tk.GROOVE, bd=2)
tweet_text.pack(pady=5)

# Analyze Button
analyze_button = tk.Button(app, text="Analyze Sentiment", font=("Helvetica", 14), bg="#4caf50", fg="white",
                           command=analyze_sentiment, relief=tk.FLAT, padx=10, pady=5)
analyze_button.pack(pady=20)

# Result Section
result_label = tk.Label(app, text="", font=("Helvetica", 16), bg="#f4f4f9")
result_label.pack(pady=10)

sentiment_scores_label = tk.Label(app, text="", font=("Helvetica", 12), bg="#f4f4f9")
sentiment_scores_label.pack(pady=5)

time_label = tk.Label(app, text="", font=("Helvetica", 10), bg="#f4f4f9")
time_label.pack(pady=5)

# Run the Tkinter event loop
app.mainloop()
