import joblib
from preprocessing_features import light_stemming, sentiment_features
from scipy.sparse import hstack
import pandas as pd
import tkinter as tk
from tkinter import messagebox
# from PIL import Image, ImageTk
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

    # Use only the first three scores (Negative, Neutral, Positive)
    sentiment_scores = sentiment_scores[:3]

    # Return the sentiment scores
    return sentiment_scores

# Tkinter App
def analyze_sentiment():
    tweet = tweet_text.get("1.0", tk.END).strip()
    if not tweet:
        messagebox.showwarning("Input Error", "Please enter a tweet!")
        return

    # Clear previous results
    time_label.config(text="")
    for widget in result_frame.winfo_children():
        widget.destroy()

    # Process sentiment prediction
    start_time = time.time()
    sentiment_scores = predict_sentiment(tweet)
    elapsed_time = time.time() - start_time

    # Display response time
    time_label.config(text=f"Response Time: {elapsed_time:.2f} seconds")

    # Display sentiment scores and emoji
    display_results(sentiment_scores)

    # Generate a bar chart for the sentiment scores
    generate_chart(sentiment_scores)

    # Resize the window to fit the content
    app.geometry("500x800")

def display_results(sentiment_scores):
    labels = ['Negative', 'Neutral', 'Positive']
    emojis = ['😡', '😐', '😊']
    colors = ['#ff4d4d', '#ffcc00', '#66ff66']
    max_index = sentiment_scores.argmax()

    emoji_label.config(text=emojis[max_index], font=("Helvetica", 50), fg=colors[max_index])
    emoji_label.pack(pady=10)

    for i, score in enumerate(sentiment_scores):
        tk.Label(result_frame, text=f"{labels[i]}: {score:.2f}", font=("Helvetica", 12), bg="#f4f4f9", fg="black").pack()

def generate_chart(sentiment_scores):
    # Sentiment labels
    labels = ['Negative', 'Neutral', 'Positive']
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, sentiment_scores, color=['red', 'orange', 'green'])
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Sentiment Analysis Scores", fontsize=14)
    ax.set_ylim(0, 1)  # Ensure all scores fit within the chart
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=result_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Create the main Tkinter window
app = tk.Tk()
app.title("Tweet Sentiment Analyzer")
app.geometry("500x400")
app.configure(bg="#f4f4f9")

# Input Section
tk.Label(app, text="Enter your tweet:", font=("Helvetica", 14, "bold"), bg="#f4f4f9").pack(pady=10)
tweet_text = tk.Text(app, font=("Helvetica", 12), height=5, width=50, wrap=tk.WORD, relief=tk.GROOVE, bd=2)
tweet_text.pack(pady=5)

# Emoji Label
emoji_label = tk.Label(app, text="", bg="#f4f4f9")
emoji_label.pack(pady=10)

# Analyze Button
analyze_button = tk.Button(app, text="Analyze Sentiment", font=("Helvetica", 14), bg="#4caf50", fg="white",
                           command=analyze_sentiment, relief=tk.FLAT, padx=10, pady=5)
analyze_button.pack(pady=20)

# Result Section
result_frame = tk.Frame(app, bg="#f4f4f9")
result_frame.pack(pady=10)

time_label = tk.Label(app, text="", font=("Helvetica", 10), bg="#f4f4f9")
time_label.pack(pady=5)

# Ensure the program stops when the main widget is closed
def on_closing():
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)
app.mainloop()
