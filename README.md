# Tweet Sentiment Analyzer

A Python-based application that uses machine learning to classify the sentiment of tweets as **Positive**, **Neutral**, or **Negative**. The project features a user-friendly GUI built with **Tkinter**, allowing users to enter a tweet, view the sentiment prediction with an emoji, and analyze detailed sentiment scores.

---

## Features
- **Light Stemming**: Preprocesses tweets by reducing words to their base forms.
- **Sentiment Analysis**:
  - Overall sentiment (Positive, Neutral, or Negative) with an emoji.
  - Detailed sentiment scores: Negative, Neutral, Positive, and Compound.
- **Machine Learning Model**:
  - Uses **Support Vector Machines (SVM)** for sentiment classification.
  - Combines **TF-IDF** and **VADER sentiment lexicon** features.
- **GUI with Tkinter**:
  - Enter a tweet in a text area.
  - Displays sentiment with emojis and processing time.
  - Visual highlighting for predicted sentiment.

---

## Screenshots
### Example Output:
![Tweet Sentiment Analyzer](screenshot.png)

---

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)

### Setup
1. Clone this repository or download the project files:
   ```bash
   git clone https://github.com/your-username/tweet-sentiment-analyzer.git
   cd tweet-sentiment-analyzer
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the following pre-trained files in the `models/` directory:
   - `sentiment_svm_model.pkl` (trained SVM model)
   - `tfidf_vectorizer.pkl` (trained TF-IDF vectorizer)

---

## Usage
1. Run the application:
   ```bash
   python app.py
   ```

2. Enter a tweet in the provided text area and click **Analyze Sentiment**.

3. View:
   - **Overall Sentiment**: Displayed with emoji (😊, 😐, 😠).
   - **Detailed Sentiment Scores**: Negative, Neutral, Positive, and Compound.
   - **Response Time**: Time taken to analyze the sentiment.

---

## File Structure
```
.
├── app.py                     # Main GUI application
├── preprocessing_features.py  # Preprocessing logic (stemming, sentiment, TF-IDF)
├── models/
│   ├── sentiment_svm_model.pkl # Trained SVM model
│   ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
├── data/
│   ├── twitter_training.csv    # (Optional) Training data
│   ├── twitter_validation.csv  # (Optional) Validation data
├── requirements.txt            # Required Python libraries
└── README.md                   # Project documentation
```

---

## Dependencies
Install these libraries via `pip`:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `nltk`
- `vaderSentiment`
- `joblib`
- `tkinter` (comes pre-installed with Python)

---

## Model Details
- **Algorithm**: Support Vector Machines (SVM) with a linear kernel.
- **Features**:
  - **TF-IDF**: Extracts n-grams (unigrams, bigrams, trigrams).
  - **Sentiment Scores**: Derived from the VADER sentiment analyzer.

---

## Future Enhancements
- Include contextual embeddings (e.g., BERT or GloVe) for richer text representation.
- Add functionality for analyzing multiple tweets in a batch.
- Improve model accuracy by augmenting the training dataset with more examples.

---

## Contributors
- **Fatrah Ahmed** (Project Creator) 😊