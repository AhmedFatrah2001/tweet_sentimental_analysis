import pandas as pd

# Define column names
columns = ['Tweet ID', 'entity', 'sentiment', 'Tweet content']

# Function to clean the dataset
def clean_dataset(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, header=None, names=columns)
    
    # Remove rows where 'Tweet content' or 'sentiment' is NaN
    data = data.dropna(subset=['Tweet content', 'sentiment'])
    
    # Remove rows where 'Tweet content' is empty or whitespace
    data = data[data['Tweet content'].str.strip() != '']
    
    # Remove rows with invalid sentiments
    valid_sentiments = {'Positive', 'Neutral', 'Negative'}
    data = data[data['sentiment'].isin(valid_sentiments)]
    
    # Ensure 'Tweet content' values are strings
    data['Tweet content'] = data['Tweet content'].astype(str)
    
    return data

# Main function to clean and save data
if __name__ == "__main__":
    train_file = 'data/twitter_training.csv'
    validation_file = 'data/twitter_validation.csv'
    cleaned_train_file = 'data/cleaned_twitter_training.csv'
    cleaned_validation_file = 'data/cleaned_twitter_validation.csv'

    cleaned_train = clean_dataset(train_file)
    cleaned_train.to_csv(cleaned_train_file, index=False)
    print(f"Cleaned training data saved to {cleaned_train_file}")

    cleaned_validation = clean_dataset(validation_file)
    cleaned_validation.to_csv(cleaned_validation_file, index=False)
    print(f"Cleaned validation data saved to {cleaned_validation_file}")
