import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string

nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return ' '.join(tokens)

def prepare_data():
    # Load data
    df = pd.read_csv("data/raw_data.csv")
    
    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Vectorize text
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, vectorizer
