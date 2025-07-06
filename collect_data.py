import pandas as pd
import nltk
from nltk.corpus import movie_reviews
import random
import os

nltk.download('movie_reviews')

def collect_imdb_data():
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((
                movie_reviews.raw(fileid),
                category
            ))

    random.shuffle(documents)

    # Build DataFrame
    texts = [doc for doc, label in documents]
    labels = [label for doc, label in documents]

    df = pd.DataFrame({"text": texts, "label": labels})

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_data.csv", index=False)
    print(f"Saved dataset to data/raw_data.csv with {len(df)} rows.")

if __name__ == "__main__":
    collect_imdb_data()