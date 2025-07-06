def predict_sentiment(text, model, vectorizer):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]

    # Normalize and map label properly
    label_map = {
        'pos': 'Positive',
        'neg': 'Negative',
        'positive': 'Positive',
        'negative': 'Negative'
    }
    return label_map.get(prediction.lower(), prediction.capitalize())
