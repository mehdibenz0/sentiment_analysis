import tkinter as tk
from tkinter import ttk
from utils.data_preprocessing import prepare_data
from utils.model import train_and_evaluate
from utils.predict import predict_sentiment

def predict():
    text = entry.get()
    if text.strip().lower() == 'q':
        root.destroy()
        return
    prediction = predict_sentiment(text, model, vectorizer)
    result_label.config(text=f"Predicted Sentiment: {prediction}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = prepare_data()
    model = train_and_evaluate(X_train, X_test, y_train, y_test)

    root = tk.Tk()
    root.title("Sentiment Analysis")

    ttk.Label(root, text="Enter text to predict sentiment:").pack(pady=10)

    entry = ttk.Entry(root, width=50)
    entry.pack(pady=5)
    entry.focus()

    predict_button = ttk.Button(root, text="Predict", command=predict)
    predict_button.pack(pady=5)

    result_label = ttk.Label(root, text="", font=("Helvetica", 16))
    result_label.pack(pady=20)

    root.mainloop()
