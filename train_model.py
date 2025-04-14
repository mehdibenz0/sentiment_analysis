import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
with open("dataset.pkl", "rb") as f:
    X, y = pickle.load(f)

# Flatten images
X_flat = X.reshape(len(X), -1)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc:.2f}")

# Save model and label encoder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("labels.pkl", "wb") as f:
    pickle.dump(le, f)
