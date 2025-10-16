import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Dummy training data
X_train = ["good product", "bad product", "excellent", "worst", "love it", "hate it"]
y_train = [1, 0, 1, 0, 1, 0]  # 1 = real, 0 = fake

# Create vectorizer & transform data
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X_train)

# Create dummy model
model = LogisticRegression()
model.fit(X_vec, y_train)

# Path to model folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Save vectorizer & model
with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "fake_review_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("Dummy model & vectorizer created successfully!")
