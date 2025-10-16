import os

# Base directory (one level above backend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_review_model.pkl")

print("Base directory:", BASE_DIR)
print("Looking for vectorizer at:", VECTORIZER_PATH)
print("Looking for model at:", MODEL_PATH)

print("Vectorizer exists:", os.path.exists(VECTORIZER_PATH))
print("Model exists:", os.path.exists(MODEL_PATH))
