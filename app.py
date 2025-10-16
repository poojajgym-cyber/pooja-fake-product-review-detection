from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import os
import sys

app = Flask(__name__)
app.secret_key = "your_secret_key"

# ---------- Dummy users ----------
users = {"admin": "admin123", "user": "user123"}

# ---------- Paths for ML model + vectorizer ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_DIR = os.path.join(BASE_DIR, "model")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_review_model.pkl")

# ---------- Ensure files exist ----------
if not os.path.exists(VECTORIZER_PATH):
    print(f"Error: Vectorizer file not found at {VECTORIZER_PATH}")
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

# ---------- Load model and vectorizer ----------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ---------- Routes ----------

@app.route('/', methods=['GET', 'POST'])
def login():
    error = ""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('categories'))
        else:
            error = "Invalid credentials!"
    return render_template("login.html", error=error)


@app.route('/categories')
def categories():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("categories.html")


@app.route('/review', methods=['GET', 'POST'])
def review():
    if 'user' not in session:
        return redirect(url_for('login'))

    result = None
    if request.method == 'POST':
        product = request.form.get('product')
        review_text = request.form.get('review')
        rating = request.form.get('rating')

        # Transform and predict
        X = vectorizer.transform([review_text])
        prediction = model.predict(X)[0]
        result = "Review is Real ✅" if prediction == 1 else "Review is Fake ❌"

        return render_template("result.html",
                               product=product,
                               rating=rating,
                               review=review_text,
                               result=result)
    return render_template("review.html")


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


# ---------- Run Server ----------
if __name__ == "__main__":
    print("Starting Flask server...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Vectorizer path: {VECTORIZER_PATH}")
    app.run(debug=True)
