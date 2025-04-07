import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
else:
    raise FileNotFoundError("🔥 model.pkl not found! Train the model using train_model.py.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        features = [float(request.form[key]) for key in request.form.keys()]
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return f"❌ Error: {str(e)}", 500  # Return error with status code 500

if __name__ == "__main__":
    app.run(debug=True)
