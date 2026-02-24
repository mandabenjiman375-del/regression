from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = []

        # Loop through form values safely
        for value in request.form.values():
            if value.strip() == "":
                value = 0   # If empty, set default 0
            features.append(float(value))

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)

        # Scale features
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: {round(prediction[0], 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
