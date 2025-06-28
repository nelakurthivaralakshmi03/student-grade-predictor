from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("grade_predictor_presence_model.pkl")
label_encoder = joblib.load("label_encoder_presence.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        marks = []
        presence = []

        for subject in ["maths", "physics", "biology", "social", "telugu", "hindi", "english"]:
            val = request.form[subject]
            if val.upper() == "AB":
                presence.append(0)
                marks.append(0)  # AB treated as 0 in input, but logic will handle it
            else:
                score = float(val)
                presence.append(1)
                marks.append(score)

        # AB or <35 -> No Grade
        for m, p in zip(marks, presence):
            if p == 0 or m < 35:
                return render_template("index.html", prediction_text="Predicted Grade: No Grade")

        input_features = marks + presence
        prediction = model.predict([input_features])
        grade = label_encoder.inverse_transform(prediction)[0]

        return render_template("index.html", prediction_text=f"Predicted Grade: {grade}")
    except Exception as e:
        return render_template("index.html", prediction_text="Error: Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
