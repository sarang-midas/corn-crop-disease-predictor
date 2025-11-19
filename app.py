from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
enc_stage = pickle.load(open("stage_encoder.pkl", "rb"))
enc_disease = pickle.load(open("disease_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    stage = request.form["stage"]

    # Encode crop stage
    stage_encoded = enc_stage.transform([stage])

    # Predict
    pred = model.predict([[stage_encoded[0]]])[0]

    # Decode disease
    disease = enc_disease.inverse_transform([pred])[0]

    return render_template("index.html", prediction=disease)

if __name__ == "__main__":
    app.run(debug=True)
