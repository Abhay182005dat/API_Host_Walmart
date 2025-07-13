import os
from dotenv import load_dotenv
import requests
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# ✅ Load env vars
load_dotenv()

app = Flask(__name__)

# ✅ Download model helper
def download_model(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        response = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(response.content)

# ✅ Download both models
download_model(os.getenv("ETA_MODEL_URL"), "eta_model.pkl")
download_model(os.getenv("DELAY_MODEL_URL"), "delay_model.pkl")

# ✅ Load both models
with open("eta_model.pkl", "rb") as f:
    eta_model = pickle.load(f)

with open("delay_model.pkl", "rb") as f:
    delay_model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    eta_pred = eta_model.predict(df)[0]
    delay_pred = delay_model.predict(df)[0]

    return jsonify({
        "ETA_hours": round(float(eta_pred), 2),
        "delayed": bool(delay_pred)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

