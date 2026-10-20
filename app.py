from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("model_regresi.joblib")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    hours = ""
    error = None
    if request.method == "POST":
        try:
            hours = float(request.form.get("jam_belajar", ""))
            if not np.isfinite(hours) or hours < 0 or hours > 24:
                raise ValueError
            value = float(model.predict([[hours]])[0])
            prediction = round(min(100, max(0, value)), 1)
        except (ValueError, TypeError):
            error = "Masukkan durasi belajar berupa angka antara 0 sampai 24 jam."
    return render_template("index.html", prediction=prediction, hours=hours, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
