from flask import Flask, jsonify, request
from flask_cors import CORS

from data_predict import model_predict

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "ProductShala sentiment API is running"})


@app.route("/predict", methods=["POST"])
def handle_predict():
    """Accepts either form data (review=...) or JSON ({"review": "..."})."""
    text = request.form.get("review") or (request.get_json(silent=True) or {}).get("review")

    if not text or not text.strip():
        return jsonify({"error": "Missing or empty 'review' field"}), 400

    label, polarity = model_predict(text=text)

    return jsonify({
        "review": text,
        "prediction": label,
        "polarity": polarity,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
