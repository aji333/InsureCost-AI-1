# =============================================================================
# local_app.py  --  InsureCost-AI Local Development Server
# =============================================================================
# Use this to TEST the application on your own laptop BEFORE pushing to Vercel.
# This file is NOT deployed -- it is only for local testing.
#
# Usage:
#   python local_app.py
#   Then open: http://localhost:5000
#
# Requirements:
#   pip install flask scikit-learn numpy joblib
#
# Make sure you have run train.py first so that model.pkl and scaler.pkl exist.
#
# IMPORTANT -- target transform:
#   train.py saves a model that predicts log1p(annual_cost_usd).
#   This file applies np.expm1() to convert predictions back to dollars.
# =============================================================================

import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="public", static_url_path="")

# -- Load model artifacts ------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "\n  ERROR: model.pkl not found.\n"
        "  Please run 'python train.py' first to generate the model."
    )
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        "\n  ERROR: scaler.pkl not found.\n"
        "  Please run 'python train.py' first to generate the scaler."
    )

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print(f"  [OK] Model loaded from  : {MODEL_PATH}")
print(f"  [OK] Scaler loaded from : {SCALER_PATH}")

# -- Risk tier thresholds (design spec) ----------------------------------------
def get_risk_level(cost):
    if cost < 10_000:
        return "Low"
    elif cost < 30_000:
        return "Medium"
    elif cost < 50_000:
        return "High"
    else:
        return "Very High"

# -- Serve the web page --------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("public", "index.html")

# -- Prediction API endpoint ---------------------------------------------------
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():

    # CORS preflight
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    # Extract and validate the 7 fields
    try:
        age                     = float(data["age"])
        gender_str              = str(data["gender"]).strip().lower()
        chronic_condition_count = float(data["chronic_condition_count"])
        prior_12m_cost          = float(data["prior_12m_cost"])
        er_visit_count_12m      = float(data["er_visit_count_12m"])
        inpatient_days_12m      = float(data["inpatient_days_12m"])
        comorbidity_index       = float(data["comorbidity_index"])
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400

    # Validate ranges
    errors = []
    if not (18 <= age <= 100):
        errors.append("Age must be between 18 and 100.")
    if gender_str not in ("male", "female"):
        errors.append("Gender must be 'Male' or 'Female'.")
    if not (0 <= chronic_condition_count <= 20):
        errors.append("Chronic conditions must be 0-20.")
    if not (0 <= prior_12m_cost <= 1_000_000):
        errors.append("Prior cost must be $0 - $1,000,000.")
    if not (0 <= er_visit_count_12m <= 50):
        errors.append("ER visits must be 0-50.")
    if not (0 <= inpatient_days_12m <= 365):
        errors.append("Inpatient days must be 0-365.")
    if not (0 <= comorbidity_index <= 30):
        errors.append("Comorbidity index must be 0-30.")
    if errors:
        return jsonify({"error": " | ".join(errors)}), 400

    # Encode gender (must match train.py: Male=1, Female=0)
    gender_encoded = 1.0 if gender_str == "male" else 0.0

    # Build feature array (must match the column order used in train.py)
    features = np.array([[
        age, gender_encoded, chronic_condition_count,
        prior_12m_cost, er_visit_count_12m,
        inpatient_days_12m, comorbidity_index,
    ]])

    # Scale -> predict (log space) -> expm1() -> dollars
    # Design spec: model predicts log1p(cost); apply expm1() at inference.
    # Guardrail: clip log prediction to training range before expm1() to
    # prevent extreme extrapolation on out-of-distribution inputs.
    # Training data max cost: ~$192K -> log1p(192010) ~= 12.17
    # Cap: expm1(12.5) ~= $268K  (reasonable upper bound for a member)
    scaled         = scaler.transform(features)
    log_pred       = float(model.predict(scaled)[0])
    log_pred       = min(log_pred, 12.5)          # guardrail in log space
    predicted_cost = float(np.expm1(log_pred))   # design spec inverse
    predicted_cost = max(0.0, predicted_cost)     # no negative costs

    response = jsonify({
        "predicted_cost": round(predicted_cost, 2),
        "risk_level":     get_risk_level(predicted_cost),
    })
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# -- Run -----------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  InsureCost-AI  --  Local Development Server")
    print("=" * 50)
    print("  Open your browser at:  http://localhost:5000")
    print("  Press Ctrl+C to stop.")
    print("=" * 50)
    print()
    app.run(debug=True, port=5000)
