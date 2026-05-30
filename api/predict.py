# =============================================================================
# api/predict.py  --  InsureCost-AI Vercel Serverless Function
# =============================================================================
# This file runs on Vercel's Python runtime.
# Vercel treats every .py file inside the /api folder as a serverless endpoint.
#
# Endpoint : POST /api/predict
# Input    : JSON body with 7 member fields
# Output   : JSON with predicted_cost (USD) and risk_level
#
# The model and scaler are loaded at module level (once per cold start).
# Subsequent warm requests reuse the already-loaded objects -- fast.
#
# IMPORTANT -- target transform:
#   train.py saves a model that predicts log1p(annual_cost_usd).
#   At inference this function applies np.expm1() to convert back to dollars.
# =============================================================================

import json
import os
import numpy as np
import joblib
from http.server import BaseHTTPRequestHandler

# -- Load model artifacts at module level --------------------------------------
# BASE_DIR is the project root (one level up from api/)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    _artifacts_loaded = True
except Exception as e:
    model  = None
    scaler = None
    _artifacts_loaded = False
    _load_error = str(e)

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

# -- CORS headers (required so the browser can call this from the frontend) ----
CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json",
}

# -- Vercel serverless handler -------------------------------------------------
class handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        """Suppress default Vercel request logs for cleanliness."""
        pass

    def _send_json(self, status_code, payload):
        self.send_response(status_code)
        for key, val in CORS_HEADERS.items():
            self.send_header(key, val)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    # -- Preflight -- browser sends this before POST to check CORS -------------
    def do_OPTIONS(self):
        self.send_response(200)
        for key, val in CORS_HEADERS.items():
            self.send_header(key, val)
        self.end_headers()

    # -- Main prediction handler -----------------------------------------------
    def do_POST(self):

        # Check model loaded correctly
        if not _artifacts_loaded:
            self._send_json(500, {
                "error": (
                    f"Model failed to load: {_load_error}. "
                    "Ensure model.pkl and scaler.pkl are committed to the repo."
                )
            })
            return

        # Read and parse request body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            data = json.loads(raw_body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json(400, {"error": f"Invalid JSON body: {str(e)}"})
            return

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
            self._send_json(400, {"error": f"Missing required field: {str(e)}"})
            return
        except (ValueError, TypeError) as e:
            self._send_json(400, {"error": f"Invalid value: {str(e)}"})
            return

        # Validate ranges
        errors = []
        if not (18 <= age <= 100):
            errors.append("Age must be between 18 and 100.")
        if gender_str not in ("male", "female"):
            errors.append("Gender must be 'Male' or 'Female'.")
        if not (0 <= chronic_condition_count <= 20):
            errors.append("Chronic conditions must be between 0 and 20.")
        if not (0 <= prior_12m_cost <= 1_000_000):
            errors.append("Prior 12-month cost must be between $0 and $1,000,000.")
        if not (0 <= er_visit_count_12m <= 50):
            errors.append("ER visits must be between 0 and 50.")
        if not (0 <= inpatient_days_12m <= 365):
            errors.append("Inpatient days must be between 0 and 365.")
        if not (0 <= comorbidity_index <= 30):
            errors.append("Comorbidity index must be between 0 and 30.")
        if errors:
            self._send_json(400, {"error": " | ".join(errors)})
            return

        # Encode gender (must match train.py: Male=1, Female=0)
        gender_encoded = 1.0 if gender_str == "male" else 0.0

        # Build feature array (must match the column order used in train.py)
        features = np.array([[
            age,
            gender_encoded,
            chronic_condition_count,
            prior_12m_cost,
            er_visit_count_12m,
            inpatient_days_12m,
            comorbidity_index,
        ]])

        # Scale -> predict (log space) -> expm1() -> dollars
        # Design spec: model predicts log1p(cost); apply expm1() at inference.
        # Guardrail: clip log prediction to training range before expm1() to
        # prevent extreme extrapolation on out-of-distribution inputs.
        # Training data max cost: ~$192K -> log1p(192010) ~= 12.17
        # Cap: expm1(12.5) ~= $268K  (reasonable upper bound for a member)
        try:
            scaled         = scaler.transform(features)
            log_pred       = float(model.predict(scaled)[0])
            log_pred       = min(log_pred, 12.5)          # guardrail in log space
            predicted_cost = float(np.expm1(log_pred))   # design spec inverse
            predicted_cost = max(0.0, predicted_cost)     # no negative costs
        except Exception as e:
            self._send_json(500, {"error": f"Prediction failed: {str(e)}"})
            return

        result = {
            "predicted_cost": round(predicted_cost, 2),
            "risk_level":     get_risk_level(predicted_cost),
        }
        self._send_json(200, result)
