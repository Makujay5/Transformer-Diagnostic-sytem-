import os
import datetime
import requests as req_lib
from flask import Flask, request, jsonify
from inference_engine import TransformerDiagnosticEngine, SensorReading

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load the trained ML model once on startup ─────────────────────────────────
# The model files must exist in a folder called "models/" in the same directory
# as this file. They are created by running: python main.py
print("=" * 60)
print("  Loading Transformer Fault Diagnostic Engine ...")
engine = TransformerDiagnosticEngine(
    model_path  = "./models/best_model.pkl",
    scaler_path = "./models/scaler.pkl",
    meta_path   = "./models/metadata.pkl",
)
print("  Engine ready. Server starting ...")
print("=" * 60)

# ── SMS Alert Configuration ───────────────────────────────────────────────────
# To enable SMS alerts:
#   1. Create a free account at termii.com
#   2. Copy your API key from Settings → API Keys
#   3. Replace the empty string below with your actual key
#   4. Replace the phone number with the engineer's real Nigerian number
#
# Leave TERMII_API_KEY as an empty string "" to disable SMS (no error will occur)

TERMII_API_KEY = ""                  # paste your Termii API key here e.g. "TLtest123abc..."
ENGINEER_PHONE = "+2347054026418"    # replace with real engineer phone number


def send_sms_alert(report):
    """
    Sends an SMS to the engineer when a CRITICAL or HIGH fault is detected.
    Only runs if TERMII_API_KEY is set. Silently skips if key is empty.
    """
    if not TERMII_API_KEY:
        # SMS disabled — no API key set. Diagnosis still works normally.
        return

    if report.severity in ["CRITICAL", "HIGH"]:
        message = (
            f"ALERT [{report.severity}]\n"
            f"Transformer: {report.transformer_id}\n"
            f"Fault: {report.fault_name}\n"
            f"Confidence: {report.confidence_pct:.1f}%\n"
            f"Action: {report.recommended_actions[0]}"
        )
        try:
            response = req_lib.post(
                "https://api.ng.termii.com/api/sms/send",
                json={
                    "to":      ENGINEER_PHONE,
                    "from":    "TransfDiag",
                    "sms":     message,
                    "type":    "plain",
                    "channel": "generic",
                    "api_key": TERMII_API_KEY,
                },
                timeout=10,
            )
            if response.status_code == 200:
                print(f"[SMS] Alert sent to {ENGINEER_PHONE}")
            else:
                print(f"[SMS] Failed — status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[SMS] Error sending alert: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    """Root page — shows system status in a browser."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    <html>
    <head>
        <title>Transformer Fault Diagnostic System</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body {{ font-family: Arial, sans-serif; background: #0d1117;
                   color: #58a6ff; padding: 40px; }}
            h1   {{ color: #ffffff; }}
            h3   {{ color: #58a6ff; }}
            .box {{ background: #161b22; border: 1px solid #30363d;
                   border-radius: 8px; padding: 20px; margin: 10px 0; }}
            .ok  {{ color: #3fb950; font-weight: bold; }}
            .url {{ color: #f78166; font-family: monospace; }}
        </style>
    </head>
    <body>
        <h1>Transformer Fault Diagnostic System</h1>
        <div class="box">
            <p>Status: <span class="ok">ONLINE</span></p>
            <p>Model: SVM (RBF) &nbsp;|&nbsp; Accuracy: 100%</p>
            <p>Rating: 200 kVA &nbsp;|&nbsp; 11 kV / 415 V &nbsp;|&nbsp; Nigeria</p>
            <p>Fault Classes: Normal, Overheating, Low Oil Level,
               Short Circuit, Open Circuit, Over Voltage, Over Current</p>
            <p>Last checked: {now}</p>
        </div>
        <div class="box">
            <h3>API Endpoint</h3>
            <p>Send a POST request to:
               <span class="url">/diagnose</span>
               with JSON sensor data to receive a fault diagnosis.</p>
            <p>Check status:
               <span class="url">/status</span></p>
        </div>
        <div class="box">
            <h3>Authors</h3>
            <p>Maku James Oluwatosin (20201749)</p>
            <p>Eniyangbagbe Oluwaniyomi Enoch (20201740)</p>
        </div>
    </body>
    </html>
    """


@app.route("/status", methods=["GET"])
def status():
    """Returns a simple JSON health check — useful for monitoring."""
    return jsonify({
        "status":       "online",
        "model":        "SVM_RBF",
        "accuracy":     "100%",
        "fault_classes": 7,
        "transformer":  "200 kVA | 11kV/415V",
        "timestamp":    datetime.datetime.now().isoformat(),
    })


@app.route("/diagnose", methods=["POST"])
def diagnose():
    """
    Main endpoint. Receives sensor readings and returns a fault diagnosis.

    Expected JSON body (all fields required):
    {
        "transformer_id":  "TXF-001",     (any string label)
        "oil_level_pct":   87.0,          (0 to 100 %)
        "temp_oil_C":      55.0,          (degrees Celsius)
        "temp_winding_C":  65.0,          (degrees Celsius)
        "voltage_A_V":     240.0,         (Volts)
        "voltage_B_V":     240.0,
        "voltage_C_V":     240.0,
        "current_A_A":     185.0,         (Amperes)
        "current_B_A":     182.0,
        "current_C_A":     183.0,
        "power_factor":    0.89,          (0.0 to 1.0)
        "resistance_pu":   1.0            (per-unit, normal = 1.0)
    }

    Returns JSON with:
        fault, severity, confidence, actions, violations, timestamp
    """
    # ── Parse incoming JSON ───────────────────────────────────────────────────
    data = request.get_json(force=True, silent=True)

    if data is None:
        return jsonify({"error": "Invalid JSON. Check your request body."}), 400

    # ── Required fields check ─────────────────────────────────────────────────
    required = [
        "transformer_id", "oil_level_pct", "temp_oil_C", "temp_winding_C",
        "voltage_A_V", "voltage_B_V", "voltage_C_V",
        "current_A_A", "current_B_A", "current_C_A",
        "power_factor", "resistance_pu",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({
            "error":   "Missing fields in request",
            "missing": missing,
            "hint":    "All sensor fields are required. See / for documentation.",
        }), 400

    # ── Run diagnosis ─────────────────────────────────────────────────────────
    try:
        reading = SensorReading(**data)
        report  = engine.diagnose(reading)
        engine.print_alert(report)   # prints to Render logs for monitoring
        send_sms_alert(report)       # sends SMS if API key is configured

        return jsonify({
            "transformer_id": report.transformer_id,
            "timestamp":      report.timestamp,
            "fault_code":     report.fault_code,
            "fault":          report.fault_name,
            "severity":       report.severity,
            "confidence_pct": report.confidence_pct,
            "probabilities":  report.class_probabilities,
            "violations":     report.threshold_violations,
            "actions":        report.recommended_actions,
        })

    except Exception as e:
        return jsonify({
            "error":  "Diagnosis failed",
            "detail": str(e),
            "hint":   "Check that all sensor values are numbers, not strings.",
        }), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Server running at http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)
