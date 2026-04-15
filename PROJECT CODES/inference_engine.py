"""
=============================================================================
  FAULT DIAGNOSTIC SYSTEM FOR DISTRIBUTION TRANSFORMER
  Module 3: Inference Engine & Alert Generator (7 Fault Classes)
=============================================================================
"""

import os, json, joblib
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List

MODEL_DIR  = "./models"
REPORT_DIR = "./reports"
os.makedirs(REPORT_DIR, exist_ok=True)

FAULT_LABELS = {
    0: "Normal",
    1: "Overheating",
    2: "Low_Oil_Level",
    3: "Short_Circuit",
    4: "Open_Circuit",
    5: "Over_Voltage",
    6: "Over_Current",
}

FAULT_SEVERITY = {
    0: "INFO",
    1: "CRITICAL",
    2: "HIGH",
    3: "CRITICAL",
    4: "HIGH",
    5: "HIGH",
    6: "CRITICAL",
}

FAULT_ACTIONS = {
    0: ["Continue normal monitoring.",
        "Schedule next preventive maintenance per standard plan."],
    1: ["URGENT: Reduce load — shed non-critical feeders immediately.",
        "Inspect and clean oil cooling radiators / fins.",
        "Check ventilation around transformer enclosure.",
        "Dispatch field technician within 2 hours.",
        "Prepare spare transformer for possible swap-out."],
    2: ["Inspect tank for visible oil leaks at gaskets, bushings and drain plugs.",
        "Top up oil using approved insulating oil (e.g. Nynas Nytro 10X).",
        "Perform dielectric strength (BDV) test on oil after topping up.",
        "Dispatch field technician within 24 hours."],
    3: ["URGENT: Trip and isolate transformer from service immediately.",
        "Do NOT re-energise until fault is cleared.",
        "Notify utility control room and protection engineers.",
        "Perform insulation resistance (Megger) and winding resistance tests.",
        "Schedule emergency repair or transformer replacement."],
    4: ["Check HV and LV fuses / circuit breakers on all three phases.",
        "Inspect cable terminations and busbar connections for open joints.",
        "Measure per-phase continuity with clamp meter before re-energising.",
        "Dispatch field technician within 4 hours."],
    5: ["Check HV supply voltage at incoming busbar and tap-changer position.",
        "Adjust on-load/off-load tap-changer to lower output voltage.",
        "Inspect voltage regulator / AVR if fitted.",
        "Monitor for 30 minutes after adjustment; escalate if over-voltage persists."],
    6: ["Identify and disconnect overloaded feeders or downstream faults.",
        "Check for downstream short circuits causing sustained over-current.",
        "Verify overcurrent relay / fuse ratings and operation.",
        "Reduce connected load below 80 % rated kVA.",
        "Dispatch field technician if current does not normalise within 15 minutes."],
}

THRESHOLDS = {
    "temp_winding_C":    {"warn": 95,  "crit": 110},
    "temp_oil_C":        {"warn": 80,  "crit": 95},
    "oil_level_pct":     {"warn": 70,  "crit": 55,  "low": True},
    "voltage_avg_V":     {"warn": 252, "crit": 264},   # +5% / +10% of 240 V
    "current_avg_A":     {"warn": 278, "crit": 334},   # 100% / 120% rated
    "voltage_unbal_pct": {"warn": 2.0, "crit": 5.0},
    "current_unbal_pct": {"warn": 10,  "crit": 20},
    "resistance_pu":     {"warn": 5.0, "crit": 10.0},  # open-circuit proxy
}


@dataclass
class SensorReading:
    timestamp:       str   = field(default_factory=lambda: datetime.now().isoformat())
    transformer_id:  str   = "TXF-001"
    oil_level_pct:   float = 0.0
    temp_oil_C:      float = 0.0
    temp_winding_C:  float = 0.0
    voltage_A_V:     float = 0.0
    voltage_B_V:     float = 0.0
    voltage_C_V:     float = 0.0
    current_A_A:     float = 0.0
    current_B_A:     float = 0.0
    current_C_A:     float = 0.0
    power_factor:    float = 0.88
    resistance_pu:   float = 1.0


@dataclass
class DiagnosticReport:
    timestamp:            str
    transformer_id:       str
    fault_code:           int
    fault_name:           str
    severity:             str
    confidence_pct:       float
    class_probabilities:  dict
    threshold_violations: list
    recommended_actions:  list
    computed_features:    dict


class TransformerDiagnosticEngine:

    def __init__(self,
                 model_path:  str = f"{MODEL_DIR}/best_model.pkl",
                 scaler_path: str = f"{MODEL_DIR}/scaler.pkl",
                 meta_path:   str = f"{MODEL_DIR}/metadata.pkl"):
        self.model         = joblib.load(model_path)
        self.scaler        = joblib.load(scaler_path)
        meta               = joblib.load(meta_path)
        self.feature_names = meta["feature_names"]
        self.class_names   = meta["class_names"]
        print(f"[Engine] Model loaded: {meta['best_model_name']}")

    @staticmethod
    def engineer_features(r: SensorReading) -> dict:
        RATED_KVA  = 200
        v_mean = (r.voltage_A_V + r.voltage_B_V + r.voltage_C_V) / 3
        i_mean = (r.current_A_A + r.current_B_A + r.current_C_A) / 3

        v_unbal = (max(abs(r.voltage_A_V - v_mean),
                       abs(r.voltage_B_V - v_mean),
                       abs(r.voltage_C_V - v_mean)) / (v_mean + 1e-9)) * 100
        i_unbal = (max(abs(r.current_A_A - i_mean),
                       abs(r.current_B_A - i_mean),
                       abs(r.current_C_A - i_mean)) / (i_mean + 1e-9)) * 100

        s_total  = ((r.voltage_A_V*r.current_A_A) +
                    (r.voltage_B_V*r.current_B_A) +
                    (r.voltage_C_V*r.current_C_A)) / 1000
        load_pct = (s_total / RATED_KVA) * 100

        return {
            "oil_level_pct":      r.oil_level_pct,
            "temp_oil_C":         r.temp_oil_C,
            "temp_winding_C":     r.temp_winding_C,
            "voltage_A_V":        r.voltage_A_V,
            "voltage_B_V":        r.voltage_B_V,
            "voltage_C_V":        r.voltage_C_V,
            "current_A_A":        r.current_A_A,
            "current_B_A":        r.current_B_A,
            "current_C_A":        r.current_C_A,
            "voltage_avg_V":      round(v_mean,  2),
            "current_avg_A":      round(i_mean,  2),
            "voltage_unbal_pct":  round(v_unbal, 3),
            "current_unbal_pct":  round(i_unbal, 3),
            "load_pct":           round(load_pct, 2),
            "delta_temp_C":       round(r.temp_winding_C - r.temp_oil_C, 2),
            "power_factor":       r.power_factor,
            "resistance_pu":      r.resistance_pu,
        }

    @staticmethod
    def check_thresholds(features: dict) -> list:
        violations = []
        for param, limits in THRESHOLDS.items():
            val = features.get(param)
            if val is None:
                continue
            low = limits.get("low", False)
            if low:
                if val <= limits["crit"]:
                    violations.append({"param": param, "value": val,
                                       "level": "CRITICAL", "threshold": limits["crit"]})
                elif val <= limits["warn"]:
                    violations.append({"param": param, "value": val,
                                       "level": "WARNING",  "threshold": limits["warn"]})
            else:
                if val >= limits["crit"]:
                    violations.append({"param": param, "value": val,
                                       "level": "CRITICAL", "threshold": limits["crit"]})
                elif val >= limits["warn"]:
                    violations.append({"param": param, "value": val,
                                       "level": "WARNING",  "threshold": limits["warn"]})
        return violations

    def diagnose(self, reading: SensorReading) -> DiagnosticReport:
        features = self.engineer_features(reading)
        X_row    = np.array([[features[f] for f in self.feature_names]])
        X_sc     = self.scaler.transform(X_row)
        pred     = int(self.model.predict(X_sc)[0])
        proba    = self.model.predict_proba(X_sc)[0]
        conf     = float(proba[pred]) * 100
        proba_d  = {self.class_names[i]: round(float(p)*100,2)
                    for i,p in enumerate(proba)}
        return DiagnosticReport(
            timestamp            = reading.timestamp,
            transformer_id       = reading.transformer_id,
            fault_code           = pred,
            fault_name           = FAULT_LABELS[pred],
            severity             = FAULT_SEVERITY[pred],
            confidence_pct       = round(conf, 2),
            class_probabilities  = proba_d,
            threshold_violations = self.check_thresholds(features),
            recommended_actions  = FAULT_ACTIONS[pred],
            computed_features    = features,
        )

    @staticmethod
    def save_report(report: DiagnosticReport) -> str:
        ts = report.timestamp.replace(":","-").replace(".","-")
        path = os.path.join(REPORT_DIR, f"report_{report.transformer_id}_{ts}.json")
        with open(path,"w") as f:
            json.dump(asdict(report), f, indent=2)
        return path

    @staticmethod
    def print_alert(report: DiagnosticReport):
        SEP = "─"*60
        print(f"\n{SEP}")
        print(f"  TRANSFORMER FAULT DIAGNOSTIC REPORT")
        print(SEP)
        print(f"  ID        : {report.transformer_id}")
        print(f"  Time      : {report.timestamp}")
        print(f"  Fault     : [{report.fault_code}] {report.fault_name}")
        print(f"  Severity  : {report.severity}")
        print(f"  Confidence: {report.confidence_pct:.1f}%")
        print(f"\n  Class Probabilities:")
        for cls, prob in report.class_probabilities.items():
            bar = "█" * int(prob / 5)
            print(f"    {cls:<18} {prob:5.1f}%  {bar}")
        if report.threshold_violations:
            print(f"\n  ⚠ Threshold Violations:")
            for v in report.threshold_violations:
                print(f"    [{v['level']}] {v['param']} = {v['value']:.2f}"
                      f"  (limit: {v['threshold']})")
        print(f"\n  Recommended Actions:")
        for i, a in enumerate(report.recommended_actions, 1):
            print(f"    {i}. {a}")
        print(SEP)


def demo_inference():
    engine = TransformerDiagnosticEngine()
    RATED_I = (200*1000)/(np.sqrt(3)*415)
    RATED_V = 415/np.sqrt(3)

    scenarios = [
        SensorReading(transformer_id="TXF-001-IKEJA",
            oil_level_pct=87, temp_oil_C=56, temp_winding_C=67,
            voltage_A_V=240, voltage_B_V=240, voltage_C_V=240,
            current_A_A=185, current_B_A=182, current_C_A=183,
            power_factor=0.89, resistance_pu=1.01),

        SensorReading(transformer_id="TXF-002-OSHODI",
            oil_level_pct=81, temp_oil_C=106, temp_winding_C=138,
            voltage_A_V=239, voltage_B_V=238, voltage_C_V=239,
            current_A_A=350, current_B_A=355, current_C_A=348,
            power_factor=0.86, resistance_pu=0.98),

        SensorReading(transformer_id="TXF-003-SURULERE",
            oil_level_pct=44, temp_oil_C=79, temp_winding_C=91,
            voltage_A_V=240, voltage_B_V=240, voltage_C_V=240,
            current_A_A=192, current_B_A=190, current_C_A=191,
            power_factor=0.86, resistance_pu=1.0),

        SensorReading(transformer_id="TXF-004-YABA-ShortCircuit",
            oil_level_pct=86, temp_oil_C=95, temp_winding_C=145,
            voltage_A_V=110, voltage_B_V=108, voltage_C_V=109,
            current_A_A=2100, current_B_A=2150, current_C_A=2080,
            power_factor=0.60, resistance_pu=0.08),

        SensorReading(transformer_id="TXF-005-APAPA-OpenCircuit",
            oil_level_pct=88, temp_oil_C=60, temp_winding_C=74,
            voltage_A_V=295, voltage_B_V=241, voltage_C_V=240,
            current_A_A=1.5, current_B_A=190, current_C_A=188,
            power_factor=0.88, resistance_pu=14.5),

        SensorReading(transformer_id="TXF-006-VI-OverVoltage",
            oil_level_pct=87, temp_oil_C=65, temp_winding_C=80,
            voltage_A_V=292, voltage_B_V=290, voltage_C_V=291,
            current_A_A=210, current_B_A=212, current_C_A=211,
            power_factor=0.83, resistance_pu=1.0),

        SensorReading(transformer_id="TXF-007-MUSHIN-OverCurrent",
            oil_level_pct=86, temp_oil_C=88, temp_winding_C=118,
            voltage_A_V=228, voltage_B_V=227, voltage_C_V=228,
            current_A_A=480, current_B_A=475, current_C_A=478,
            power_factor=0.82, resistance_pu=0.97),
    ]

    reports = []
    for s in scenarios:
        r = engine.diagnose(s)
        engine.print_alert(r)
        path = engine.save_report(r)
        print(f"  [Saved] {path}")
        reports.append(r)
    return reports


if __name__ == "__main__":
    demo_inference()
