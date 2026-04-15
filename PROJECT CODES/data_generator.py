"""
=============================================================================
  FAULT DIAGNOSTIC SYSTEM FOR DISTRIBUTION TRANSFORMER
  Module 1: Refined Synthetic Data Generator (5 Target Fault Parameters)
=============================================================================
  Authors : Maku James Oluwatosin (20201749)
           & Eniyangbagbe Oluwaniyomi Enoch (20201740)

  Fault Classes (6 total):
    0 - Normal Operation
    1 - Overheating (Temperature Fault)
    2 - Low Oil Level (Oil Fault)
    3 - Short Circuit
    4 - Open Circuit
    5 - Over Voltage
    6 - Over Current

  Transformer Nameplate: 200 kVA | 11 kV / 415 V | Nigeria Standard
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

SEED = 42
np.random.seed(SEED)

# ── Transformer nameplate constants ──────────────────────────────────────────
RATED_KVA       = 200
RATED_HV_KV     = 11.0
RATED_LV_V      = 415.0
RATED_CURRENT_A = (RATED_KVA * 1000) / (np.sqrt(3) * RATED_LV_V)  # ≈ 278 A
RATED_PHASE_V   = RATED_LV_V / np.sqrt(3)                          # ≈ 239.6 V

# ── Fault definitions ────────────────────────────────────────────────────────
FAULT_LABELS = {
    0: "Normal",
    1: "Overheating",
    2: "Low_Oil_Level",
    3: "Short_Circuit",
    4: "Open_Circuit",
    5: "Over_Voltage",
    6: "Over_Current",
}
N_CLASSES = len(FAULT_LABELS)


def _noise(size, std=1.0):
    return np.random.normal(0, std, size)


def generate_transformer_data(n_samples_per_class: int = 2000,
                               save_csv: bool = True,
                               output_dir: str = ".") -> pd.DataFrame:
    """
    Generate physics-inspired sensor readings for a 200 kVA distribution
    transformer under 7 operating conditions (Normal + 6 fault types).

    Monitored Parameters
    --------------------
    Primary (your 5 fault targets):
      • oil_level_pct          - Tank oil level (float sensor, %)
      • temp_oil_C             - Oil temperature (°C)
      • temp_winding_C         - Winding hot-spot temperature (°C)
      • voltage_A/B/C_V        - Per-phase secondary voltages (V)
      • current_A/B/C_A        - Per-phase load currents (A)

    Derived / supporting:
      • voltage_avg_V          - Mean 3-phase voltage
      • current_avg_A          - Mean 3-phase current
      • voltage_unbal_pct      - Voltage unbalance index (%)
      • current_unbal_pct      - Current unbalance index (%)
      • load_pct               - % of rated kVA loading
      • delta_temp_C           - Winding temp minus oil temp
      • power_factor           - Estimated power factor
      • resistance_pu          - Estimated winding resistance (p.u.)
                                 drops sharply in short circuit,
                                 rises to inf proxy in open circuit
    """
    records = []

    for label, fault_name in FAULT_LABELS.items():
        n = n_samples_per_class

        # ── Baseline (normal operating envelope) ──────────────────────────
        temp_winding = np.random.normal(65,  5, n)
        temp_oil     = np.random.normal(55,  4, n)
        oil_level    = np.random.normal(87,  3, n)

        v_a = np.random.normal(RATED_PHASE_V, 2.5, n)
        v_b = np.random.normal(RATED_PHASE_V, 2.5, n)
        v_c = np.random.normal(RATED_PHASE_V, 2.5, n)

        i_a = np.random.normal(RATED_CURRENT_A * 0.70, 8, n)
        i_b = np.random.normal(RATED_CURRENT_A * 0.70, 8, n)
        i_c = np.random.normal(RATED_CURRENT_A * 0.70, 8, n)

        pf           = np.random.normal(0.88, 0.02, n)
        resistance_pu = np.random.normal(1.0,  0.05, n)   # per-unit winding R

        # ── Fault-specific perturbations ──────────────────────────────────

        # ── 1. OVERHEATING ─────────────────────────────────────────────
        if label == 1:
            temp_winding += np.random.uniform(30, 65, n)   # hot-spot rise
            temp_oil     += np.random.uniform(20, 40, n)
            oil_level    -= np.random.uniform(3,  10, n)   # evaporation
            # Overload drives the heat
            i_a += np.random.uniform(40, 90, n)
            i_b += np.random.uniform(40, 90, n)
            i_c += np.random.uniform(40, 90, n)
            pf  -= np.random.uniform(0.01, 0.04, n)

        # ── 2. LOW OIL LEVEL ───────────────────────────────────────────
        elif label == 2:
            oil_level    -= np.random.uniform(30, 60, n)   # major oil loss
            temp_winding += np.random.uniform(15, 35, n)   # less cooling
            temp_oil     += np.random.uniform(10, 25, n)
            pf           -= np.random.uniform(0.02, 0.06, n)

        # ── 3. SHORT CIRCUIT ───────────────────────────────────────────
        elif label == 3:
            # Current surges dramatically (typically 5–20× rated)
            sc_factor = np.random.uniform(4, 18, n)
            i_a *= sc_factor
            i_b *= sc_factor
            i_c *= sc_factor
            # Voltage collapses at fault point
            v_a -= np.random.uniform(30, 120, n)
            v_b -= np.random.uniform(30, 120, n)
            v_c -= np.random.uniform(30, 120, n)
            temp_winding += np.random.uniform(40, 80, n)   # rapid heating
            temp_oil     += np.random.uniform(20, 45, n)
            resistance_pu -= np.random.uniform(0.5, 0.9, n)  # R drops sharply
            pf            -= np.random.uniform(0.10, 0.30, n)

        # ── 4. OPEN CIRCUIT ────────────────────────────────────────────
        elif label == 4:
            # One phase current drops to near zero
            open_phase = np.random.randint(0, 3, n)  # 0=A, 1=B, 2=C
            i_a = np.where(open_phase == 0,
                           np.random.normal(2, 0.5, n),   # nearly zero
                           i_a)
            i_b = np.where(open_phase == 1,
                           np.random.normal(2, 0.5, n),
                           i_b)
            i_c = np.where(open_phase == 2,
                           np.random.normal(2, 0.5, n),
                           i_c)
            # Corresponding voltage on open phase rises (floating)
            v_a = np.where(open_phase == 0,
                           v_a + np.random.uniform(20, 60, n),
                           v_a)
            v_b = np.where(open_phase == 1,
                           v_b + np.random.uniform(20, 60, n),
                           v_b)
            v_c = np.where(open_phase == 2,
                           v_c + np.random.uniform(20, 60, n),
                           v_c)
            resistance_pu += np.random.uniform(5, 20, n)   # R spikes (near open)
            temp_winding  += np.random.uniform(5, 20, n)   # slight rise on loaded phases

        # ── 5. OVER VOLTAGE ────────────────────────────────────────────
        elif label == 5:
            ov_factor = np.random.uniform(1.10, 1.35, n)   # 110–135% of rated V
            v_a *= ov_factor
            v_b *= ov_factor
            v_c *= ov_factor
            temp_winding += np.random.uniform(5,  20, n)   # core losses increase
            temp_oil     += np.random.uniform(3,  12, n)
            # Magnetising current rises with over-excitation
            i_a += np.random.uniform(5, 25, n)
            i_b += np.random.uniform(5, 25, n)
            i_c += np.random.uniform(5, 25, n)
            pf  -= np.random.uniform(0.02, 0.08, n)        # lagging increases

        # ── 6. OVER CURRENT ────────────────────────────────────────────
        elif label == 6:
            oc_factor = np.random.uniform(1.20, 2.50, n)   # 120–250% rated I
            i_a *= oc_factor
            i_b *= oc_factor
            i_c *= oc_factor
            temp_winding += np.random.uniform(20, 55, n)   # I²R heating
            temp_oil     += np.random.uniform(10, 30, n)
            # Voltage sags under heavy load
            v_a -= np.random.uniform(5, 20, n)
            v_b -= np.random.uniform(5, 20, n)
            v_c -= np.random.uniform(5, 20, n)
            pf  -= np.random.uniform(0.03, 0.08, n)

        # ── Derived features ───────────────────────────────────────────
        v_mean  = (v_a + v_b + v_c) / 3
        i_mean  = (i_a + i_b + i_c) / 3

        v_unbal = (np.max(np.abs(np.stack([v_a - v_mean,
                                            v_b - v_mean,
                                            v_c - v_mean], axis=0)), axis=0)
                   / (v_mean + 1e-9)) * 100

        i_unbal = (np.max(np.abs(np.stack([i_a - i_mean,
                                            i_b - i_mean,
                                            i_c - i_mean], axis=0)), axis=0)
                   / (i_mean + 1e-9)) * 100

        # Three-phase apparent power → load %
        s_total  = ((v_a * i_a) + (v_b * i_b) + (v_c * i_c)) / 1000
        load_pct = (s_total / RATED_KVA) * 100

        delta_temp = temp_winding - temp_oil

        # ── Physical clipping ──────────────────────────────────────────
        oil_level    = np.clip(oil_level,    0,   100)
        temp_winding = np.clip(temp_winding, 20,  250)
        temp_oil     = np.clip(temp_oil,     20,  180)
        pf           = np.clip(pf,           0.3, 1.0)
        i_a          = np.clip(i_a,          0,   5000)
        i_b          = np.clip(i_b,          0,   5000)
        i_c          = np.clip(i_c,          0,   5000)
        v_a          = np.clip(v_a,          0,   500)
        v_b          = np.clip(v_b,          0,   500)
        v_c          = np.clip(v_c,          0,   500)
        resistance_pu = np.clip(resistance_pu, 0.01, 25)

        for idx in range(n):
            records.append({
                # ── Primary fault-target parameters ────────────────────
                "oil_level_pct":      round(float(oil_level[idx]),    2),
                "temp_oil_C":         round(float(temp_oil[idx]),     2),
                "temp_winding_C":     round(float(temp_winding[idx]), 2),
                "voltage_A_V":        round(float(v_a[idx]),          2),
                "voltage_B_V":        round(float(v_b[idx]),          2),
                "voltage_C_V":        round(float(v_c[idx]),          2),
                "current_A_A":        round(float(i_a[idx]),          2),
                "current_B_A":        round(float(i_b[idx]),          2),
                "current_C_A":        round(float(i_c[idx]),          2),
                # ── Derived / supporting features ───────────────────────
                "voltage_avg_V":      round(float(v_mean[idx]),       2),
                "current_avg_A":      round(float(i_mean[idx]),       2),
                "voltage_unbal_pct":  round(float(v_unbal[idx]),      3),
                "current_unbal_pct":  round(float(i_unbal[idx]),      3),
                "load_pct":           round(float(load_pct[idx]),     2),
                "delta_temp_C":       round(float(delta_temp[idx]),   2),
                "power_factor":       round(float(pf[idx]),           4),
                "resistance_pu":      round(float(resistance_pu[idx]),4),
                # ── Target ─────────────────────────────────────────────
                "fault_label": label,
                "fault_name":  fault_name,
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "transformer_fault_data.csv")
        df.to_csv(path, index=False)
        print(f"[DataGenerator] Saved → {path}")
        print(f"[DataGenerator] Shape: {df.shape}")
        print(f"[DataGenerator] Class distribution:\n{df['fault_name'].value_counts().to_string()}")

    return df


def load_and_preprocess(df: pd.DataFrame,
                         test_size:   float = 0.20,
                         val_size:    float = 0.10,
                         scaler_path: str   = "./models/scaler.pkl"):
    feature_cols = [c for c in df.columns if c not in ("fault_label", "fault_name")]
    X = df[feature_cols].values
    y = df["fault_label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=SEED, stratify=y)
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - rel_val), random_state=SEED, stratify=y_temp)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[Preprocessing] Scaler saved  → {scaler_path}")
    print(f"[Preprocessing] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler


if __name__ == "__main__":
    df = generate_transformer_data(n_samples_per_class=2000, save_csv=True, output_dir=".")
    print("\nMean sensor values per fault class:")
    print(df.groupby("fault_name")[[
        "oil_level_pct", "temp_winding_C", "voltage_avg_V",
        "current_avg_A", "resistance_pu"
    ]].mean().round(2).to_string())
