"""
=============================================================================
  FAULT DIAGNOSTIC SYSTEM FOR DISTRIBUTION TRANSFORMER — main.py
  Run: python main.py
=============================================================================
"""
import time

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   FAULT DIAGNOSTIC SYSTEM FOR DISTRIBUTION TRANSFORMER          ║
║   200 kVA | 11 kV / 415 V  |  Nigeria Standard                  ║
║                                                                  ║
║   Fault Classes: Normal | Overheating | Low Oil Level            ║
║                  Short Circuit | Open Circuit                    ║
║                  Over Voltage  | Over Current                    ║
║                                                                  ║
║   Authors: Maku James Oluwatosin    (20201749)                   ║
║            Eniyangbagbe Oluwaniyomi Enoch (20201740)             ║
╚══════════════════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)

    print("="*66)
    print("  PHASE 1 — DATA GENERATION & MODEL TRAINING")
    print("="*66)
    t0 = time.time()
    from train_models import train_all_models
    trained, results, feature_names, scaler, X_test, y_test = train_all_models()
    print(f"\n  ✓ Training complete ({time.time()-t0:.1f}s)")

    print("\n"+"="*66)
    print("  PHASE 2 — REAL-TIME INFERENCE DEMO (7 scenarios)")
    print("="*66)
    from inference_engine import demo_inference
    reports = demo_inference()
    print(f"\n  ✓ {len(reports)} diagnostic reports generated")

    print("\n"+"="*66)
    print("  RESULTS SUMMARY")
    print("="*66)
    print(f"\n  {'Model':<22}  {'Accuracy':>9}  {'F1-Macro':>9}")
    print(f"  {'-'*44}")
    for name, res in results.items():
        print(f"  {name:<22}  {res['accuracy']:>9.4f}  {res['f1_macro']:>9.4f}")

    best = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"\n  ★ Best Model : {best}")
    print(f"  ★ Accuracy   : {results[best]['accuracy']:.4f}")
    print(f"  ★ F1 (macro) : {results[best]['f1_macro']:.4f}")
    print("\n  All plots  → ./plots/")
    print("  All models → ./models/")
    print("  Reports    → ./reports/\n")

if __name__ == "__main__":
    main()
