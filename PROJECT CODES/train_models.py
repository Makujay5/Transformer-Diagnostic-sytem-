"""
=============================================================================
  FAULT DIAGNOSTIC SYSTEM FOR DISTRIBUTION TRANSFORMER
  Module 2: ML Training Pipeline (7 Classes)
=============================================================================
  Fault Classes:
    0-Normal | 1-Overheating | 2-Low_Oil_Level | 3-Short_Circuit
    4-Open_Circuit | 5-Over_Voltage | 6-Over_Current

  Models: Random Forest | Gradient Boosting | SVM (RBF) | ANN (MLP)
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve

from data_generator import (
    generate_transformer_data, load_and_preprocess,
    FAULT_LABELS, N_CLASSES,
)

warnings.filterwarnings("ignore")
SEED = 42

MODEL_DIR = "./models"
PLOT_DIR  = "./plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

CLASS_NAMES = list(FAULT_LABELS.values())
PALETTE = ["#2ECC71","#E74C3C","#F39C12","#9B59B6","#3498DB","#E91E63","#FF9800"]


# ─── Confusion Matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax, annot_kws={"size": 10})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, rotation=0,  fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"cm_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [Plot] Confusion matrix  → {path}")


# ─── Feature Importance ───────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp)))
    ax.bar(range(len(imp)), imp[idx], color=colors[np.argsort(np.argsort(imp[::-1]))])
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importances — {model_name}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"feat_imp_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [Plot] Feature importance → {path}")


# ─── Per-class Accuracy Bar ───────────────────────────────────────────────────
def plot_per_class_metrics(report_dict, model_name):
    classes  = CLASS_NAMES
    precision = [report_dict[c]["precision"] for c in classes]
    recall    = [report_dict[c]["recall"]    for c in classes]
    f1        = [report_dict[c]["f1-score"]  for c in classes]

    x     = np.arange(len(classes))
    width = 0.28
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#2196F3")
    ax.bar(x,         recall,    width, label="Recall",    color="#4CAF50")
    ax.bar(x + width, f1,        width, label="F1-Score",  color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Class Metrics — {model_name}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"per_class_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [Plot] Per-class metrics  → {path}")


# ─── Model Comparison ─────────────────────────────────────────────────────────
def plot_model_comparison(results):
    names  = list(results.keys())
    acc    = [results[n]["accuracy"]  for n in names]
    f1_mac = [results[n]["f1_macro"]  for n in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, acc,    w, label="Accuracy",   color="#2196F3", edgecolor="white")
    b2 = ax.bar(x + w/2, f1_mac, w, label="F1 (macro)", color="#4CAF50", edgecolor="white")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.80, 1.01)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — 7-Class Transformer Fault Classifier",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Model comparison       → {path}")


# ─── ROC Curves ──────────────────────────────────────────────────────────────
def plot_roc(model, X_test, y_test, model_name):
    y_bin   = label_binarize(y_test, classes=list(range(N_CLASSES)))
    y_score = model.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i],
                label=f"{CLASS_NAMES[i]}  (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curves (OvR) — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"roc_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [Plot] ROC curves         → {path}")


# ─── Learning Curve ───────────────────────────────────────────────────────────
def plot_learning_curve(model, X_train, y_train, model_name):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
    tm, ts = train_scores.mean(axis=1), train_scores.std(axis=1)
    vm, vs = val_scores.mean(axis=1),   val_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(train_sizes, tm-ts, tm+ts, alpha=0.12, color="#2196F3")
    ax.fill_between(train_sizes, vm-vs, vm+vs, alpha=0.12, color="#4CAF50")
    ax.plot(train_sizes, tm, "o-", color="#2196F3", lw=2, label="Training F1")
    ax.plot(train_sizes, vm, "s-", color="#4CAF50", lw=2, label="CV F1")
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("F1 (macro)",        fontsize=12)
    ax.set_title(f"Learning Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"learning_curve_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [Plot] Learning curve     → {path}")


# ─── Sensor Distribution ──────────────────────────────────────────────────────
def plot_sensor_distributions(df):
    sensors = ["oil_level_pct","temp_winding_C","temp_oil_C",
               "voltage_avg_V","current_avg_A","resistance_pu",
               "voltage_unbal_pct","current_unbal_pct"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, sensor in enumerate(sensors):
        for j, (lbl, name) in enumerate(FAULT_LABELS.items()):
            axes[i].hist(df[df["fault_label"]==lbl][sensor],
                         bins=35, alpha=0.55, color=PALETTE[j],
                         label=name, edgecolor="none")
        axes[i].set_title(sensor.replace("_"," "), fontsize=9, fontweight="bold")
        axes[i].set_xlabel("Value", fontsize=8)
        axes[i].tick_params(labelsize=7)
        if i == 0:
            axes[i].legend(fontsize=6, loc="upper right")
    fig.suptitle("Sensor Value Distributions by Fault Class\n"
                 "200 kVA Distribution Transformer | Nigeria",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "sensor_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Sensor distributions   → {path}")


# ─── 24-Hour Monitoring Simulation ───────────────────────────────────────────
def plot_timeseries_simulation(model, scaler, feature_names):
    """Simulates a transformer drifting from normal → over-current → recovery."""
    import matplotlib.gridspec as gridspec
    n  = 48  # 30-min intervals over 24 hrs
    hr = np.linspace(0, 24, n)

    # Load ramps up to 140% then is shed at hr 18
    load = np.where(hr < 6,  0.60,
           np.where(hr < 12, 0.60 + 0.05*(hr-6),
           np.where(hr < 18, 0.90 + 0.025*(hr-12),
                              0.65)))

    RATED_CURRENT_A_val = (200*1000)/(np.sqrt(3)*415)
    RATED_PHASE_V_val   = 415/np.sqrt(3)

    rows = []
    for lf in load:
        I  = RATED_CURRENT_A_val * lf + np.random.normal(0, 3)
        Tw = 65 + 45*lf + np.random.normal(0, 2)
        To = 55 + 30*lf + np.random.normal(0, 1.5)
        Va = RATED_PHASE_V_val - 8*lf + np.random.normal(0,1.2)
        row = {
            "oil_level_pct":     87 - 2*lf + np.random.normal(0,0.4),
            "temp_oil_C":        To,
            "temp_winding_C":    Tw,
            "voltage_A_V":       Va,
            "voltage_B_V":       Va + np.random.normal(0,0.8),
            "voltage_C_V":       Va + np.random.normal(0,0.8),
            "current_A_A":       I,
            "current_B_A":       I + np.random.normal(0,3),
            "current_C_A":       I + np.random.normal(0,3),
            "voltage_avg_V":     Va,
            "current_avg_A":     I,
            "voltage_unbal_pct": 0.4,
            "current_unbal_pct": 1.2,
            "load_pct":          lf*100,
            "delta_temp_C":      Tw - To,
            "power_factor":      0.88 - 0.03*lf,
            "resistance_pu":     1.0 - 0.01*lf,
        }
        rows.append(row)

    df_ts    = pd.DataFrame(rows)
    X_scaled = scaler.transform(df_ts[feature_names].values)
    probas   = model.predict_proba(X_scaled)
    preds    = model.predict(X_scaled)

    COLOR_MAP = {n: PALETTE[i] for i, n in enumerate(CLASS_NAMES)}

    fig = plt.figure(figsize=(15,10))
    gs  = gridspec.GridSpec(3,2, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1:,1])
    ax5 = fig.add_subplot(gs[2,0])

    ax1.plot(hr, df_ts["temp_winding_C"], color="#E74C3C", lw=2)
    ax1.axhline(95,  ls="--", color="orange", lw=1, label="Warn 95°C")
    ax1.axhline(110, ls="--", color="red",    lw=1, label="Crit 110°C")
    ax1.set_title("Winding Temperature (°C)", fontweight="bold")
    ax1.set_xlabel("Hour"); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ax2.plot(hr, df_ts["current_avg_A"], color="#3498DB", lw=2)
    ax2.axhline(RATED_CURRENT_A_val,       ls="--", color="orange", lw=1, label="Rated")
    ax2.axhline(RATED_CURRENT_A_val*1.20,  ls="--", color="red",    lw=1, label="120% Warn")
    ax2.set_title("Average Load Current (A)", fontweight="bold")
    ax2.set_xlabel("Hour"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    ax3.plot(hr, df_ts["oil_level_pct"], color="#F39C12", lw=2)
    ax3.axhline(70, ls="--", color="orange", lw=1, label="Warn 70%")
    ax3.axhline(55, ls="--", color="red",    lw=1, label="Crit 55%")
    ax3.set_title("Oil Level (%)", fontweight="bold")
    ax3.set_xlabel("Hour"); ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    for i, name in enumerate(CLASS_NAMES):
        ax4.plot(hr, probas[:,i]*100, lw=2, color=PALETTE[i], label=name)
    ax4.axhline(50, color="gray", lw=1, ls=":", label="50% threshold")
    ax4.set_title("Fault Probability Over 24 Hours", fontweight="bold")
    ax4.set_xlabel("Hour"); ax4.set_ylabel("Probability (%)")
    ax4.legend(fontsize=7, loc="upper left"); ax4.grid(alpha=0.3)
    ax4.set_ylim(-2,102)

    pred_labels = [CLASS_NAMES[p] for p in preds]
    colors_ts   = [COLOR_MAP[l] for l in pred_labels]
    ax5.scatter(hr, [1]*n, c=colors_ts, s=90, zorder=3)
    ax5.set_yticks([]); ax5.set_xlabel("Hour")
    ax5.set_title("Predicted Health State", fontweight="bold")
    from matplotlib.patches import Patch
    ax5.legend(handles=[Patch(facecolor=PALETTE[i], label=n)
                        for i,n in enumerate(CLASS_NAMES)],
               fontsize=7, loc="upper right")
    ax5.grid(axis="x", alpha=0.3)

    fig.suptitle("24-Hour Monitoring Simulation — TXF-001 | 200 kVA | Lagos, Nigeria",
                 fontsize=13, fontweight="bold")
    path = os.path.join(PLOT_DIR, "timeseries_monitoring.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Time-series monitoring → {path}")


# ─── Evaluate one model ───────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    print(f"\n{'='*62}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*62}")
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 (macro) : {f1:.4f}")
    print("\n  Classification Report:")
    report_str  = classification_report(y_test, y_pred,
                                         target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(y_test, y_pred,
                                         target_names=CLASS_NAMES,
                                         output_dict=True)
    print(report_str)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)
    plot_per_class_metrics(report_dict, model_name)
    if feature_names and hasattr(model, "feature_importances_"):
        plot_feature_importance(model, feature_names, model_name)
    return {"accuracy": acc, "f1_macro": f1, "report": report_dict}


# ─── Main ─────────────────────────────────────────────────────────────────────
def train_all_models():
    print("\n[Step 1] Generating refined 7-class dataset …")
    df = generate_transformer_data(n_samples_per_class=2000,
                                   save_csv=True, output_dir=".")

    print("\n[Step 2] Preprocessing …")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names, scaler) = load_and_preprocess(df)

    X_tr = np.vstack([X_train, X_val])
    y_tr = np.concatenate([y_train, y_val])

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=4, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08,
            max_depth=5, subsample=0.85,
            min_samples_leaf=3, random_state=SEED),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10.0, gamma="scale",
            decision_function_shape="ovr", probability=True,
            class_weight="balanced", random_state=SEED),
        "ANN (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam",
            alpha=1e-4, learning_rate_init=0.001,
            max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=SEED),
    }

    results = {}
    trained = {}

    print("\n[Step 3] Training & evaluating …")
    for name, clf in models.items():
        print(f"\n  ► Training {name} …")
        clf.fit(X_tr, y_tr)
        trained[name] = clf
        results[name] = evaluate_model(clf, X_test, y_test, name,
            feature_names=(feature_names if hasattr(clf,"feature_importances_") else None))
        plot_roc(clf, X_test, y_test, name)
        joblib.dump(clf, os.path.join(MODEL_DIR, f"{name.replace(' ','_')}.pkl"))

    plot_model_comparison(results)

    best_name  = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = trained[best_name]
    print(f"\n{'*'*62}")
    print(f"  ★ Best Model : {best_name}")
    print(f"  Accuracy     : {results[best_name]['accuracy']:.4f}")
    print(f"  F1 (macro)   : {results[best_name]['f1_macro']:.4f}")
    print(f"{'*'*62}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    meta = {"best_model_name": best_name,
            "feature_names":   feature_names,
            "class_names":     CLASS_NAMES,
            "results":         {k: {kk:vv for kk,vv in v.items() if kk!="report"}
                                for k,v in results.items()}}
    joblib.dump(meta, os.path.join(MODEL_DIR, "metadata.pkl"))

    # Visualisations
    print("\n[Step 4] Generating extra plots …")
    plot_sensor_distributions(df)
    plot_learning_curve(best_model, X_train, y_train, best_name)
    plot_timeseries_simulation(best_model, scaler, feature_names)

    return trained, results, feature_names, scaler, X_test, y_test


if __name__ == "__main__":
    train_all_models()
