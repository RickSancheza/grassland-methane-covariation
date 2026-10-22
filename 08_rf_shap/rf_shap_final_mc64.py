"""Final 11-predictor Random Forest + Monte Carlo interventional SHAP.

Status: RECONSTRUCTED implementation of the paper figure pipeline.
- RF hyperparameters from the recovered RF script family (see config.RF_PARAMS).
- SHAP: MC interventional permutation on predict_proba[:,1],
  64 random feature orderings, train-median baseline (best match to
  SHAP_feature_importance.xlsx; not bit-identical to the original run).
- Final features = Table S9 / SHAP_feature_importance.xlsx (11 variables).
- Labels: Enteric-positive = cluster 2, Enteric-negative = cluster 4.

This is NOT recovered original source and does not claim exact SHAP values.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    FINAL_RF_FEATURES,
    MODEL_INPUT_CSV,
    NEGATIVE_CLUSTER,
    OUT_RF_SHAP,
    POSITIVE_CLUSTER,
    RF_PARAMS,
    RF_RANDOM_STATE,
    RF_TEST_SIZE,
    SHAP_N_ORDERINGS,
    SHAP_SEED,
    ensure_output_dirs,
)


def mc_interventional_perm_shap(rf, X_eval, baseline, n_order=SHAP_N_ORDERINGS, seed=SHAP_SEED):
    rng = np.random.default_rng(seed)
    n, d = X_eval.shape
    x0 = np.tile(baseline, (n, 1))
    f0 = rf.predict_proba(x0)[:, 1]
    acc = np.zeros((n, d))
    for _ in range(n_order):
        order = rng.permutation(d)
        x = x0.copy()
        prev = f0.copy()
        for j in order:
            x[:, j] = X_eval[:, j]
            cur = rf.predict_proba(x)[:, 1]
            acc[:, j] += cur - prev
            prev = cur
    phi = acc / n_order
    f1 = rf.predict_proba(X_eval)[:, 1]
    return phi, f1, f0


def main() -> None:
    ensure_output_dirs()
    if not MODEL_INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing model table: {MODEL_INPUT_CSV}")

    df = pd.read_csv(MODEL_INPUT_CSV)
    missing = [f for f in FINAL_RF_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Model table missing features: {missing}")

    if "response_binary" not in df.columns:
        df["response_binary"] = (df["cluster"] == POSITIVE_CLUSTER).astype(int)

    y = df["response_binary"].to_numpy()
    X = df[FINAL_RF_FEATURES].to_numpy(dtype=float)

    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(**RF_PARAMS)),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=RF_TEST_SIZE, random_state=RF_RANDOM_STATE, stratify=y
    )
    pipe.fit(X_train, y_train)
    rf = pipe.named_steps["rf"]
    Xtr = pipe.named_steps["impute"].transform(X_train)
    Xte = pipe.named_steps["impute"].transform(X_test)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RF_RANDOM_STATE)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    metrics = {
        "status": "RECONSTRUCTED",
        "n_samples": int(len(df)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "features": FINAL_RF_FEATURES,
        "rf_params": RF_PARAMS,
        "test_size": RF_TEST_SIZE,
        "random_state": RF_RANDOM_STATE,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_auc": float(roc_auc_score(y_test, prob)),
        "test_precision": float(precision_score(y_test, pred)),
        "test_recall": float(recall_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred)),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "shap_method": (
            "MC interventional permutation SHAP on predict_proba[:,1]; "
            f"{SHAP_N_ORDERINGS} orderings; train-median baseline; seed={SHAP_SEED}"
        ),
        "shap_caveat": (
            "Reconstructed implementation; not bit-identical to original SHAP values."
        ),
    }

    baseline = np.median(Xtr, axis=0)
    phi, f1, f0 = mc_interventional_perm_shap(rf, Xte, baseline)
    mean_abs = np.abs(phi).mean(axis=0)
    order = np.argsort(-mean_abs)
    shap_table = pd.DataFrame(
        {
            "rank": np.arange(1, len(FINAL_RF_FEATURES) + 1),
            "feature": [FINAL_RF_FEATURES[i] for i in order],
            "mean_abs_shap": mean_abs[order],
        }
    )
    shap_table.to_csv(OUT_RF_SHAP / "shap_mean_abs_importance.csv", index=False)

    dir_rows = []
    for i, f in enumerate(FINAL_RF_FEATURES):
        dir_rows.append(
            {
                "feature": f,
                "mean_shap_all": float(phi[:, i].mean()),
                "mean_shap_y1": float(phi[y_test == 1, i].mean()),
                "mean_shap_y0": float(phi[y_test == 0, i].mean()),
                "frac_positive": float((phi[:, i] > 0).mean()),
            }
        )
    pd.DataFrame(dir_rows).to_csv(OUT_RF_SHAP / "shap_direction_summary.csv", index=False)

    metrics["efficiency_residual_mean_abs"] = float(np.mean(np.abs(phi.sum(1) - (f1 - f0))))
    (OUT_RF_SHAP / "rf_shap_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(shap_table.to_string(index=False))
    print("Wrote", OUT_RF_SHAP)


if __name__ == "__main__":
    main()
