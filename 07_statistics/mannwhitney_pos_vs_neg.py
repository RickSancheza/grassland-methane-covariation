"""Mann-Whitney U + BH-FDR: Enteric-positive vs Enteric-negative cells.

Final analysis frame: cluster 2 vs cluster 4 (union-fill K=6), not legacy 1/3.
Source lineage: reconstructed (verified exact p-values on frozen model table).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from config import (
    FINAL_RF_FEATURES,
    MODEL_INPUT_CSV,
    NEGATIVE_CLUSTER,
    OUT_STATS,
    POSITIVE_CLUSTER,
    ensure_output_dirs,
)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    qr = ranked * m / np.arange(1, m + 1)
    qr = np.minimum.accumulate(qr[::-1])[::-1]
    qr = np.minimum(qr, 1.0)
    qv = np.empty_like(qr)
    qv[order] = qr
    q[ok] = qv
    return q


def load_frame() -> pd.DataFrame:
    if not MODEL_INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing frozen model table: {MODEL_INPUT_CSV}. "
            "Export cluster-2/4 cells with explanatory columns first."
        )
    df = pd.read_csv(MODEL_INPUT_CSV)
    if "response_binary" not in df.columns:
        # derive from cluster column if present
        if "cluster" not in df.columns:
            raise ValueError("Need response_binary or cluster column")
        df["response_binary"] = (
            df["cluster"] == POSITIVE_CLUSTER
        ).astype(int)
    return df


def main() -> None:
    ensure_output_dirs()
    df = load_frame()
    y = df["response_binary"].to_numpy()
    features = [
        c
        for c in df.columns
        if c not in ("lon", "lat", "cluster", "response_type", "response_binary")
    ]

    rows = []
    for v in features:
        a = df.loc[y == 1, v].to_numpy(dtype=float)
        b = df.loc[y == 0, v].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            p = u = np.nan
        else:
            res = mannwhitneyu(a, b, alternative="two-sided")
            u, p = float(res.statistic), float(res.pvalue)
        rows.append(
            {
                "variable": v,
                "n_positive": int(len(a)),
                "n_negative": int(len(b)),
                "median_positive": float(np.median(a)) if len(a) else np.nan,
                "median_negative": float(np.median(b)) if len(b) else np.nan,
                "U_statistic": u,
                "p_value": p,
            }
        )

    rec = pd.DataFrame(rows).sort_values("p_value")
    rec["p_adj_BH"] = bh_fdr(rec["p_value"].to_numpy())
    rec["significant_FDR_0.05"] = rec["p_adj_BH"] < 0.05
    rec.to_csv(OUT_STATS / "mannwhitney_results.csv", index=False)
    rec[rec["significant_FDR_0.05"]].to_csv(
        OUT_STATS / "mannwhitney_significant.csv", index=False
    )

    report = {
        "n_samples": int(len(df)),
        "positive_cluster": POSITIVE_CLUSTER,
        "negative_cluster": NEGATIVE_CLUSTER,
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_significant": int(rec["significant_FDR_0.05"].sum()),
    }
    (OUT_STATS / "mannwhitney_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(rec.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
