"""K-means response-type clustering on 1-degree signed contribution (SC) grid.

Method (verified against published K6_cluster_results.xlsx / DBI table):
  - Features: SC_Livestock, SC_Agriculture, SC_Energy, SC_Industrial, SC_Waste, SC_Wetland
  - Retain cell if >=1 SC is finite
  - Per-feature z-score (population SD); missing -> 0 in standardized space
  - sklearn KMeans k-means++ / Lloyd, random_state=42, n_init=50, max_iter=500
  - DBI computed in standardized space
  - Cluster SC means: observed-only

If 1-degree SC rasters are not present, this script accepts pre-aggregated
SC rasters in DATA_ROOT/signed_contribution_1deg/ (six GeoTIFFs on one grid).

Source lineage: reconstructed implementation (verified exact DBI/labels).
Not recovered original source.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

from config import (
    KMEANS_K,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    OUT_CLUSTERING,
    POSITIVE_CLUSTER,
    NEGATIVE_CLUSTER,
    SC_GRID_DIR,
    SECTORS,
    ensure_output_dirs,
)

SC_COLS = [f"SC_{s}" for s in SECTORS]


def load_sc_stack(folder):
    """Return (profiles, matrix [n_valid, n_sectors], mask, rowcol)."""
    arrs = []
    profile = None
    for s in SECTORS:
        path = folder / f"signed_contrib_{s}.tif"
        if not path.exists():
            path = folder / f"agg1deg_SC_{s}.tif"
        with rasterio.open(path) as src:
            arrs.append(src.read(1).astype("float64"))
            if profile is None:
                profile = src.profile.copy()
            nd = src.nodata
        if nd is not None:
            arrs[-1][arrs[-1] == nd] = np.nan
    stack = np.stack(arrs, axis=-1)  # H, W, S
    valid = np.isfinite(stack).any(axis=-1)
    rows, cols = np.where(valid)
    X = stack[valid]
    return profile, X, valid, rows, cols


def standardize_fill_zero(X_raw: np.ndarray):
    mu = np.nanmean(X_raw, axis=0)
    sd = np.nanstd(X_raw, axis=0, ddof=0)
    Xs = (X_raw - mu) / sd
    Xs = np.where(np.isfinite(Xs), Xs, 0.0)
    return Xs, mu, sd


def main() -> None:
    ensure_output_dirs()
    sc_dir = SC_GRID_DIR
    alt = SC_GRID_DIR.parent / "signed_contribution_1deg"
    if not sc_dir.exists() and alt.exists():
        sc_dir = alt
    if not sc_dir.exists():
        raise FileNotFoundError(
            "No SC grid directory. Place signed_contrib_<Sector>.tif under "
            f"{SC_GRID_DIR} or 1-degree aggregates in {alt}."
        )

    profile, X_raw, valid, rows, cols = load_sc_stack(sc_dir)
    Xs, mu, sd = standardize_fill_zero(X_raw)

    km = KMeans(
        n_clusters=KMEANS_K,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        algorithm="lloyd",
    )
    labels0 = km.fit_predict(Xs)
    dbi = float(davies_bouldin_score(Xs, labels0))

    # observed-only cluster means
    means = []
    for cl in range(KMEANS_K):
        m = labels0 == cl
        rec = {"cluster_raw": int(cl), "n_cells": int(m.sum())}
        for j, col in enumerate(SC_COLS):
            rec[col] = float(np.nanmean(X_raw[m, j])) if m.any() else np.nan
        # broad class: dominant |SC| among livestock first
        sc_vals = {col: rec[col] for col in SC_COLS}
        rec["dominant_source"] = max(
            (k for k, v in sc_vals.items() if np.isfinite(v)),
            key=lambda k: abs(sc_vals[k]),
            default="NA",
        )
        rec["livestock_sign"] = (
            "positive" if sc_vals.get("SC_Livestock", np.nan) > 0 else "negative"
        )
        means.append(rec)

    centers = pd.DataFrame(means)
    centers.to_csv(OUT_CLUSTERING / "kmeans_cluster_centers_observed_means.csv", index=False)

    # Write cluster raster
    out_profile = profile.copy()
    out_profile.update(dtype="int16", count=1, nodata=-9999, compress="lzw")
    grid = np.full(valid.shape, -9999, dtype="int16")
    grid[valid] = labels0.astype("int16") + 1  # 1..K
    with rasterio.open(OUT_CLUSTERING / "kmeans_k6_clusters.tif", "w", **out_profile) as dst:
        dst.write(grid, 1)

    report = {
        "status": "RECONSTRUCTED_verified_method",
        "n_cells": int(valid.sum()),
        "kmeans": {
            "n_clusters": KMEANS_K,
            "random_state": KMEANS_RANDOM_STATE,
            "n_init": KMEANS_N_INIT,
            "max_iter": KMEANS_MAX_ITER,
            "algorithm": "lloyd",
        },
        "dbi": dbi,
        "feature_mean": mu.tolist(),
        "feature_scale_pop_sd": sd.tolist(),
        "positive_cluster_id_note": (
            f"Paper union-fill K=6 uses cluster {POSITIVE_CLUSTER}=Enteric-positive and "
            f"{NEGATIVE_CLUSTER}=Enteric-negative after label orientation by centers. "
            "Raw KMeans IDs may need remapping via centers table."
        ),
        "centers": centers.to_dict(orient="records"),
    }
    (OUT_CLUSTERING / "kmeans_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(centers.to_string(index=False))
    print("DBI", dbi)
    print("Wrote", OUT_CLUSTERING)


if __name__ == "__main__":
    main()
