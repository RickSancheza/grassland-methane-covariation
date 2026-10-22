"""Soil texture PCA from Sand / Silt / Clay (complete cases, z-score, SVD).

Source lineage: reconstructed (verified against Texture_PCA_*.csv).
PC component signs may flip; explained variance and |loadings| are invariant.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio

from config import OUT_FACTORS, SOIL_DIR, ensure_output_dirs


def main() -> None:
    ensure_output_dirs()
    arrays = {}
    for name in ("Sand", "Silt", "Clay"):
        path = SOIL_DIR / f"{name}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        with rasterio.open(path) as src:
            arrays[name] = src.read(1).astype("float64")
            nd = src.nodata
        if nd is not None:
            arrays[name][arrays[name] == nd] = np.nan

    X = np.stack([arrays["Sand"], arrays["Silt"], arrays["Clay"]], axis=-1)
    mask = np.isfinite(X).all(axis=-1)
    Xc = X[mask]
    mu = Xc.mean(axis=0)
    sd = Xc.std(axis=0, ddof=0)
    Z = (Xc - mu) / sd
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S**2 / (len(Z) - 1)
    ratio = var / var.sum()
    loadings = Vt.T  # columns = PC

    # Write PC1 / PC2 rasters (same grid as Sand)
    for k, name in enumerate(("Texture_PC1", "Texture_PC2")):
        out = np.full(mask.shape, np.nan, dtype="float32")
        # score = Z @ loading
        scores = Z @ loadings[:, k]
        out[mask] = scores.astype("float32")
        path_out = OUT_FACTORS / f"{name}.tif"
        with rasterio.open(SOIL_DIR / "Sand.tif") as src:
            profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
        with rasterio.open(path_out, "w", **profile) as dst:
            dst.write(np.where(np.isfinite(out), out, -9999.0).astype("float32"), 1)

    report = {
        "n_complete_cases": int(mask.sum()),
        "explained_variance_ratio": ratio[:3].tolist(),
        "loadings_rows_Sand_Silt_Clay_cols_PC1_PC2": loadings[:, :2].tolist(),
        "zscore_mean": mu.tolist(),
        "zscore_pop_sd": sd.tolist(),
        "note": "PC signs may be flipped relative to published Texture_PCA_loadings.csv",
    }
    pd.DataFrame(
        {
            "Variable": ["Sand", "Silt", "Clay"],
            "PC1_loading": loadings[:, 0],
            "PC2_loading": loadings[:, 1],
            "Mean": mu,
            "Std": sd,
        }
    ).to_csv(OUT_FACTORS / "texture_pca_loadings.csv", index=False)
    (OUT_FACTORS / "texture_pca_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Wrote", OUT_FACTORS)


if __name__ == "__main__":
    main()
