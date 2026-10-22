"""Theil-Sen trend rasters from annual stacks (verified vs published *_trend.tif).

Method: scipy.stats.theilslopes on year index t = 0 .. n-1.
Missing: annual rasters for each driver (GPP, MAT, MAP, SM, ST, etc.).
"""

from __future__ import annotations

import json

import numpy as np
import rasterio
from scipy.stats import theilslopes

from config import (
    ANNUAL_CLIMATE_DIRS,
    ANNUAL_GPP_DIR,
    OUT_FACTORS,
    YEARS,
    ensure_output_dirs,
)


def theilsen_stack(annual_dir, out_name: str) -> dict:
    files = []
    for y in YEARS:
        p = annual_dir / f"{y}.tif"
        if p.exists():
            files.append((y, p))
    if len(files) < 8:
        return {"skipped": out_name, "reason": f"only {len(files)} annual rasters in {annual_dir}"}

    stack = []
    profile = None
    nodata = None
    for _, p in files:
        with rasterio.open(p) as src:
            if profile is None:
                profile = src.profile.copy()
                nodata = src.nodata
            stack.append(src.read(1).astype("float64"))
    S = np.stack(stack, axis=0)
    if nodata is not None:
        S[S == nodata] = np.nan

    t = np.arange(S.shape[0], dtype="float64")
    h, w = S.shape[1:]
    out = np.full((h, w), np.nan, dtype="float32")
    # Same algorithm as published trends; only finite cells are evaluated.
    valid = np.isfinite(S).any(axis=0)
    rows, cols = np.where(valid)
    for i, j in zip(rows, cols):
        v = S[:, i, j]
        ok = np.isfinite(v)
        if int(ok.sum()) < 5:
            continue
        slope, _, _, _ = theilslopes(v[ok], t[ok])
        out[i, j] = slope

    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    out_path = OUT_FACTORS / out_name
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(out), out, -9999.0).astype("float32"), 1)
    return {
        "output": out_name,
        "n_years": int(S.shape[0]),
        "method": "theilslopes on year index",
        "valid_cells": int(np.isfinite(out).sum()),
    }


def main() -> None:
    ensure_output_dirs()
    reports = [theilsen_stack(ANNUAL_GPP_DIR, "GPP_trend.tif")]
    for key, folder in ANNUAL_CLIMATE_DIRS.items():
        reports.append(theilsen_stack(folder, f"{key}_trend.tif"))
    (OUT_FACTORS / "theilsen_trend_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
