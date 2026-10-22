"""Grid-level signed relative covariance contribution (SC) against Delta GR.

Formula (verified against published signed_contrib_*.tif):
    SC_k = cov(dE_k, dGR) / sum_j |cov(dE_j, dGR)| * 100

Source lineage: recovered contribution-grid script (signed_contribution_grid.py).
Requires pre-aligned yearly Delta GR and Delta emission rasters.
"""

from __future__ import annotations

import re

import numpy as np
import rasterio
from rasterio.windows import Window

from config import (
    DELTA_EMISSION_DIR,
    DELTA_GR_DIR,
    EMISSION_TO_TG_DIVISOR,
    MIN_VALID_YEARS,
    OUT_COVARIANCE,
    SC_NODATA,
    SECTORS,
    YEARS,
    ensure_output_dirs,
    resolve_sector_dir,
    year_tif,
)
from raster_utils import clean_array

BLOCK_ROWS = 64


def nan_cov(x_stack: np.ndarray, y_stack: np.ndarray, min_n: int = MIN_VALID_YEARS) -> np.ndarray:
    valid = np.isfinite(x_stack) & np.isfinite(y_stack)
    n = valid.sum(axis=0).astype("float32")
    x_m = np.where(valid, x_stack, np.nan)
    y_m = np.where(valid, y_stack, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = x_m - np.nanmean(x_m, axis=0)
        dy = y_m - np.nanmean(y_m, axis=0)
        cov = np.nansum(dx * dy, axis=0) / (n - 1)
    cov = np.where(n >= min_n, cov, np.nan)
    cov[~np.isfinite(cov)] = np.nan
    return cov.astype("float32")


def main() -> None:
    ensure_output_dirs()
    out_dir = OUT_COVARIANCE / "signed_contribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    sector_dirs = {s: resolve_sector_dir(DELTA_EMISSION_DIR, s) for s in SECTORS}
    gr_by_year = {y: year_tif(DELTA_GR_DIR, y) for y in YEARS}
    em_by_year = {
        s: {y: year_tif(sector_dirs[s], y) for y in YEARS} for s in SECTORS
    }

    with rasterio.open(gr_by_year[YEARS[0]]) as tmpl:
        profile = tmpl.profile.copy()
        height, width = tmpl.height, tmpl.width
        profile.update(dtype="float32", count=1, nodata=SC_NODATA, compress="lzw")

        # Require identical grids (no silent resample — matches recovered script)
        for path in list(gr_by_year.values()) + [p for d in em_by_year.values() for p in d.values()]:
            with rasterio.open(path) as src:
                if (src.crs, src.transform, src.width, src.height) != (
                    tmpl.crs,
                    tmpl.transform,
                    tmpl.width,
                    tmpl.height,
                ):
                    raise ValueError(f"Grid mismatch for {path}; align inputs first.")

        writers = {
            s: rasterio.open(out_dir / f"signed_contrib_{s}.tif", "w", **profile)
            for s in SECTORS
        }
        try:
            for row0 in range(0, height, BLOCK_ROWS):
                h = min(BLOCK_ROWS, height - row0)
                window = Window(0, row0, width, h)
                gr_stack = []
                for y in YEARS:
                    with rasterio.open(gr_by_year[y]) as src:
                        gr_stack.append(clean_array(src.read(1, window=window), src.nodata))
                gr_stack = np.stack(gr_stack, axis=0)

                cov_list = []
                for s in SECTORS:
                    e_stack = []
                    for y in YEARS:
                        with rasterio.open(em_by_year[s][y]) as src:
                            arr = clean_array(src.read(1, window=window), src.nodata)
                        e_stack.append(arr / EMISSION_TO_TG_DIVISOR)
                    cov_list.append(nan_cov(np.stack(e_stack, axis=0), gr_stack))

                cov_stack = np.stack(cov_list, axis=0)
                denom = np.nansum(np.abs(cov_stack), axis=0)
                ok = np.isfinite(denom) & (denom > 0)
                for i, s in enumerate(SECTORS):
                    out = np.full((h, width), np.nan, dtype="float32")
                    out[ok] = cov_stack[i][ok] / denom[ok] * 100.0
                    out = np.where(np.isfinite(out), out, SC_NODATA).astype("float32")
                    writers[s].write(out, 1, window=window)
        finally:
            for w in writers.values():
                w.close()

    print(f"SC rasters written to {out_dir}")
    print("SC_k = cov(dE_k, dGR) / sum |cov(dE_j, dGR)| * 100")


if __name__ == "__main__":
    main()
