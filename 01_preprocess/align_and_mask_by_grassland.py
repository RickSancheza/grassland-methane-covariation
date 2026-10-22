"""Align grassland type raster to the emission grid and mask non-grassland cells.

Source lineage: recovered preprocessing scripts (clip / resample), rewritten in
English with relative paths. Math unchanged: nearest resample; keep codes 1-6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from config import (
    EMISSION_DIR,
    GRASSLAND_CODES,
    GRASSLAND_TYPE_TIF,
    OUT_PREPROCESS,
    SECTORS,
    ensure_output_dirs,
    resolve_sector_dir,
    year_tif,
)
from raster_utils import clean_array, read_aligned, same_grid


def main() -> None:
    ensure_output_dirs()
    if not GRASSLAND_TYPE_TIF.exists():
        raise FileNotFoundError(f"Missing grassland type raster: {GRASSLAND_TYPE_TIF}")

    # Use the first sector's first year as the reference emission grid
    first_sector = resolve_sector_dir(EMISSION_DIR, SECTORS[0])
    ref_path = year_tif(first_sector, 2005)

    out_grass = OUT_PREPROCESS / "grasslandtype_aligned.tif"
    report = ["# Preprocess alignment report"]

    with rasterio.open(ref_path) as ref, rasterio.open(GRASSLAND_TYPE_TIF) as gsrc:
        report.append(f"reference={ref_path}")
        report.append(f"grassland_source={GRASSLAND_TYPE_TIF}")
        report.append(f"same_grid={same_grid(gsrc, ref)}")

        # Align grassland types to emission grid (nearest)
        grass = read_aligned(GRASSLAND_TYPE_TIF, ref, Resampling.nearest)
        grass_masked = np.where(np.isin(grass, GRASSLAND_CODES), grass, np.nan)

        profile = ref.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
        with rasterio.open(out_grass, "w", **profile) as dst:
            data = np.where(np.isfinite(grass_masked), grass_masked, -9999.0).astype("float32")
            dst.write(data, 1)

        # Mask each sector/year emission stack
        n_masked = 0
        for sector in SECTORS:
            sector_dir = resolve_sector_dir(EMISSION_DIR, sector)
            out_dir = OUT_PREPROCESS / sector
            out_dir.mkdir(parents=True, exist_ok=True)
            for year in range(2005, 2021):
                src_path = year_tif(sector_dir, year)
                with rasterio.open(src_path) as src:
                    em = read_aligned(src_path, ref, Resampling.nearest)
                    em_out = np.where(np.isfinite(grass_masked) & np.isfinite(em), em, np.nan)
                    out_path = out_dir / f"{year}.tif"
                    prof = src.profile.copy() if same_grid(src, ref) else profile.copy()
                    prof.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
                    with rasterio.open(out_path, "w", **prof) as dst:
                        data = np.where(np.isfinite(em_out), em_out, -9999.0).astype("float32")
                        dst.write(data, 1)
                n_masked += 1

    report.append(f"masked_rasters={n_masked}")
    report.append(f"grassland_codes_kept={list(GRASSLAND_CODES)}")
    report.append(f"output={OUT_PREPROCESS}")
    (OUT_PREPROCESS / "preprocess_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
