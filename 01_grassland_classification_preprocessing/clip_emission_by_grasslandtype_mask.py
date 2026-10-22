from __future__ import annotations

import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


BASE_DIR = Path(r"F:\DOCTOR\graduating\test2")
TARGET_DIR = BASE_DIR / "40" / "1_firstpart" / "不同草地类型分部门排放构成" / "1"
GRASSLAND_PATH = TARGET_DIR / "grasslandtype.tif"
EMISSION_PREFIX = "Emission_"
GRASSLAND_CODES = {1, 2, 3, 4, 5, 6}


def is_same_grid(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader) -> bool:
    return (
        a.width == b.width
        and a.height == b.height
        and a.crs == b.crs
        and tuple(round(x, 12) for x in a.transform)
        == tuple(round(x, 12) for x in b.transform)
    )


def raster_summary(path: Path) -> dict[str, object]:
    with rasterio.open(path) as ds:
        data = ds.read(1, masked=True)
        arr = np.asarray(data.compressed(), dtype=np.float64)
        return {
            "file": str(path),
            "shape": f"{ds.height}x{ds.width}",
            "crs": str(ds.crs),
            "transform": tuple(round(x, 12) for x in ds.transform),
            "resolution": ds.res,
            "nodata": ds.nodata,
            "valid_count": int(arr.size),
            "min": float(np.min(arr)) if arr.size else np.nan,
            "max": float(np.max(arr)) if arr.size else np.nan,
            "mean": float(np.mean(arr)) if arr.size else np.nan,
        }


def main() -> None:
    if not TARGET_DIR.exists():
        raise FileNotFoundError(TARGET_DIR)
    if not GRASSLAND_PATH.exists():
        raise FileNotFoundError(GRASSLAND_PATH)

    emission_files = sorted(
        p
        for p in TARGET_DIR.glob("*.tif")
        if p.name.startswith(EMISSION_PREFIX) and p.name != "grasslandtype.tif"
    )
    if len(emission_files) != 6:
        raise RuntimeError(f"Expected 6 emission tif files, found {len(emission_files)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = TARGET_DIR / f"_backup_before_grassland_mask_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    audit_rows: list[dict[str, object]] = []

    with rasterio.open(GRASSLAND_PATH) as gds:
        grass_meta = {
            "shape": (gds.height, gds.width),
            "crs": str(gds.crs),
            "transform": tuple(round(x, 12) for x in gds.transform),
            "res": gds.res,
            "nodata": gds.nodata,
        }

        for src_path in emission_files:
            backup_path = backup_dir / src_path.name
            shutil.copy2(src_path, backup_path)

            before = raster_summary(src_path)

            tmp_path = src_path.with_suffix(".masked_tmp.tif")
            with rasterio.open(src_path) as eds:
                same_grid = is_same_grid(eds, gds)
                profile = eds.profile.copy()
                nodata = eds.nodata
                if nodata is None:
                    nodata = -9999.0
                    profile["nodata"] = nodata

                with WarpedVRT(
                    gds,
                    crs=eds.crs,
                    transform=eds.transform,
                    width=eds.width,
                    height=eds.height,
                    resampling=Resampling.nearest,
                    src_nodata=gds.nodata,
                    nodata=gds.nodata,
                ) as grass_vrt, rasterio.open(tmp_path, "w", **profile) as out:
                    total_pixels = 0
                    kept_pixels = 0
                    masked_pixels = 0
                    for _, window in eds.block_windows(1):
                        arr = eds.read(1, window=window)
                        grass = grass_vrt.read(1, window=window)

                        valid_emission = np.ones(arr.shape, dtype=bool)
                        if eds.nodata is not None:
                            valid_emission &= arr != eds.nodata
                        valid_emission &= np.isfinite(arr)

                        grass_mask = np.isin(grass, list(GRASSLAND_CODES))
                        out_arr = arr.copy()
                        out_arr[valid_emission & ~grass_mask] = nodata
                        out.write(out_arr, 1, window=window)

                        total_pixels += int(arr.size)
                        kept_pixels += int(np.sum(valid_emission & grass_mask))
                        masked_pixels += int(np.sum(valid_emission & ~grass_mask))

            os.replace(tmp_path, src_path)
            after = raster_summary(src_path)
            audit_rows.append(
                {
                    "file_name": src_path.name,
                    "backup_path": str(backup_path),
                    "same_grid_as_grasslandtype": same_grid,
                    "method": "grasslandtype nearest-resampled to emission raster grid; non-grassland classes outside 1-6 set to emission nodata",
                    "total_pixels": total_pixels,
                    "kept_grassland_valid_pixels": kept_pixels,
                    "masked_non_grassland_valid_pixels": masked_pixels,
                    "before_valid_count": before["valid_count"],
                    "after_valid_count": after["valid_count"],
                    "before_min": before["min"],
                    "before_max": before["max"],
                    "before_mean": before["mean"],
                    "after_min": after["min"],
                    "after_max": after["max"],
                    "after_mean": after["mean"],
                    "emission_shape": before["shape"],
                    "emission_crs": before["crs"],
                    "emission_transform": before["transform"],
                    "emission_resolution": before["resolution"],
                    "emission_nodata": before["nodata"],
                    "grassland_shape": f"{grass_meta['shape'][0]}x{grass_meta['shape'][1]}",
                    "grassland_crs": grass_meta["crs"],
                    "grassland_transform": grass_meta["transform"],
                    "grassland_resolution": grass_meta["res"],
                    "grassland_nodata": grass_meta["nodata"],
                }
            )

    audit_csv = TARGET_DIR / f"grassland_mask_replace_audit_{stamp}.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Target dir: {TARGET_DIR}")
    print(f"Grasslandtype: {GRASSLAND_PATH}")
    print(f"Backup dir: {backup_dir}")
    print(f"Audit csv: {audit_csv}")
    for row in audit_rows:
        print(
            f"{row['file_name']}: aligned={row['same_grid_as_grasslandtype']}, "
            f"before_valid={row['before_valid_count']}, after_valid={row['after_valid_count']}, "
            f"masked={row['masked_non_grassland_valid_pixels']}"
        )


if __name__ == "__main__":
    main()
