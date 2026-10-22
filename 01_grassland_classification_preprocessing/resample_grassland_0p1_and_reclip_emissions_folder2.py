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
TARGET_DIR = BASE_DIR / "40" / "1_firstpart" / "不同草地类型分部门排放构成" / "2"
GRASSLAND_PATH = TARGET_DIR / "grasslandtype.tif"
GRASSLAND_CODES = {1, 2, 3, 4, 5, 6}
DEPARTMENTS = {"Agriculture", "Energy", "Enteric", "Industrial", "Waste", "Wetland"}


def emission_files() -> list[Path]:
    files = []
    for p in sorted(TARGET_DIR.glob("*.tif")):
        if p.name == "grasslandtype.tif":
            continue
        stem = p.stem.lower()
        if any(d.lower() in stem for d in DEPARTMENTS):
            files.append(p)
    return files


def aligned(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader) -> bool:
    return (
        a.width == b.width
        and a.height == b.height
        and a.crs == b.crs
        and tuple(round(x, 12) for x in a.transform)
        == tuple(round(x, 12) for x in b.transform)
    )


def raster_summary(path: Path) -> dict[str, object]:
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).compressed().astype("float64")
        return {
            "file_name": path.name,
            "shape": f"{ds.height}x{ds.width}",
            "crs": str(ds.crs),
            "transform": tuple(round(x, 12) for x in ds.transform),
            "resolution": tuple(round(x, 12) for x in ds.res),
            "nodata": ds.nodata,
            "valid_count": int(arr.size),
            "min": float(np.min(arr)) if arr.size else np.nan,
            "max": float(np.max(arr)) if arr.size else np.nan,
            "mean": float(np.mean(arr)) if arr.size else np.nan,
        }


def backup_current(stamp: str, files: list[Path]) -> Path:
    backup_dir = TARGET_DIR / f"_backup_before_0p1_reclip_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for p in files + [GRASSLAND_PATH]:
        shutil.copy2(p, backup_dir / p.name)
    return backup_dir


def resample_grassland_to_reference(reference: Path, stamp: str) -> Path:
    original_backup = TARGET_DIR / f"grasslandtype_before_0p1_{stamp}.tif"
    shutil.copy2(GRASSLAND_PATH, original_backup)
    tmp = TARGET_DIR / "grasslandtype_0p1_tmp.tif"

    with rasterio.open(reference) as ref, rasterio.open(GRASSLAND_PATH) as src:
        profile = ref.profile.copy()
        profile.update(dtype="int16", count=1, nodata=-9999, compress="lzw", predictor=2)
        with WarpedVRT(
            src,
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            nodata=-9999,
        ) as vrt, rasterio.open(tmp, "w", **profile) as out:
            for _, window in ref.block_windows(1):
                grass = vrt.read(1, window=window)
                grass = np.where(np.isin(grass, list(GRASSLAND_CODES)), grass, -9999)
                out.write(grass.astype("int16"), 1, window=window)

    os.replace(tmp, GRASSLAND_PATH)
    return original_backup


def clip(files: list[Path]) -> list[dict[str, object]]:
    rows = []
    with rasterio.open(GRASSLAND_PATH) as gds:
        for p in files:
            before = raster_summary(p)
            tmp = p.with_suffix(".clip_tmp.tif")
            with rasterio.open(p) as src:
                if not aligned(src, gds):
                    raise RuntimeError(f"{p.name} is not aligned with grasslandtype.tif after resampling")
                profile = src.profile.copy()
                nodata = src.nodata if src.nodata is not None else -9999.0
                profile["nodata"] = nodata
                total_pixels = kept = masked = 0
                with rasterio.open(tmp, "w", **profile) as out:
                    for _, window in src.block_windows(1):
                        arr = src.read(1, window=window)
                        grass = gds.read(1, window=window)
                        valid = np.isfinite(arr)
                        if src.nodata is not None:
                            valid &= arr != src.nodata
                        grass_mask = np.isin(grass, list(GRASSLAND_CODES))
                        out_arr = arr.copy()
                        out_arr[valid & ~grass_mask] = nodata
                        out.write(out_arr, 1, window=window)
                        total_pixels += int(arr.size)
                        kept += int(np.sum(valid & grass_mask))
                        masked += int(np.sum(valid & ~grass_mask))
            os.replace(tmp, p)
            after = raster_summary(p)
            rows.append(
                {
                    "file_name": p.name,
                    "method": "resample grasslandtype to emission grid with nearest; keep classes 1-6; set non-grassland emission pixels to nodata",
                    "total_pixels": total_pixels,
                    "kept_grassland_valid_pixels": kept,
                    "masked_non_grassland_valid_pixels": masked,
                    "before_valid_count": before["valid_count"],
                    "after_valid_count": after["valid_count"],
                    "before_min": before["min"],
                    "before_max": before["max"],
                    "before_mean": before["mean"],
                    "after_min": after["min"],
                    "after_max": after["max"],
                    "after_mean": after["mean"],
                    "shape": after["shape"],
                    "crs": after["crs"],
                    "transform": after["transform"],
                    "resolution": after["resolution"],
                    "nodata": after["nodata"],
                }
            )
    return rows


def main() -> None:
    if not TARGET_DIR.exists():
        raise FileNotFoundError(TARGET_DIR)
    if not GRASSLAND_PATH.exists():
        raise FileNotFoundError(GRASSLAND_PATH)
    files = emission_files()
    if len(files) != 6:
        raise RuntimeError(f"Expected 6 emission files, found {len(files)}: {[p.name for p in files]}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_current(stamp, files)
    grass_original = resample_grassland_to_reference(files[0], stamp)
    rows = clip(files)

    with rasterio.open(GRASSLAND_PATH) as gds, rasterio.open(files[0]) as ref:
        is_aligned = aligned(gds, ref)
        grass_summary = raster_summary(GRASSLAND_PATH)

    audit_csv = TARGET_DIR / f"grassland_0p1_reclip_audit_{stamp}.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Target dir: {TARGET_DIR}")
    print(f"Backup dir: {backup_dir}")
    print(f"Original grassland backup: {grass_original}")
    print(f"New grassland aligned with emissions: {is_aligned}")
    print(f"Grassland summary: {grass_summary}")
    print(f"Audit csv: {audit_csv}")
    for row in rows:
        print(
            f"{row['file_name']}: before_valid={row['before_valid_count']}, "
            f"after_valid={row['after_valid_count']}, masked={row['masked_non_grassland_valid_pixels']}"
        )


if __name__ == "__main__":
    main()
