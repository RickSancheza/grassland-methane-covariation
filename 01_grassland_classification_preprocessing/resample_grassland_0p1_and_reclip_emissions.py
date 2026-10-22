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


def find_latest_original_backup() -> Path | None:
    backups = sorted(TARGET_DIR.glob("_backup_before_grassland_mask_*"))
    return backups[-1] if backups else None


def backup_current_state(stamp: str, emission_files: list[Path]) -> Path:
    backup_dir = TARGET_DIR / f"_backup_before_0p1_reclip_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for p in emission_files:
        shutil.copy2(p, backup_dir / p.name)
    if GRASSLAND_PATH.exists():
        shutil.copy2(GRASSLAND_PATH, backup_dir / GRASSLAND_PATH.name)
    return backup_dir


def restore_emissions_from_backup(backup_dir: Path, emission_files: list[Path]) -> list[str]:
    restored = []
    for p in emission_files:
        src = backup_dir / p.name
        if src.exists():
            shutil.copy2(src, p)
            restored.append(p.name)
    return restored


def raster_summary(path: Path) -> dict[str, object]:
    with rasterio.open(path) as ds:
        data = ds.read(1, masked=True)
        vals = data.compressed().astype("float64")
        return {
            "file_name": path.name,
            "shape": f"{ds.height}x{ds.width}",
            "crs": str(ds.crs),
            "transform": tuple(round(x, 12) for x in ds.transform),
            "resolution": tuple(round(x, 12) for x in ds.res),
            "nodata": ds.nodata,
            "valid_count": int(vals.size),
            "min": float(np.min(vals)) if vals.size else np.nan,
            "max": float(np.max(vals)) if vals.size else np.nan,
            "mean": float(np.mean(vals)) if vals.size else np.nan,
        }


def aligned(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader) -> bool:
    return (
        a.width == b.width
        and a.height == b.height
        and a.crs == b.crs
        and tuple(round(x, 12) for x in a.transform)
        == tuple(round(x, 12) for x in b.transform)
    )


def write_grassland_0p1(reference_emission: Path, stamp: str) -> Path:
    highres_backup = TARGET_DIR / f"grasslandtype_highres_before_0p1_{stamp}.tif"
    shutil.copy2(GRASSLAND_PATH, highres_backup)

    tmp_path = TARGET_DIR / "grasslandtype_0p1_tmp.tif"
    with rasterio.open(reference_emission) as ref, rasterio.open(GRASSLAND_PATH) as src:
        profile = ref.profile.copy()
        profile.update(
            dtype="int16",
            count=1,
            nodata=-9999,
            compress="lzw",
            predictor=2,
        )
        with WarpedVRT(
            src,
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            nodata=-9999,
        ) as vrt, rasterio.open(tmp_path, "w", **profile) as out:
            for _, window in ref.block_windows(1):
                grass = vrt.read(1, window=window)
                out.write(grass.astype("int16"), 1, window=window)

    os.replace(tmp_path, GRASSLAND_PATH)
    return highres_backup


def reclip_emissions(emission_files: list[Path]) -> list[dict[str, object]]:
    rows = []
    with rasterio.open(GRASSLAND_PATH) as gds:
        for p in emission_files:
            before = raster_summary(p)
            tmp_path = p.with_suffix(".reclip_tmp.tif")
            with rasterio.open(p) as src:
                if not aligned(src, gds):
                    raise RuntimeError(f"{p.name} is not aligned with resampled grasslandtype.tif")
                profile = src.profile.copy()
                nodata = src.nodata if src.nodata is not None else -9999.0
                profile["nodata"] = nodata

                total_pixels = 0
                kept_pixels = 0
                masked_pixels = 0

                with rasterio.open(tmp_path, "w", **profile) as out:
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
                        kept_pixels += int(np.sum(valid & grass_mask))
                        masked_pixels += int(np.sum(valid & ~grass_mask))

            os.replace(tmp_path, p)
            after = raster_summary(p)
            rows.append(
                {
                    "file_name": p.name,
                    "method": "restore original emission if backup exists; resample grasslandtype to emission 0.1-degree grid with nearest; mask non-grassland classes outside 1-6",
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

    emission_files = sorted(
        p
        for p in TARGET_DIR.glob("*.tif")
        if p.name.startswith(EMISSION_PREFIX) and p.name != "grasslandtype.tif"
    )
    if len(emission_files) != 6:
        raise RuntimeError(f"Expected 6 emission tif files, found {len(emission_files)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_backup = backup_current_state(stamp, emission_files)

    original_backup = find_latest_original_backup()
    restored = []
    if original_backup is not None:
        restored = restore_emissions_from_backup(original_backup, emission_files)

    highres_grassland_backup = write_grassland_0p1(emission_files[0], stamp)
    rows = reclip_emissions(emission_files)

    with rasterio.open(GRASSLAND_PATH) as gds, rasterio.open(emission_files[0]) as ref:
        grassland_aligned = aligned(gds, ref)
        grassland_summary = raster_summary(GRASSLAND_PATH)

    audit_csv = TARGET_DIR / f"grassland_0p1_reclip_audit_{stamp}.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "file_name",
            "method",
            "total_pixels",
            "kept_grassland_valid_pixels",
            "masked_non_grassland_valid_pixels",
            "before_valid_count",
            "after_valid_count",
            "before_min",
            "before_max",
            "before_mean",
            "after_min",
            "after_max",
            "after_mean",
            "shape",
            "crs",
            "transform",
            "resolution",
            "nodata",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Target dir: {TARGET_DIR}")
    print(f"Current backup: {current_backup}")
    print(f"Original emission backup used: {original_backup}")
    print(f"Restored emissions: {', '.join(restored) if restored else 'none'}")
    print(f"High-res grassland backup: {highres_grassland_backup}")
    print(f"New grasslandtype: {GRASSLAND_PATH}")
    print(f"Grassland aligned with emissions: {grassland_aligned}")
    print(f"Grassland summary: {grassland_summary}")
    print(f"Audit csv: {audit_csv}")
    for row in rows:
        print(
            f"{row['file_name']}: before_valid={row['before_valid_count']}, "
            f"after_valid={row['after_valid_count']}, masked={row['masked_non_grassland_valid_pixels']}"
        )


if __name__ == "__main__":
    main()
