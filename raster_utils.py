"""Shared raster helpers (pathlib / rasterio only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from config import NODATA_VALUES


def same_grid(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader) -> bool:
    return (
        a.width == b.width
        and a.height == b.height
        and a.crs == b.crs
        and tuple(round(x, 12) for x in a.transform)
        == tuple(round(x, 12) for x in b.transform)
    )


def clean_array(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    out = arr.astype("float32", copy=True)
    if nodata is not None and np.isfinite(nodata):
        out[out == nodata] = np.nan
    for nd in NODATA_VALUES:
        out[out == nd] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def read_aligned(
    path: Path,
    ref: rasterio.io.DatasetReader,
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    """Read path aligned to ref grid (nearest by default for categorical / counts)."""
    with rasterio.open(path) as src:
        if same_grid(src, ref):
            return clean_array(src.read(1), src.nodata)
        dst = np.full((ref.height, ref.width), np.nan, dtype="float32")
        with WarpedVRT(
            src,
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
            resampling=resampling,
            src_nodata=src.nodata,
            nodata=np.nan,
        ) as vrt:
            dst = clean_array(vrt.read(1), np.nan)
        return dst
