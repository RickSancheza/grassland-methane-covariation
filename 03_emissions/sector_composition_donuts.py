"""Sectoral CH4 composition donuts by grassland eco-type.

Source lineage: recovered plot_donut_groups_for_folder1_folder2.py.
Formula (unchanged): share_k = sum(E_k) / sum(E_total) per grassland type.
Sector nodata inside grassland is treated as 0 in the total.

Writes CSV + PNG/PDF under OUT_EMISSIONS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    EMISSION_DIR,
    GRASSLAND_CODES,
    GRASSLAND_LABELS_EN,
    GRASSLAND_TYPE_TIF,
    OUT_EMISSIONS,
    SECTORS,
    ensure_output_dirs,
    resolve_sector_dir,
    year_tif,
)
from raster_utils import read_aligned

# Plot order and colors (stable figure colors from the paper workflow)
DEPARTMENTS = ["Agriculture", "Wetland", "Livestock", "Energy", "Waste", "Industrial"]
COLORS = {
    "Agriculture": "#E52521",
    "Wetland": "#58C6EF",
    "Livestock": "#FFE08A",
    "Energy": "#FF8953",
    "Waste": "#3F73B5",
    "Industrial": "#B6B6B6",
}


def sector_sum_by_type(mean_year: int = 2015) -> pd.DataFrame:
    """Sum sector emissions per grassland type for one representative year.

    Use a multi-year mean stack if you prefer; the paper donuts used either
    annual folder1 or multi-year means in folder2. Set mean_year as needed.
    """
    sector_dirs = {s: resolve_sector_dir(EMISSION_DIR, s) for s in SECTORS}
    ref_path = year_tif(sector_dirs[SECTORS[0]], mean_year)

    with rasterio.open(ref_path) as ref:
        grass = read_aligned(GRASSLAND_TYPE_TIF, ref)
        stacks = {
            d: read_aligned(year_tif(sector_dirs[d], mean_year), ref)
            for d in SECTORS
        }

    # Map internal sector folder names to DEPARTMENTS labels
    alias = {"Livestock": "Livestock", "Enteric": "Livestock"}
    acc = {
        gt: {d: 0.0 for d in DEPARTMENTS} | {"pixel_count": 0, "total": 0.0}
        for gt in GRASSLAND_CODES
    }

    total = None
    arrays = {}
    for d in SECTORS:
        arr = stacks[d]
        arr = np.where(np.isfinite(arr), arr, 0.0)  # nodata -> 0 inside grassland
        key = alias.get(d, d)
        arrays[key] = arr
        if total is None:
            total = arr.copy()
        else:
            total = total + arr

    for gt in GRASSLAND_CODES:
        mask = (grass == gt) & np.isfinite(total) & (total > 0)
        n = int(mask.sum())
        if n == 0:
            continue
        acc[gt]["pixel_count"] = n
        acc[gt]["total"] = float(total[mask].sum())
        for d in DEPARTMENTS:
            if d in arrays:
                acc[gt][d] = float(arrays[d][mask].sum())

    rows = []
    for gt in GRASSLAND_CODES:
        tot = acc[gt]["total"]
        row = {
            "grassland_type": gt,
            "grassland_name": GRASSLAND_LABELS_EN[gt],
            "pixel_count": acc[gt]["pixel_count"],
            "total_emission": tot,
        }
        for d in DEPARTMENTS:
            row[f"{d.lower()}_sum"] = acc[gt][d]
            row[f"{d.lower()}_share"] = (acc[gt][d] / tot) if tot > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_donuts(df: pd.DataFrame, out_stem: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(6.0, 4.85), dpi=300)
    fig.patch.set_facecolor("#f6f6f6")
    colors = [COLORS[d] for d in DEPARTMENTS]

    def autopct(pct: float) -> str:
        return "" if (not np.isfinite(pct) or pct < 0.7) else f"{pct:.1f}%"

    for ax, gt in zip(axes.flat, GRASSLAND_CODES):
        ax.set_facecolor("#f6f6f6")
        row = df[df["grassland_type"].eq(gt)].iloc[0]
        vals = [max(float(row[f"{d.lower()}_share"]), 0.0) for d in DEPARTMENTS]
        ax.pie(vals, colors=colors, startangle=90, counterclock=False, autopct=autopct, pctdistance=0.75)
        ax.set_title(GRASSLAND_LABELS_EN[gt].replace("_", " "), fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_EMISSIONS / f"{out_stem}.png", dpi=300)
    fig.savefig(OUT_EMISSIONS / f"{out_stem}.pdf")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    df = sector_sum_by_type()
    df.to_csv(OUT_EMISSIONS / "sector_composition_by_grassland.csv", index=False)
    plot_donuts(df, "sector_composition_donuts")
    print(df.to_string(index=False))
    print(f"Wrote outputs to {OUT_EMISSIONS}")


if __name__ == "__main__":
    main()
