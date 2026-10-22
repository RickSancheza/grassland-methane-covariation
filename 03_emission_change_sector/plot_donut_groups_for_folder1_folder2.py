from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import matplotlib.pyplot as plt


BASE_DIR = Path(r"F:\DOCTOR\graduating\test2")
ROOT = BASE_DIR / "40" / "1_firstpart" / "不同草地类型分部门排放构成"
FOLDERS = ["1", "2"]

DEPARTMENTS = ["Agriculture", "Wetland", "Enteric", "Energy", "Waste", "Industrial"]
TOTAL_DEPARTMENTS = ["Agriculture", "Enteric", "Energy", "Industrial", "Waste", "Wetland"]
CN = {
    "Agriculture": "农业",
    "Wetland": "湿地",
    "Enteric": "肠道",
    "Energy": "能源",
    "Waste": "废弃物",
    "Industrial": "工业",
}
COLORS = {
    "Agriculture": "#E52521",
    "Wetland": "#58C6EF",
    "Enteric": "#FFE08A",
    "Energy": "#FF8953",
    "Waste": "#3F73B5",
    "Industrial": "#B6B6B6",
}
GRASSLAND_NAMES = {
    1: "热带亚热带\n草地",
    2: "温带\n草地",
    3: "洪泛湿生\n草地",
    4: "山地\n草地",
    5: "冻原",
    6: "干旱草地\n荒漠草地",
}


def find_folder(name: str) -> Path:
    matches = [d for d in ROOT.rglob("*") if d.is_dir() and d.name == name and (d / "grasslandtype.tif").exists()]
    if not matches:
        raise FileNotFoundError(f"Cannot find folder {name} under {ROOT}")
    return matches[0]


def find_department_files(folder: Path) -> dict[str, Path]:
    files = [p for p in folder.glob("*.tif") if p.name.lower() != "grasslandtype.tif"]
    files = [
        p
        for p in files
        if not p.name.lower().startswith("grasslandtype_before")
        and not p.name.lower().startswith("grasslandtype_highres")
    ]
    out: dict[str, Path] = {}
    for dept in TOTAL_DEPARTMENTS:
        matches = [p for p in files if dept.lower() in p.name.lower()]
        if not matches:
            # Folder 1 uses short lower-case names, still handled by dept.lower().
            raise FileNotFoundError(f"{folder}: missing {dept}")
        matches.sort(key=lambda p: (len(p.name), p.name.lower()))
        out[dept] = matches[0]
    return out


def same_grid(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader) -> bool:
    return (
        a.width == b.width
        and a.height == b.height
        and a.crs == b.crs
        and tuple(round(x, 12) for x in a.transform)
        == tuple(round(x, 12) for x in b.transform)
    )


def calculate(folder: Path, files: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    stats = {
        gt: {"pixel_count": 0, "total": 0.0, "sum": {d: 0.0 for d in TOTAL_DEPARTMENTS}}
        for gt in range(1, 7)
    }
    nodata_or_invalid = 0
    zero_total = 0

    ref_path = files["Agriculture"]
    grass_path = folder / "grasslandtype.tif"

    with rasterio.open(ref_path) as ref, rasterio.open(grass_path) as gsrc:
        grass_aligned = same_grid(ref, gsrc)
        with WarpedVRT(
            gsrc,
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
            resampling=Resampling.nearest,
            src_nodata=gsrc.nodata,
            nodata=-9999,
        ) as gvrt:
            srcs = {d: rasterio.open(p) for d, p in files.items()}
            try:
                for _, window in ref.block_windows(1):
                    grass = gvrt.read(1, window=window)
                    grass_valid = np.isin(grass, [1, 2, 3, 4, 5, 6])
                    arrays = {}
                    for d, src in srcs.items():
                        if same_grid(ref, src):
                            data = src.read(1, window=window).astype("float64")
                        else:
                            with WarpedVRT(
                                src,
                                crs=ref.crs,
                                transform=ref.transform,
                                width=ref.width,
                                height=ref.height,
                                resampling=Resampling.nearest,
                                src_nodata=src.nodata,
                                nodata=np.nan,
                            ) as vrt:
                                data = vrt.read(1, window=window).astype("float64")
                        v = np.isfinite(data)
                        if src.nodata is not None and np.isfinite(src.nodata):
                            v &= data != src.nodata
                        # Folder 1 uses float min as nodata.
                        v &= data > -1.0e20
                        arrays[d] = np.where(v, data, 0.0)

                    total = sum(arrays[d] for d in TOTAL_DEPARTMENTS)
                    valid = grass_valid & np.isfinite(total)
                    zero_total += int((valid & (total == 0)).sum())
                    valid &= total > 0

                    for gt in range(1, 7):
                        mask = valid & (grass == gt)
                        n = int(mask.sum())
                        if n == 0:
                            continue
                        stats[gt]["pixel_count"] += n
                        stats[gt]["total"] += float(np.sum(total[mask], dtype=np.float64))
                        for d in TOTAL_DEPARTMENTS:
                            stats[gt]["sum"][d] += float(np.sum(arrays[d][mask], dtype=np.float64))
            finally:
                for src in srcs.values():
                    src.close()

    rows = []
    for gt in range(1, 7):
        total = stats[gt]["total"]
        row = {
            "grassland_type": gt,
            "grassland_name": GRASSLAND_NAMES[gt].replace("\n", ""),
            "pixel_count": stats[gt]["pixel_count"],
            "total_emission": total,
        }
        for d in TOTAL_DEPARTMENTS:
            row[f"{d.lower()}_sum"] = stats[gt]["sum"][d]
            row[f"{d.lower()}_share"] = stats[gt]["sum"][d] / total if total > 0 else np.nan
        rows.append(row)

    diag = {
        "folder": str(folder),
        "reference_grid": str(ref_path),
        "grassland_aligned_to_reference": grass_aligned,
        "nodata_or_invalid_grass_pixels": nodata_or_invalid,
        "zero_total_pixels": zero_total,
    }
    return pd.DataFrame(rows), diag


def autopct(pct: float) -> str:
    if not np.isfinite(pct) or pct <= 0 or pct < 0.7:
        return ""
    return f"{pct:.1f}%"


def plot(df: pd.DataFrame, title: str, png_path: Path, pdf_path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(2, 3, figsize=(6.0, 4.85), dpi=300)
    bg = "#f6f6f6"
    fig.patch.set_facecolor(bg)
    colors = [COLORS[d] for d in DEPARTMENTS]

    for ax, gt in zip(axes.flat, [1, 2, 3, 4, 5, 6]):
        ax.set_facecolor(bg)
        row = df[df["grassland_type"].eq(gt)].iloc[0]
        vals = [max(float(row[f"{d.lower()}_share"]), 0.0) for d in DEPARTMENTS]
        ax.pie(
            vals,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.36, "edgecolor": bg, "linewidth": 1.4},
            autopct=autopct,
            pctdistance=0.84,
            textprops={"fontsize": 7, "color": "white", "family": "SimSun"},
        )
        ax.text(
            0,
            0,
            GRASSLAND_NAMES[gt],
            ha="center",
            va="center",
            fontsize=9,
            color="#4f4f4f",
            linespacing=1.2,
            family="SimSun",
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[d]) for d in DEPARTMENTS]
    labels = [CN[d] for d in DEPARTMENTS]
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        columnspacing=1.6,
        prop={"family": "SimSun", "size": 10},
    )
    fig.suptitle(title, fontsize=20, y=0.965, color="#333333", family="SimSun")
    plt.subplots_adjust(left=0.035, right=0.965, top=0.89, bottom=0.18, wspace=0.14, hspace=0.33)
    fig.savefig(png_path, dpi=300, facecolor=bg)
    fig.savefig(pdf_path, facecolor=bg)
    plt.close(fig)


def write_note(path: Path, diags: list[dict[str, object]]) -> None:
    lines = ["两组环形图说明", "=" * 28, ""]
    lines.append("统计口径：先按 grasslandtype.tif 的草地类型编码 1-6 掩膜，再按每个草地类型汇总 6 个部门总排放。")
    lines.append("nodata 处理：草地范围内某一部门为 nodata 时，该部门按 0 参与该像元总排放；不再使用六部门有效像元交集。")
    lines.append("占比公式：share_k = sum(E_k) / sum(E_total)，E_total = Agriculture + Enteric + Energy + Industrial + Waste + Wetland。")
    lines.append("部门顺序与颜色：农业红色、湿地浅蓝、肠道浅黄、能源橙色、废弃物深蓝、工业灰色。")
    lines.append("")
    for diag in diags:
        lines.append(f"目录：{diag['folder']}")
        lines.append(f"参考网格：{diag['reference_grid']}")
        lines.append(f"grasslandtype 是否原本与参考网格一致：{diag['grassland_aligned_to_reference']}")
        lines.append(f"无效/nodata 草地像元计数：{diag['nodata_or_invalid_grass_pixels']}")
        lines.append(f"E_total=0 像元计数：{diag['zero_total_pixels']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    diags = []
    outputs = []
    for folder_name in FOLDERS:
        folder = find_folder(folder_name)
        files = find_department_files(folder)
        df, diag = calculate(folder, files)
        csv_path = folder / f"{folder_name}_分部门排放构成统计表.csv"
        png_path = folder / f"{folder_name}_草地类型分部门排放构成_环形图.png"
        pdf_path = folder / f"{folder_name}_草地类型分部门排放构成_环形图.pdf"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        plot(df, f"第{folder_name}组", png_path, pdf_path)
        diags.append(diag)
        outputs.extend([csv_path, png_path, pdf_path])
        print(f"Folder {folder_name}: {folder}")
        print(df[["grassland_type", "grassland_name", "enteric_share", "energy_share", "waste_share", "pixel_count", "total_emission"]].to_string(index=False))

    note_path = ROOT / "两组环形图说明.txt"
    write_note(note_path, diags)
    script_copy = ROOT / "plot_donut_groups_for_folder1_folder2.py"
    if Path(__file__).resolve() != script_copy.resolve():
        script_copy.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8-sig")
    outputs.extend([note_path, script_copy])

    print("\nOutputs:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
