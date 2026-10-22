# -*- coding: utf-8 -*-
# ============================================================
# 网格尺度计算各排放部门与 ΔGR 的带符号相对协变强度
#
# 当前版本：只输出聚类需要的 6 个 signed_contrib_*.tif
#
# 输入：
# 1. 已有 ΔGR：
#    F:\DOCTOR\graduating\test2\40\0_data\1_Delta_GR
#
# 2. 已有各部门 Δ排放：
#    F:\DOCTOR\graduating\test2\40\0_data\3_change
#
# 输出：
# 只输出 signed_contrib_*.tif：
#    signed_contrib_k = cov(ΔE_k, ΔGR) / sum(|cov(ΔE_k, ΔGR)|) * 100
#
# 不输出：
# cov_*.tif
# contrib_*.tif
# corr_*.tif
# deltaGR_trend.tif
# dominant_sector.tif
# png
# ============================================================

import os
import re
import numpy as np
import rasterio
from rasterio.windows import Window

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# 1. 路径设置
# ============================================================

base_dir = r"F:\DOCTOR\graduating\test2\40\0_data"

# 已有 ΔGR
gr_dir = os.path.join(base_dir, "1_Delta_GR")

# 已有各部门 Δ排放
sector_delta_root = os.path.join(base_dir, "3_change")

# 输出文件夹：只输出 signed_contrib
out_base = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\1_基础数据计算"

signed_contrib_dir = os.path.join(
    out_base,
    "3_signed_contribution_percent"
)

os.makedirs(signed_contrib_dir, exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

years = list(range(2005, 2021))

# 至少多少年有效才计算协方差
MIN_VALID_YEARS = 5

# 输出 NoData
OUT_NODATA = -9999.0

# 分块行数
BLOCK_ROWS = 64

# 原始 Δ排放单位：ton CH4 / cell / yr
# 转换为 Tg CH4 / cell / yr
# 1 Tg = 1e6 ton
EMISSION_TO_TG_DIVISOR = 1e6
CONVERT_EMISSION_TON_TO_TG = True


# ============================================================
# 3. 部门设置
# ============================================================

sector_folder_candidates = {
    "Livestock": ["Enteric", "Livestock", "enteric", "livestock"],
    "Agriculture": ["Agriculture", "agriculture"],
    "Energy": ["Energy", "energy"],
    "Industrial": ["Industrial", "industrial"],
    "Waste": ["Waste", "waste"],
    "Wetland": ["Wetland", "wetland"]
}

sector_cn = {
    "Livestock": "牲畜",
    "Agriculture": "农业",
    "Energy": "能源",
    "Industrial": "工业",
    "Waste": "废弃物",
    "Wetland": "湿地"
}


# ============================================================
# 4. 工具函数
# ============================================================

def find_sector_folder(root, candidates):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"部门根目录不存在：{root}")

    subdirs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and not d.lower().startswith("_backup")
    ]

    subdirs_lower = {d.lower(): d for d in subdirs}

    for cand in candidates:
        if cand.lower() in subdirs_lower:
            return os.path.join(root, subdirs_lower[cand.lower()])

    raise FileNotFoundError(
        f"没有找到部门文件夹。候选名称={candidates}；当前子文件夹={subdirs}"
    )


def extract_year(filename):
    name = os.path.basename(filename)
    m = re.search(r"(19\d{2}|20\d{2})", name)
    if m:
        return int(m.group(1))
    return None


def find_year_file(folder, year):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"文件夹不存在：{folder}")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".tif")
        and not f.lower().endswith(".ovr")
        and "_backup" not in f.lower()
    ]

    matched = []
    for f in files:
        if extract_year(f) == year:
            matched.append(f)

    if len(matched) == 0:
        raise FileNotFoundError(f"缺少 {year} 年 tif 文件：{folder}")

    if len(matched) > 1:
        print(f"[警告] {folder} 中 {year} 年识别到多个文件，将使用第一个：")
        for m in matched:
            print("   ", m)

    return matched[0]


def clean_nodata(arr, nodata=None):
    arr = arr.astype("float32")

    if nodata is not None:
        try:
            if np.isnan(nodata):
                arr[~np.isfinite(arr)] = np.nan
            else:
                arr[arr == nodata] = np.nan
        except TypeError:
            arr[arr == nodata] = np.nan

    arr[arr == OUT_NODATA] = np.nan
    arr[arr == -9999] = np.nan
    arr[arr == -9999.0] = np.nan
    arr[arr == -3.4e38] = np.nan
    arr[arr == 3.4e38] = np.nan
    arr[~np.isfinite(arr)] = np.nan

    return arr


def arr_to_outnodata(arr):
    out = arr.astype("float32").copy()
    out[~np.isfinite(out)] = OUT_NODATA
    return out


def convert_emission_to_Tg(arr):
    if CONVERT_EMISSION_TON_TO_TG:
        return arr / EMISSION_TO_TG_DIVISOR
    return arr


def same_grid_as_template(path, template_info):
    with rasterio.open(path) as src:
        same = (
            src.crs == template_info["crs"]
            and src.transform == template_info["transform"]
            and src.width == template_info["width"]
            and src.height == template_info["height"]
        )

        if not same:
            print("\n[网格不一致]")
            print("文件：", path)
            print("文件 CRS:", src.crs)
            print("模板 CRS:", template_info["crs"])
            print("文件 transform:", src.transform)
            print("模板 transform:", template_info["transform"])
            print("文件 size:", src.width, src.height)
            print("模板 size:", template_info["width"], template_info["height"])

        return same


def check_all_grids(template_file, all_files):
    with rasterio.open(template_file) as tmpl:
        template_info = {
            "crs": tmpl.crs,
            "transform": tmpl.transform,
            "width": tmpl.width,
            "height": tmpl.height
        }

    bad_files = []

    for f in all_files:
        if not same_grid_as_template(f, template_info):
            bad_files.append(f)

    if len(bad_files) > 0:
        raise ValueError(
            "存在未对齐的栅格。当前版本不进行重采样，请先统一 CRS、transform、行列数后再运行。"
        )

    print("所有输入栅格 CRS、transform、行列数完全一致。")


def read_window_direct(path, window):
    with rasterio.open(path) as src:
        arr = src.read(1, window=window).astype("float32")
        arr = clean_nodata(arr, src.nodata)
    return arr


def nan_cov_only(x_stack, y_stack, min_n=5):
    """
    x_stack: Δ排放，Tg CH4 / cell / yr
    y_stack: ΔGR

    返回：
    cov(x, y)
    """
    valid = np.isfinite(x_stack) & np.isfinite(y_stack)
    n = valid.sum(axis=0).astype("float32")

    x_masked = np.where(valid, x_stack, np.nan)
    y_masked = np.where(valid, y_stack, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = np.nanmean(x_masked, axis=0)
        y_mean = np.nanmean(y_masked, axis=0)

        dx = x_masked - x_mean
        dy = y_masked - y_mean

        cov_num = np.nansum(dx * dy, axis=0)
        cov = cov_num / (n - 1)

    cov[n < min_n] = np.nan
    cov[~np.isfinite(cov)] = np.nan

    return cov.astype("float32")


def make_windows(width, height, block_rows=64):
    windows = []
    for row_start in range(0, height, block_rows):
        row_count = min(block_rows, height - row_start)
        windows.append(
            Window(
                col_off=0,
                row_off=row_start,
                width=width,
                height=row_count
            )
        )
    return windows


# ============================================================
# 5. 建立部门路径
# ============================================================

sector_dirs = {}

for sec, candidates in sector_folder_candidates.items():
    sector_dirs[sec] = find_sector_folder(sector_delta_root, candidates)

print("\n识别到的部门路径：")
for sec, folder in sector_dirs.items():
    print(f"{sec}: {folder}")


# ============================================================
# 6. 检查输入文件
# ============================================================

print("\n正在检查输入文件...")

gr_files_by_year = {
    y: find_year_file(gr_dir, y)
    for y in years
}

sector_files_by_year = {}

for sec, folder in sector_dirs.items():
    sector_files_by_year[sec] = {
        y: find_year_file(folder, y)
        for y in years
    }

template_file = gr_files_by_year[years[0]]

all_files = list(gr_files_by_year.values())

for sec in sector_files_by_year:
    all_files.extend(list(sector_files_by_year[sec].values()))

check_all_grids(template_file, all_files)

print("输入文件检查完成。")


# ============================================================
# 7. 建立模板和输出 profile
# ============================================================

with rasterio.open(template_file) as tmpl:
    template_profile = tmpl.profile.copy()
    height = tmpl.height
    width = tmpl.width

template_profile.update(
    dtype="float32",
    count=1,
    nodata=OUT_NODATA,
    compress="lzw"
)

windows = make_windows(width, height, BLOCK_ROWS)

print("\n模板文件：", template_file)
print("模板大小：", height, width)
print("窗口数量：", len(windows))
print("注意：本版本不重采样、不裁剪，保持输入数据原始网格。")
print("注意：只输出 signed_contrib_*.tif。")
print("注意：Δ排放按 ton -> Tg 转换后参与协方差计算。")


# ============================================================
# 8. 只计算并输出 signed_contrib_*.tif
# ============================================================

print("\n开始计算带符号相对协变强度 signed_contrib_*.tif ...")

signed_contrib_paths = {
    sec: os.path.join(signed_contrib_dir, f"signed_contrib_{sec}.tif")
    for sec in sector_dirs.keys()
}

signed_contrib_dsts = {
    sec: rasterio.open(path, "w", **template_profile)
    for sec, path in signed_contrib_paths.items()
}

try:
    for window in tqdm(windows, desc="Signed contribution"):

        h = int(window.height)
        w = int(window.width)

        sec_list = list(sector_dirs.keys())

        # 先读取该窗口的 ΔGR 年序列，只读一次
        gr_stack = []

        for y in years:
            gr = read_window_direct(
                gr_files_by_year[y],
                window=window
            )
            gr_stack.append(gr)

        gr_stack = np.stack(gr_stack, axis=0)

        # 逐部门计算协方差，保存在内存中，不输出 cov
        cov_list = []

        for sec in sec_list:

            e_stack = []

            for y in years:
                em = read_window_direct(
                    sector_files_by_year[sec][y],
                    window=window
                )

                em = convert_emission_to_Tg(em)

                e_stack.append(em)

            e_stack = np.stack(e_stack, axis=0)

            cov = nan_cov_only(
                e_stack,
                gr_stack,
                min_n=MIN_VALID_YEARS
            )

            cov_list.append(cov)

        cov_stack = np.stack(cov_list, axis=0)

        abs_stack = np.abs(cov_stack)
        denom = np.nansum(abs_stack, axis=0)

        # 只要至少一个部门 cov 有效且 denom > 0，就计算
        ok = np.isfinite(denom) & (denom > 0)

        for i, sec in enumerate(sec_list):

            signed_arr = np.full((h, w), np.nan, dtype="float32")

            # 带符号相对协变强度：
            # cov_k / sum(|cov_k|) * 100
            signed_arr[ok] = cov_stack[i][ok] / denom[ok] * 100.0

            signed_contrib_dsts[sec].write(
                arr_to_outnodata(signed_arr),
                1,
                window=window
            )

finally:
    for dst in signed_contrib_dsts.values():
        dst.close()

print("signed_contrib_*.tif 计算完成。")


# ============================================================
# 9. 输出说明
# ============================================================

print("\n全部完成！")
print("只输出以下文件：")

for sec in sector_dirs.keys():
    print(os.path.join(signed_contrib_dir, f"signed_contrib_{sec}.tif"))

print("\n输出文件夹：")
print(signed_contrib_dir)

print("\n方法说明：")
print("signed_contrib_k = cov(ΔE_k, ΔGR) / sum(|cov(ΔE_k, ΔGR)|) * 100")
print("该指标同时包含相对协变强度和协变方向。")
print("后续 6 指标聚类直接读取这 6 个 signed_contrib_*.tif。")