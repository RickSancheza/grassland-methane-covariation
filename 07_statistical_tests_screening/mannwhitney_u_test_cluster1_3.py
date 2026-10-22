# -*- coding: utf-8 -*-
# ============================================================
# Mann-Whitney U 秩和检验：
# 牲畜主导正向响应型 vs 牲畜主导反向响应型
#
# 当前版本：
# 1. 解释变量来自：
#    F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\0_解释变量
#
# 2. 解释变量均为 0.1°，聚类图为 1°；
#    因此以 1° 聚类图为模板，将解释变量用 average 聚合到 1°。
#
# 3. 正向型：Cluster 1
#    反向型：Cluster 3
#
# 4. 只输出秩和检验核心结果：
#    - mannwhitney_all_variables_full.csv
#    - mannwhitney_significant_variables.csv
#    - mannwhitney_paper_table.csv
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import mannwhitneyu


# ============================================================
# 1. 路径设置
# ============================================================

# 新解释变量文件夹：全部 0.1°
factor_dir = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\0_解释变量"

# 6SC 聚类结果：1°
cluster_tif = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\2_聚类\response_cluster_1deg_6signedcontrib_min1valid_k6.tif"


# 输出文件夹
out_dir = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\1_秩和检验"

os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

# 根据你当前 k=6 聚类结果：
# Cluster 1 = 牲畜主导正向响应型
# Cluster 3 = 牲畜主导反向响应型
positive_cluster = 1
negative_cluster = 3

positive_label = "Positive_livestock_dominant"
negative_label = "Negative_livestock_dominant"

# FDR 阈值
fdr_threshold = 0.05

# 每组最少样本数
min_group_n = 10

# NoData 值
nodata_values = [-9999, -9999.0, -3.4e38, 3.4e38]


# ============================================================
# 3. 工具函数
# ============================================================

def clean_var_name(filename):
    """
    tif 文件名转变量名。
    保留你现在的新命名：
    R_Agriculture_mean
    I_log1p_Agriculture_mean
    Entericshare_mean
    """
    name = os.path.splitext(os.path.basename(filename))[0]

    remove_suffixes = [
        "_1deg",
        "_0p1",
        "_tif"
    ]

    for s in remove_suffixes:
        name = name.replace(s, "")

    name = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fa5]+", "_", name)

    return name


def apply_nodata(arr, nodata):
    arr = arr.astype("float32")

    if nodata is not None:
        try:
            if np.isnan(nodata):
                arr[~np.isfinite(arr)] = np.nan
            else:
                arr[arr == nodata] = np.nan
        except TypeError:
            arr[arr == nodata] = np.nan

    for nd in nodata_values:
        arr[arr == nd] = np.nan

    arr[~np.isfinite(arr)] = np.nan

    return arr


def read_cluster_template(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        arr = apply_nodata(arr, src.nodata)
        profile = src.profile.copy()

    arr = np.rint(arr).astype("float32")

    return arr, profile


def read_factor_to_cluster_grid(path, template_profile):
    """
    读取解释变量 tif。
    如果变量已经和聚类图同网格，直接读取；
    如果是 0.1°，则用 Resampling.average 聚合到 1° 聚类图网格。
    """
    with rasterio.open(path) as src:

        same_grid = (
            src.height == template_profile["height"] and
            src.width == template_profile["width"] and
            src.transform == template_profile["transform"] and
            src.crs == template_profile["crs"]
        )

        src_arr = src.read(1).astype("float32")
        src_arr = apply_nodata(src_arr, src.nodata)

        if same_grid:
            return src_arr

        dst = np.full(
            (template_profile["height"], template_profile["width"]),
            np.nan,
            dtype="float32"
        )

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.average
        )

        dst = apply_nodata(dst, np.nan)

        return dst.astype("float32")


def bh_fdr(p_values):
    """
    Benjamini-Hochberg FDR 校正。
    """
    p = np.asarray(p_values, dtype="float64")
    q = np.full_like(p, np.nan, dtype="float64")

    valid = np.isfinite(p)
    p_valid = p[valid]

    if len(p_valid) == 0:
        return q

    order = np.argsort(p_valid)
    ranked_p = p_valid[order]
    m = len(ranked_p)

    q_ranked = ranked_p * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked[q_ranked > 1] = 1

    q_valid = np.empty_like(q_ranked)
    q_valid[order] = q_ranked

    q[valid] = q_valid

    return q


def sig_label(p):
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    else:
        return "ns"


def summarize(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan
        }

    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "q25": float(x.quantile(0.25)),
        "q75": float(x.quantile(0.75))
    }


def factor_type(var):
    """
    根据新解释变量命名判断变量类型。
    """
    v = var.lower()

    # 牲畜排放占比
    if "entericshare" in v or "livestockshare" in v:
        if "trend" in v:
            return "排放结构变化"
        return "排放结构"

    # R_：排放结构
    # 例如 R_Agriculture_mean.tif
    if v.startswith("r_"):
        if "trend" in v:
            return "排放结构变化"
        return "排放结构"

    # I_：排放强度
    # 例如 I_log1p_Agriculture_mean.tif
    if v.startswith("i_") or "intensity" in v:
        if "trend" in v:
            return "排放强度变化"
        return "排放强度"

    # 气候
    if (
        v.startswith("mat") or
        v.startswith("map") or
        "temperature" in v or
        "precip" in v
    ):
        return "气候因子"

    # 土壤水热
    if (
        v.startswith("st") or
        v.startswith("sm") or
        "soiltem" in v or
        "soiltemp" in v or
        "soilmoisture" in v or
        "soil_moist" in v
    ):
        return "土壤水热"

    # 植被生产力
    if "gpp" in v or "ndvi" in v:
        return "植被生产力"

    # 土壤质地
    if "texture" in v or "clay" in v or "sand" in v or "silt" in v:
        return "土壤质地"

    # 土壤性质
    if v in ["soc", "bd"] or v.startswith("soc_") or v.startswith("bd_") or "bulk" in v:
        return "土壤性质"

    return "其他"


def extract_sector_from_name(var):
    """
    从变量名中识别部门，用于中文变量名。
    """
    sectors = {
        "agriculture": "农业",
        "energy": "能源",
        "industrial": "工业",
        "waste": "废弃物",
        "wetland": "湿地",
        "livestock": "牲畜",
        "enteric": "牲畜"
    }

    v = var.lower()

    for key, cn in sectors.items():
        if key in v:
            return cn

    return ""


def variable_cn_name(var):
    """
    输出论文表用的中文变量名。
    """
    v = var.lower()
    sec_cn = extract_sector_from_name(var)

    if "entericshare" in v or "livestockshare" in v:
        if "trend" in v:
            return "牲畜排放占比趋势"
        return "牲畜排放占比"

    if v.startswith("r_"):
        if sec_cn:
            if "trend" in v:
                return f"{sec_cn}排放结构趋势"
            return f"{sec_cn}排放结构"
        return var

    if v.startswith("i_") or "intensity" in v:
        if sec_cn:
            if "trend" in v:
                return f"{sec_cn}排放强度趋势"
            return f"{sec_cn}排放强度"
        return var

    if v.startswith("mat"):
        return "MAT趋势" if "trend" in v else "MAT"
    if v.startswith("map"):
        return "MAP趋势" if "trend" in v else "MAP"

    if v.startswith("st") or "soiltem" in v or "soiltemp" in v:
        return "ST趋势" if "trend" in v else "ST"

    if v.startswith("sm") or "soilmoisture" in v or "soil_moist" in v:
        return "SM趋势" if "trend" in v else "SM"

    if "gpp" in v:
        return "GPP趋势" if "trend" in v else "GPP"

    if "texture_pc1" in v:
        return "Texture_PC1"
    if "texture_pc2" in v:
        return "Texture_PC2"

    if v.startswith("soc"):
        return "SOC"
    if v.startswith("bd"):
        return "BD"

    return var


def direction_cn(diff):
    if not np.isfinite(diff):
        return "NA"
    if diff > 0:
        return "正向型较高"
    elif diff < 0:
        return "反向型较高"
    else:
        return "中位数相同"


def fmt_p(p):
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_num(x, digits=3):
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def fmt_median_iqr(median, q25, q75, digits=3):
    if not np.isfinite(median):
        return "NA"
    return f"{median:.{digits}f} ({q25:.{digits}f}–{q75:.{digits}f})"


# ============================================================
# 4. 检查输入
# ============================================================

if not os.path.exists(factor_dir):
    raise FileNotFoundError(f"解释变量文件夹不存在：{factor_dir}")

if not os.path.exists(cluster_tif):
    raise FileNotFoundError(f"聚类图不存在：{cluster_tif}")

tif_files = []

for f in os.listdir(factor_dir):
    if not f.lower().endswith(".tif"):
        continue

    lower = f.lower()

    if "backup" in lower or "tmp" in lower or ".ovr" in lower:
        continue

    tif_files.append(os.path.join(factor_dir, f))

if len(tif_files) == 0:
    raise FileNotFoundError(f"解释变量文件夹中没有 tif 文件：{factor_dir}")

print("找到解释变量 tif 数量：", len(tif_files))
for f in tif_files:
    print(" -", os.path.basename(f))


# ============================================================
# 5. 读取 1° 聚类图作为模板
# ============================================================

cluster_arr, template_profile = read_cluster_template(cluster_tif)

height = template_profile["height"]
width = template_profile["width"]
transform = template_profile["transform"]

base_mask = (
    np.isfinite(cluster_arr) &
    ((cluster_arr == positive_cluster) | (cluster_arr == negative_cluster))
)

n_pos = int((cluster_arr == positive_cluster).sum())
n_neg = int((cluster_arr == negative_cluster).sum())

print("\n聚类图：", cluster_tif)
print("模板大小：", height, width)
print("模板 CRS：", template_profile["crs"])
print("模板 transform：", transform)

print("\n正向型 cluster =", positive_cluster, "像元数：", n_pos)
print("反向型 cluster =", negative_cluster, "像元数：", n_neg)
print("正向/反向总有效像元数：", int(base_mask.sum()))

if int(base_mask.sum()) == 0:
    raise ValueError("没有提取到正向/反向有效像元，请检查 positive_cluster 和 negative_cluster 设置。")


# ============================================================
# 6. 提取正向/反向样本位置
# ============================================================

rows, cols = np.where(base_mask)
xs, ys = rasterio.transform.xy(transform, rows, cols)

sample_y = np.where(
    cluster_arr[rows, cols] == positive_cluster,
    1,
    0
)

print("\n用于检验的样本数：")
print("正向型：", int((sample_y == 1).sum()))
print("反向型：", int((sample_y == 0).sum()))


# ============================================================
# 7. 逐变量读取并进行 Mann-Whitney U 检验
# ============================================================

print("\n开始 Mann-Whitney U 检验...")

records = []
used_names = set()

for tif in tif_files:

    var_name = clean_var_name(tif)

    original_name = var_name
    idx = 1

    while var_name in used_names:
        var_name = f"{original_name}_{idx}"
        idx += 1

    used_names.add(var_name)

    print("检验变量：", var_name)

    arr = read_factor_to_cluster_grid(
        tif,
        template_profile
    )

    values = arr[rows, cols]

    x_pos = pd.Series(values[sample_y == 1])
    x_neg = pd.Series(values[sample_y == 0])

    x_pos = x_pos.replace([np.inf, -np.inf], np.nan).dropna()
    x_neg = x_neg.replace([np.inf, -np.inf], np.nan).dropna()

    s_pos = summarize(x_pos)
    s_neg = summarize(x_neg)

    if s_pos["n"] < min_group_n or s_neg["n"] < min_group_n:
        U = np.nan
        p = np.nan
        rbc = np.nan
        status = "insufficient_n"
    else:
        try:
            res = mannwhitneyu(
                x_pos.values,
                x_neg.values,
                alternative="two-sided",
                method="auto"
            )

            U = float(res.statistic)
            p = float(res.pvalue)

            # rank-biserial correlation:
            # >0 表示正向型整体更高；<0 表示反向型整体更高
            rbc = 2 * U / (s_pos["n"] * s_neg["n"]) - 1

            status = "ok"

        except Exception as e:
            U = np.nan
            p = np.nan
            rbc = np.nan
            status = f"failed: {str(e)[:100]}"

    median_diff = s_pos["median"] - s_neg["median"]
    mean_diff = s_pos["mean"] - s_neg["mean"]

    records.append({
        "factor_type": factor_type(var_name),
        "variable": var_name,
        "variable_cn": variable_cn_name(var_name),
        "source_tif": tif,

        "n_positive": s_pos["n"],
        "n_negative": s_neg["n"],

        "positive_mean": s_pos["mean"],
        "negative_mean": s_neg["mean"],

        "positive_median": s_pos["median"],
        "negative_median": s_neg["median"],

        "positive_q25": s_pos["q25"],
        "positive_q75": s_pos["q75"],
        "negative_q25": s_neg["q25"],
        "negative_q75": s_neg["q75"],

        "median_difference_pos_minus_neg": median_diff,
        "mean_difference_pos_minus_neg": mean_diff,

        "mannwhitney_U": U,
        "p_value": p,
        "rank_biserial_correlation": rbc,
        "direction": direction_cn(median_diff),
        "test_status": status
    })


test_df = pd.DataFrame(records)

if len(test_df) == 0:
    raise ValueError("没有得到任何检验结果，请检查解释变量 tif。")


# ============================================================
# 8. FDR 校正和排序
# ============================================================

test_df["p_adj_BH"] = bh_fdr(test_df["p_value"].values)
test_df["significance_FDR"] = test_df["p_adj_BH"].apply(sig_label)

test_df["is_significant_FDR"] = (
    test_df["p_adj_BH"].notna() &
    (test_df["p_adj_BH"] < fdr_threshold)
)

test_df = test_df.sort_values(
    by=["p_adj_BH", "p_value"],
    na_position="last"
).reset_index(drop=True)


# ============================================================
# 9. 输出完整检验结果
# ============================================================

out_full = os.path.join(out_dir, "mannwhitney_all_variables_full.csv")

test_df.to_csv(
    out_full,
    index=False,
    encoding="utf-8-sig"
)

print("\n完整 Mann-Whitney 检验结果已输出：")
print(out_full)


# ============================================================
# 10. 输出显著变量表
# ============================================================

sig_df = test_df[test_df["is_significant_FDR"]].copy()

out_sig = os.path.join(out_dir, "mannwhitney_significant_variables.csv")

sig_df.to_csv(
    out_sig,
    index=False,
    encoding="utf-8-sig"
)

print("\nFDR 显著变量表已输出：")
print(out_sig)
print("FDR <", fdr_threshold, "显著变量数量：", len(sig_df))


# ============================================================
# 11. 输出论文简表
# ============================================================

paper_df = test_df.copy()

paper_df["正向型中位数(IQR)"] = paper_df.apply(
    lambda r: fmt_median_iqr(
        r["positive_median"],
        r["positive_q25"],
        r["positive_q75"],
        digits=3
    ),
    axis=1
)

paper_df["反向型中位数(IQR)"] = paper_df.apply(
    lambda r: fmt_median_iqr(
        r["negative_median"],
        r["negative_q25"],
        r["negative_q75"],
        digits=3
    ),
    axis=1
)

paper_df["r_rb"] = paper_df["rank_biserial_correlation"].apply(
    lambda x: fmt_num(x, digits=3)
)

paper_df["FDR校正P值"] = paper_df["p_adj_BH"].apply(fmt_p)

paper_df = paper_df[[
    "factor_type",
    "variable",
    "variable_cn",
    "正向型中位数(IQR)",
    "反向型中位数(IQR)",
    "direction",
    "r_rb",
    "FDR校正P值",
    "significance_FDR",
    "is_significant_FDR"
]]

paper_df = paper_df.rename(columns={
    "factor_type": "因子类别",
    "variable": "变量",
    "variable_cn": "变量中文名",
    "direction": "差异方向",
    "significance_FDR": "显著性",
    "is_significant_FDR": "FDR显著"
})

out_paper = os.path.join(out_dir, "mannwhitney_paper_table.csv")

paper_df.to_csv(
    out_paper,
    index=False,
    encoding="utf-8-sig"
)

print("\n论文简表已输出：")
print(out_paper)


# ============================================================
# 12. 打印结果
# ============================================================

print("\n前 20 个检验结果：")
print(test_df[[
    "factor_type",
    "variable",
    "variable_cn",
    "positive_median",
    "negative_median",
    "median_difference_pos_minus_neg",
    "rank_biserial_correlation",
    "p_adj_BH",
    "significance_FDR",
    "direction",
    "is_significant_FDR"
]].head(20))

print("\nFDR 显著变量：")
print(sig_df[["variable", "variable_cn"]].to_string(index=False))


# ============================================================
# 13. 完成
# ============================================================

print("\n全部完成！输出文件夹：")
print(out_dir)

print("\n主要输出：")
print("1. 完整检验结果：", out_full)
print("2. FDR 显著变量表：", out_sig)
print("3. 论文简表：", out_paper)

print("\n注意：")
print("解释变量原始为 0.1°，本代码已使用 Resampling.average 聚合到 1° 聚类网格后再检验。")
print("正向型 = Cluster 1；反向型 = Cluster 3。")