# -*- coding: utf-8 -*-
# ============================================================
# Mann-Whitney 显著变量
# -> 从 0.1° 解释变量 tif 重新提取显著变量
# -> 聚合到 1° 聚类网格
# -> Spearman 相关性网络
# -> 迭代删除高相关网络中的高连接变量
# -> 输出最终保留变量、删除日志和相关性热图
#
# 当前版本适配：
# 1. 解释变量路径：
#    F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\0_解释变量
#
# 2. Mann-Whitney 结果路径：
#    F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\1_mannwhitney_6SC_pos_vs_neg
#
# 3. 不再读取：
#    raw_factor_table_positive_vs_negative_from_tif.csv
#    significant_variables_for_correlation.csv
#
# 4. 直接读取：
#    mannwhitney_all_variables_full.csv
#    mannwhitney_significant_variables.csv
#
# 5. 解释变量原始为 0.1°，本代码用 Resampling.average 聚合到 1°。
# ============================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm


# ============================================================
# 1. 路径设置
# ============================================================

# 新解释变量文件夹：全部 0.1°
factor_dir = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\0_解释变量"

# 6SC 聚类结果：1°
cluster_tif = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\2_聚类\response_cluster_1deg_6signedcontrib_min1valid_k6.tif"

# 新 Mann-Whitney 输出文件夹
mw_dir = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\1_秩和检验"

mannwhitney_full_file = os.path.join(
    mw_dir,
    "mannwhitney_all_variables_full.csv"
)

mannwhitney_sig_file = os.path.join(
    mw_dir,
    "mannwhitney_significant_variables.csv"
)

# 输出文件夹
out_dir = os.path.join(
    mw_dir,
    "2_spearman_corr_filter"
)

os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

# Cluster 1 = 牲畜主导正向响应型
# Cluster 3 = 牲畜主导反向响应型
positive_cluster = 1
negative_cluster = 3

# 高相关阈值
corr_threshold = 0.70

# Spearman 相关最小成对样本数
min_pairwise_n = 50

# 图片分辨率
figure_dpi = 300

# NoData
nodata_values = [-9999, -9999.0, -3.4e38, 3.4e38]

# 如果多个变量条件相同，是否优先删除 PCA / PC 类综合变量
prefer_drop_pc_variables_when_tie = True


# ============================================================
# 3. 可选：强制保留 / 强制删除
# ============================================================

force_keep_vars = set([
    # "Entericshare_mean"
])

force_drop_vars = set([
    # "Texture_PC2"
])


# ============================================================
# 4. 字体设置
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 5. 工具函数
# ============================================================

def clean_var_name(filename):
    """
    tif 文件名转变量名。
    与 Mann-Whitney 脚本保持一致。
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
    如果和聚类图同网格，直接读取；
    如果是 0.1°，则用 average 聚合到 1° 聚类图网格。
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
            return src_arr.astype("float32")

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


def build_tif_lookup(factor_dir):
    """
    根据 clean_var_name 建立 variable -> tif 路径的映射。
    """
    lookup = {}

    for f in os.listdir(factor_dir):
        if not f.lower().endswith(".tif"):
            continue

        lower = f.lower()

        if "backup" in lower or "tmp" in lower or ".ovr" in lower:
            continue

        tif = os.path.join(factor_dir, f)
        var_name = clean_var_name(tif)

        if var_name in lookup:
            print(f"[警告] 变量名重复：{var_name}")
            print("  已有：", lookup[var_name])
            print("  新的：", tif)
            print("  将保留第一个。")
            continue

        lookup[var_name] = tif

    return lookup


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def infer_factor_type(var):
    v = var.lower()

    # 牲畜排放占比
    if "entericshare" in v or "livestockshare" in v:
        if "trend" in v:
            return "排放结构变化"
        return "排放结构"

    # R_：排放结构
    if v.startswith("r_"):
        if "trend" in v:
            return "排放结构变化"
        return "排放结构"

    # I_：排放强度
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


def is_pc_variable(var):
    v = var.lower()
    return ("texture_pc" in v) or re.search(r"(^|_)pc\d+($|_)", v) is not None


def sector_short_name(var):
    v = var.lower()

    if "agriculture" in v:
        return "Agriculture"
    if "energy" in v:
        return "Energy"
    if "industrial" in v:
        return "Industrial"
    if "waste" in v:
        return "Waste"
    if "wetland" in v:
        return "Wetland"
    if "livestock" in v or "enteric" in v:
        return "Livestock"

    return ""


def short_label(name):
    """
    热图短标签，适配新变量命名。
    """
    v = name.lower()
    sec = sector_short_name(name)

    if "entericshare" in v:
        if "trend" in v:
            return "Enteric share\ntrend"
        return "Enteric share\nmean"

    if v.startswith("r_"):
        if sec:
            if "trend" in v:
                return f"R {sec}\ntrend"
            return f"R {sec}\nmean"
        return name.replace("_", "\n")

    if v.startswith("i_"):
        if sec:
            if "trend" in v:
                return f"I {sec}\ntrend"
            return f"I {sec}\nmean"
        return name.replace("_", "\n")

    label_map = {
        "MAT_mean": "MAT\nmean",
        "MAT_trend": "MAT\ntrend",
        "MAP_mean": "MAP\nmean",
        "MAP_trend": "MAP\ntrend",
        "ST_mean": "ST\nmean",
        "ST_trend": "ST\ntrend",
        "SM_mean": "SM\nmean",
        "SM_trend": "SM\ntrend",
        "GPP_mean": "GPP\nmean",
        "GPP_trend": "GPP\ntrend",
        "Texture_PC1": "Texture\nPC1",
        "Texture_PC2": "Texture\nPC2",
        "SOC": "SOC",
        "BD": "BD"
    }

    return label_map.get(name, name.replace("_", "\n"))


def p_to_stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def compute_spearman(df, vars_list):
    """
    计算 Spearman 相关矩阵、P 值矩阵、样本量矩阵和变量对表。
    """
    corr_mat = pd.DataFrame(np.nan, index=vars_list, columns=vars_list)
    p_mat = pd.DataFrame(np.nan, index=vars_list, columns=vars_list)
    n_mat = pd.DataFrame(0, index=vars_list, columns=vars_list)

    pair_rows = []

    for i, v1 in enumerate(vars_list):
        for j, v2 in enumerate(vars_list):

            if j < i:
                corr_mat.loc[v1, v2] = corr_mat.loc[v2, v1]
                p_mat.loc[v1, v2] = p_mat.loc[v2, v1]
                n_mat.loc[v1, v2] = n_mat.loc[v2, v1]
                continue

            x = safe_numeric(df[v1])
            y = safe_numeric(df[v2])

            ok = x.notna() & y.notna()
            n = int(ok.sum())

            n_mat.loc[v1, v2] = n

            if v1 == v2:
                corr_mat.loc[v1, v2] = 1.0
                p_mat.loc[v1, v2] = 0.0
                continue

            if n < min_pairwise_n:
                continue

            r, p = spearmanr(x[ok], y[ok])

            if not np.isfinite(r):
                continue

            corr_mat.loc[v1, v2] = r
            p_mat.loc[v1, v2] = p

            pair_rows.append({
                "var1": v1,
                "var2": v2,
                "spearman_r": r,
                "abs_r": abs(r),
                "p_value": p,
                "n_pairwise": n
            })

    pair_df = pd.DataFrame(
        pair_rows,
        columns=["var1", "var2", "spearman_r", "abs_r", "p_value", "n_pairwise"]
    )

    if len(pair_df) > 0:
        pair_df = pair_df.sort_values("abs_r", ascending=False).reset_index(drop=True)

    return corr_mat, p_mat, n_mat, pair_df


def univariate_logistic_aic(df, y_col, var):
    """
    单变量 Logistic AIC，仅用于高相关变量并列删除时的辅助判断。
    """
    tmp = df[[y_col, var]].copy()
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp[var] = safe_numeric(tmp[var])

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    tmp = tmp[tmp[y_col].isin([0, 1])].copy()

    if len(tmp) < 30:
        return np.nan

    y = tmp[y_col].astype(int).values
    x = tmp[var].astype(float)

    x_mean = x.mean()
    x_std = x.std(ddof=0)

    if not np.isfinite(x_std) or x_std == 0:
        return np.nan

    x_z = (x - x_mean) / x_std

    X = sm.add_constant(pd.DataFrame({var: x_z}), has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = sm.GLM(
                y,
                X,
                family=sm.families.Binomial()
            )

            res = model.fit(maxiter=300, disp=0)

        return float(res.aic)

    except Exception:
        return np.nan


def get_mw_info(mw_df, var):
    sub = mw_df[mw_df["variable"] == var]

    if len(sub) == 0:
        return {
            "factor_type": infer_factor_type(var),
            "variable_cn": var,
            "p_adj_BH": np.nan,
            "rank_biserial_correlation": np.nan,
            "abs_rbc": np.nan,
            "direction": ""
        }

    rbc = sub["rank_biserial_correlation"].iloc[0] if "rank_biserial_correlation" in sub.columns else np.nan

    return {
        "factor_type": sub["factor_type"].iloc[0] if "factor_type" in sub.columns else infer_factor_type(var),
        "variable_cn": sub["variable_cn"].iloc[0] if "variable_cn" in sub.columns else var,
        "p_adj_BH": sub["p_adj_BH"].iloc[0] if "p_adj_BH" in sub.columns else np.nan,
        "rank_biserial_correlation": rbc,
        "abs_rbc": abs(rbc) if np.isfinite(rbc) else np.nan,
        "direction": sub["direction"].iloc[0] if "direction" in sub.columns else ""
    }


def choose_hub_variable_to_drop(df, mw_df, high_pairs_df):
    """
    从当前高相关变量网络中选择一个要删除的变量。

    删除规则：
    1. 不删除 force_keep_vars；
    2. force_drop_vars 若出现在高相关网络中，优先删除；
    3. 删除高相关连接数 degree 最大的变量；
    4. 如果 degree 并列，删除平均 |r| 更大的变量；
    5. 如果仍并列，优先删除 PC/PCA 类综合变量；
    6. 如果仍并列，删除单变量 AIC 更差的变量；
    7. 如果仍并列，删除 |rbc| 更小的变量；
    8. 如果仍并列，删除 FDR P 值更大的变量。
    """
    vars_in_pairs = sorted(set(high_pairs_df["var1"]).union(set(high_pairs_df["var2"])))

    force_drop_in_pairs = [
        v for v in vars_in_pairs
        if v in force_drop_vars and v not in force_keep_vars
    ]

    if len(force_drop_in_pairs) > 0:
        return sorted(force_drop_in_pairs)[0], "force_drop"

    records = []

    for v in vars_in_pairs:

        if v in force_keep_vars:
            continue

        sub = high_pairs_df[
            (high_pairs_df["var1"] == v) |
            (high_pairs_df["var2"] == v)
        ].copy()

        degree = len(sub)
        mean_abs_r = float(sub["abs_r"].mean()) if degree > 0 else np.nan
        max_abs_r = float(sub["abs_r"].max()) if degree > 0 else np.nan

        aic = univariate_logistic_aic(df, "Y", v)
        mw = get_mw_info(mw_df, v)

        records.append({
            "variable": v,
            "degree": degree,
            "mean_abs_r_in_high_pairs": mean_abs_r,
            "max_abs_r_in_high_pairs": max_abs_r,
            "single_variable_AIC": aic,
            "is_PC_variable": int(is_pc_variable(v)),
            "p_adj_BH": mw["p_adj_BH"],
            "rank_biserial_correlation": mw["rank_biserial_correlation"],
            "abs_rbc": mw["abs_rbc"],
            "factor_type": mw["factor_type"],
            "direction": mw["direction"]
        })

    score_df = pd.DataFrame(records)

    if len(score_df) == 0:
        raise ValueError("所有高相关变量都在 force_keep_vars 中，无法继续删除。")

    score_df["AIC_sort"] = score_df["single_variable_AIC"].fillna(-999999)
    score_df["abs_rbc_sort"] = score_df["abs_rbc"].fillna(999999)
    score_df["p_sort"] = score_df["p_adj_BH"].fillna(-999999)

    sort_cols = [
        "degree",
        "mean_abs_r_in_high_pairs"
    ]

    ascending = [
        False,
        False
    ]

    if prefer_drop_pc_variables_when_tie:
        sort_cols.append("is_PC_variable")
        ascending.append(False)

    sort_cols += [
        "AIC_sort",
        "abs_rbc_sort",
        "p_sort",
        "variable"
    ]

    ascending += [
        False,   # AIC 大，模型更差，优先删
        True,    # |rbc| 小，效应弱，优先删
        False,   # p_adj 大，显著性弱，优先删
        True
    ]

    score_df = score_df.sort_values(
        by=sort_cols,
        ascending=ascending
    ).reset_index(drop=True)

    drop_var = score_df.loc[0, "variable"]

    reason = (
        f"max_degree={score_df.loc[0, 'degree']}; "
        f"mean_abs_r={score_df.loc[0, 'mean_abs_r_in_high_pairs']:.3f}; "
        f"AIC={score_df.loc[0, 'single_variable_AIC']}; "
        f"abs_rbc={score_df.loc[0, 'abs_rbc']}"
    )

    return drop_var, reason


def plot_upper_triangle_corr(corr_mat, p_mat, out_png, out_pdf, title):
    vars_list = list(corr_mat.index)
    n = len(vars_list)

    if n == 0:
        return

    corr_values = corr_mat.values.astype(float)
    p_values = p_mat.values.astype(float)

    mask = np.tril(np.ones_like(corr_values, dtype=bool), k=-1)
    corr_masked = np.ma.array(corr_values, mask=mask)

    fig_size = max(7, n * 0.75)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=figure_dpi)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        corr_masked,
        vmin=-1,
        vmax=1,
        cmap=cmap
    )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    ax.set_xticklabels(
        [short_label(v) for v in vars_list],
        rotation=90,
        fontsize=7
    )

    ax.set_yticklabels(
        [short_label(v) for v in vars_list],
        fontsize=7
    )

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    ax.grid(
        which="minor",
        color="gray",
        linestyle="-",
        linewidth=0.3,
        alpha=0.45
    )

    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            if j < i:
                continue

            r = corr_values[i, j]
            p = p_values[i, j]

            if not np.isfinite(r):
                continue

            if i == j:
                text = "1.00"
            else:
                text = f"{r:.2f}{p_to_stars(p)}"

            text_color = "white" if abs(r) >= 0.75 else "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=6,
                color=text_color
            )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label("Spearman r", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=11, pad=18)

    fig.text(
        0.5,
        0.02,
        "* P<0.05; ** P<0.01; *** P<0.001",
        ha="center",
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# 6. 检查输入
# ============================================================

if not os.path.exists(factor_dir):
    raise FileNotFoundError(f"解释变量文件夹不存在：{factor_dir}")

if not os.path.exists(cluster_tif):
    raise FileNotFoundError(f"聚类图不存在：{cluster_tif}")

if not os.path.exists(mannwhitney_full_file):
    raise FileNotFoundError(f"Mann-Whitney 完整结果不存在：{mannwhitney_full_file}")

if not os.path.exists(mannwhitney_sig_file):
    raise FileNotFoundError(f"Mann-Whitney 显著变量表不存在：{mannwhitney_sig_file}")


# ============================================================
# 7. 读取 Mann-Whitney 结果
# ============================================================

mw_df = pd.read_csv(mannwhitney_full_file)
sig_df = pd.read_csv(mannwhitney_sig_file)

if "variable" not in sig_df.columns:
    raise ValueError("mannwhitney_significant_variables.csv 缺少 variable 列。")

sig_vars = sig_df["variable"].dropna().astype(str).tolist()
sig_vars = list(dict.fromkeys(sig_vars))

if len(sig_vars) == 0:
    raise ValueError("没有 Mann-Whitney FDR 显著变量，无法进行相关性筛选。")

if "factor_type" not in mw_df.columns:
    mw_df["factor_type"] = mw_df["variable"].apply(infer_factor_type)

mw_df["factor_type"] = mw_df.apply(
    lambda r: infer_factor_type(r["variable"]) if pd.isna(r["factor_type"]) else r["factor_type"],
    axis=1
)

print("Mann-Whitney FDR 显著变量数量：", len(sig_vars))


# ============================================================
# 8. 匹配显著变量对应 tif
# ============================================================

tif_lookup = build_tif_lookup(factor_dir)

var_to_tif = {}

for _, row in sig_df.iterrows():
    v = str(row["variable"])

    # 优先使用 Mann-Whitney 表中的 source_tif
    if "source_tif" in sig_df.columns:
        src = row.get("source_tif", "")
        if isinstance(src, str) and os.path.exists(src):
            var_to_tif[v] = src
            continue

    # 否则按变量名匹配 tif
    if v in tif_lookup:
        var_to_tif[v] = tif_lookup[v]

missing_vars = [v for v in sig_vars if v not in var_to_tif]

if len(missing_vars) > 0:
    miss_file = os.path.join(out_dir, "00_missing_significant_variables_no_tif.csv")

    pd.DataFrame({
        "missing_variable": missing_vars
    }).to_csv(
        miss_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("[警告] 以下显著变量没有找到对应 tif，将跳过：")
    for v in missing_vars:
        print(" -", v)

sig_vars = [v for v in sig_vars if v in var_to_tif]

if len(sig_vars) < 2:
    raise ValueError("可用于相关性分析的显著变量少于 2 个。")

pd.DataFrame({
    "selected_variable": sig_vars,
    "tif_path": [var_to_tif[v] for v in sig_vars],
    "factor_type": [get_mw_info(mw_df, v)["factor_type"] for v in sig_vars],
    "variable_cn": [get_mw_info(mw_df, v)["variable_cn"] for v in sig_vars]
}).to_csv(
    os.path.join(out_dir, "00_significant_variable_tif_mapping.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("匹配到 tif 的显著变量数量：", len(sig_vars))


# ============================================================
# 9. 读取聚类图并建立样本表
# ============================================================

cluster_arr, template_profile = read_cluster_template(cluster_tif)

height = template_profile["height"]
width = template_profile["width"]
transform = template_profile["transform"]

base_mask = (
    np.isfinite(cluster_arr) &
    ((cluster_arr == positive_cluster) | (cluster_arr == negative_cluster))
)

if int(base_mask.sum()) == 0:
    raise ValueError("没有正向/反向样本，请检查 positive_cluster 和 negative_cluster。")

rows, cols = np.where(base_mask)
xs, ys = rasterio.transform.xy(transform, rows, cols)

df = pd.DataFrame({
    "row": rows,
    "col": cols,
    "lon": xs,
    "lat": ys,
    "cluster": cluster_arr[rows, cols].astype(int)
})

df["Y"] = np.where(df["cluster"] == positive_cluster, 1, 0)

print("正向型样本数：", int((df["Y"] == 1).sum()))
print("反向型样本数：", int((df["Y"] == 0).sum()))


# ============================================================
# 10. 从 tif 提取显著变量值
# ============================================================

print("\n正在从 0.1° tif 提取显著变量，并聚合到 1° 聚类网格...")

for v in sig_vars:

    tif = var_to_tif[v]

    print("提取变量：", v)

    arr = read_factor_to_cluster_grid(
        tif,
        template_profile
    )

    df[v] = arr[rows, cols]

factor_table_file = os.path.join(
    out_dir,
    "01_significant_factor_table_from_tif_aggregated_to_1deg.csv"
)

df.to_csv(
    factor_table_file,
    index=False,
    encoding="utf-8-sig"
)

print("显著变量样本表已输出：", factor_table_file)


# ============================================================
# 11. 初始相关性矩阵和高相关对
# ============================================================

corr_initial, p_initial, n_initial, pair_initial = compute_spearman(df, sig_vars)

corr_initial.to_csv(
    os.path.join(out_dir, "02_initial_spearman_corr_matrix.csv"),
    encoding="utf-8-sig"
)

p_initial.to_csv(
    os.path.join(out_dir, "03_initial_spearman_pvalue_matrix.csv"),
    encoding="utf-8-sig"
)

n_initial.to_csv(
    os.path.join(out_dir, "04_initial_pairwise_n_matrix.csv"),
    encoding="utf-8-sig"
)

pair_initial.to_csv(
    os.path.join(out_dir, "05_initial_spearman_pairs.csv"),
    index=False,
    encoding="utf-8-sig"
)

initial_high_pairs = pair_initial[pair_initial["abs_r"] >= corr_threshold].copy()

initial_high_pairs.to_csv(
    os.path.join(out_dir, "06_initial_high_correlation_pairs.csv"),
    index=False,
    encoding="utf-8-sig"
)

plot_upper_triangle_corr(
    corr_initial,
    p_initial,
    os.path.join(out_dir, "07_initial_correlation_heatmap_upper_triangle.png"),
    os.path.join(out_dir, "07_initial_correlation_heatmap_upper_triangle.pdf"),
    title="Initial Spearman correlation among significant variables"
)

print("初始高相关变量对数量：", len(initial_high_pairs))


# ============================================================
# 12. 迭代删除高连接变量
# ============================================================

remaining_vars = [v for v in sig_vars if v not in force_drop_vars]
dropped_vars = []

drop_log_rows = []
degree_log_rows = []

iteration = 1

while True:

    if len(remaining_vars) < 2:
        break

    corr_now, p_now, n_now, pair_now = compute_spearman(df, remaining_vars)

    high_pairs_now = pair_now[pair_now["abs_r"] >= corr_threshold].copy()

    if len(high_pairs_now) == 0:
        print("最终变量中已无高相关变量对。")
        break

    vars_in_pairs = sorted(set(high_pairs_now["var1"]).union(set(high_pairs_now["var2"])))

    for v in vars_in_pairs:
        sub = high_pairs_now[
            (high_pairs_now["var1"] == v) |
            (high_pairs_now["var2"] == v)
        ]

        mw = get_mw_info(mw_df, v)

        degree_log_rows.append({
            "iteration": iteration,
            "variable": v,
            "degree": len(sub),
            "mean_abs_r_in_high_pairs": float(sub["abs_r"].mean()),
            "max_abs_r_in_high_pairs": float(sub["abs_r"].max()),
            "single_variable_AIC": univariate_logistic_aic(df, "Y", v),
            "factor_type": mw["factor_type"],
            "variable_cn": mw["variable_cn"],
            "p_adj_BH": mw["p_adj_BH"],
            "rank_biserial_correlation": mw["rank_biserial_correlation"],
            "abs_rbc": mw["abs_rbc"],
            "is_PC_variable": is_pc_variable(v)
        })

    drop_var, reason = choose_hub_variable_to_drop(
        df=df,
        mw_df=mw_df,
        high_pairs_df=high_pairs_now
    )

    related_pairs = high_pairs_now[
        (high_pairs_now["var1"] == drop_var) |
        (high_pairs_now["var2"] == drop_var)
    ].copy()

    related_vars = sorted(
        set(related_pairs["var1"]).union(set(related_pairs["var2"])) - {drop_var}
    )

    drop_log_rows.append({
        "iteration": iteration,
        "dropped_variable": drop_var,
        "drop_reason": reason,
        "degree_before_drop": len(related_pairs),
        "related_variables": ";".join(related_vars),
        "related_pairs": "; ".join(
            related_pairs.apply(
                lambda r: f"{r['var1']}~{r['var2']}:{r['spearman_r']:.3f}",
                axis=1
            ).tolist()
        ),
        "n_remaining_before_drop": len(remaining_vars)
    })

    print(
        f"第 {iteration} 轮删除：{drop_var}；"
        f"连接数={len(related_pairs)}；相关变量={related_vars}"
    )

    remaining_vars = [v for v in remaining_vars if v != drop_var]
    dropped_vars.append(drop_var)

    iteration += 1


# ============================================================
# 13. 输出筛选日志
# ============================================================

drop_log_df = pd.DataFrame(drop_log_rows)
degree_log_df = pd.DataFrame(degree_log_rows)

drop_log_df.to_csv(
    os.path.join(out_dir, "08_iterative_hub_variable_drop_log.csv"),
    index=False,
    encoding="utf-8-sig"
)

degree_log_df.to_csv(
    os.path.join(out_dir, "09_degree_score_each_iteration.csv"),
    index=False,
    encoding="utf-8-sig"
)

final_var_df = pd.DataFrame({
    "selected_variable": remaining_vars,
    "factor_type": [get_mw_info(mw_df, v)["factor_type"] for v in remaining_vars],
    "variable_cn": [get_mw_info(mw_df, v)["variable_cn"] for v in remaining_vars]
})

final_var_df.to_csv(
    os.path.join(out_dir, "10_final_variables_after_corr_filter.csv"),
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "dropped_variable": dropped_vars,
    "factor_type": [get_mw_info(mw_df, v)["factor_type"] for v in dropped_vars],
    "variable_cn": [get_mw_info(mw_df, v)["variable_cn"] for v in dropped_vars]
}).to_csv(
    os.path.join(out_dir, "11_dropped_variables_by_corr_filter.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 最终相关性矩阵和热图
# ============================================================

corr_final, p_final, n_final, pair_final = compute_spearman(df, remaining_vars)

corr_final.to_csv(
    os.path.join(out_dir, "12_final_spearman_corr_matrix.csv"),
    encoding="utf-8-sig"
)

p_final.to_csv(
    os.path.join(out_dir, "13_final_spearman_pvalue_matrix.csv"),
    encoding="utf-8-sig"
)

n_final.to_csv(
    os.path.join(out_dir, "14_final_pairwise_n_matrix.csv"),
    encoding="utf-8-sig"
)

pair_final.to_csv(
    os.path.join(out_dir, "15_final_spearman_pairs.csv"),
    index=False,
    encoding="utf-8-sig"
)

final_high_pairs = pair_final[pair_final["abs_r"] >= corr_threshold].copy()

final_high_pairs.to_csv(
    os.path.join(out_dir, "16_final_high_correlation_pairs_remaining.csv"),
    index=False,
    encoding="utf-8-sig"
)

if len(remaining_vars) >= 2:
    plot_upper_triangle_corr(
        corr_final,
        p_final,
        os.path.join(out_dir, "17_final_correlation_heatmap_upper_triangle.png"),
        os.path.join(out_dir, "17_final_correlation_heatmap_upper_triangle.pdf"),
        title="Final Spearman correlation after deleting high-degree variables"
    )
else:
    print("[提示] 最终变量少于 2 个，跳过最终相关性热图。")


# ============================================================
# 15. 变量筛选总表
# ============================================================

summary_rows = []

for v in sig_vars:
    mw = get_mw_info(mw_df, v)

    if v in remaining_vars:
        status = "selected_final"
    elif v in dropped_vars:
        status = "dropped_high_correlation"
    elif v in force_drop_vars:
        status = "force_dropped"
    else:
        status = "not_selected_unknown"

    summary_rows.append({
        "variable": v,
        "variable_cn": mw["variable_cn"],
        "factor_type": mw["factor_type"],
        "selection_status": status,
        "p_adj_BH": mw["p_adj_BH"],
        "rank_biserial_correlation": mw["rank_biserial_correlation"],
        "abs_rbc": mw["abs_rbc"],
        "direction": mw["direction"],
        "tif_path": var_to_tif.get(v, "")
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    os.path.join(out_dir, "18_variable_selection_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 完成
# ============================================================

print("\n全部完成！输出文件夹：")
print(out_dir)

print("\n初始显著变量数量：", len(sig_vars))
print("最终变量数量：", len(remaining_vars))

print("\n最终变量：")
for v in remaining_vars:
    print(" -", v, "｜", get_mw_info(mw_df, v)["variable_cn"])

print("\n删除变量数量：", len(dropped_vars))
for v in dropped_vars:
    print(" -", v, "｜", get_mw_info(mw_df, v)["variable_cn"])

print("\n重点查看：")
print("00_significant_variable_tif_mapping.csv")
print("01_significant_factor_table_from_tif_aggregated_to_1deg.csv")
print("06_initial_high_correlation_pairs.csv")
print("08_iterative_hub_variable_drop_log.csv")
print("10_final_variables_after_corr_filter.csv")
print("16_final_high_correlation_pairs_remaining.csv")
print("17_final_correlation_heatmap_upper_triangle.pdf")
print("18_variable_selection_summary.csv")

print("\n说明：")
print("1. 本代码只对 Mann-Whitney FDR 显著变量做 Spearman 冗余筛选。")
print("2. 0.1°解释变量已用 average 聚合到 1°聚类网格。")
print("3. 若最终仍存在高相关变量，请查看 16_final_high_correlation_pairs_remaining.csv。")