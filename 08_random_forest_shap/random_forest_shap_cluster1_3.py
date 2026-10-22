# -*- coding: utf-8 -*-
# ============================================================
# RF 二分类 + SHAP 解释
#
# 目标：
# 用相关性筛选后的最终变量区分：
# Y = 1：牲畜主导正向响应型 Cluster 1
# Y = 0：牲畜主导反向响应型 Cluster 3
#
# 当前版本适配新流程：
# 1. 解释变量来自 0.1° tif：
#    F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\0_解释变量
#
# 2. 最终变量来自 Spearman 冗余筛选：
#    ...\2_spearman_corr_filter\10_final_variables_after_corr_filter.csv
#
# 3. 聚类结果为 1°：
#    0.1°解释变量用 Resampling.average 聚合到 1° 聚类网格
#
# 输出：
# 1. RF 模型性能
# 2. 混淆矩阵
# 3. ROC 曲线
# 4. RF Gini 变量重要性
# 5. Permutation importance
# 6. SHAP importance
# 7. SHAP summary plot
# 8. SHAP dependence plots
# 9. 测试集预测结果
# ============================================================

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import shap


# ============================================================
# 1. 路径设置
# ============================================================

# 解释变量 tif 文件夹：0.1°
factor_dir = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\0_解释变量"

# 6SC 聚类结果：1°
cluster_tif = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\3_聚类_6SC_only\03_response_cluster_1deg_6SC_k6.tif"

# Mann-Whitney 输出文件夹
mw_dir = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\1_mannwhitney_6SC_pos_vs_neg"

# Spearman 相关性筛选结果文件夹
corr_filter_dir = os.path.join(
    mw_dir,
    "2_spearman_corr_filter"
)

# 最终变量清单
final_var_file = os.path.join(
    corr_filter_dir,
    "10_final_variables_after_corr_filter.csv"
)

# 可选：显著变量与 tif 映射表
sig_tif_mapping_file = os.path.join(
    corr_filter_dir,
    "00_significant_variable_tif_mapping.csv"
)

# RF/SHAP 输出文件夹
out_dir = r"F:\DOCTOR\graduating\test2\40\3_thirdpart\3_解释\3_RF_SHAP_6SC_pos_vs_neg"

os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

# Cluster 1 = 牲畜主导正向响应型
# Cluster 3 = 牲畜主导反向响应型
positive_cluster = 1
negative_cluster = 3

positive_label = "Positive_livestock_dominant"
negative_label = "Negative_livestock_dominant"

random_state = 42
test_size = 0.30

n_estimators = 1000
max_depth = None
min_samples_leaf = 3
min_samples_split = 6
max_features = "sqrt"

n_splits_cv = 5

# SHAP dependence plot 输出前几个变量
top_n_dependence = 6

# 变量有效性过滤
min_valid_total = 30
min_valid_per_class = 10

# NoData
nodata_values = [-9999, -9999.0, -3.4e38, 3.4e38]


# ============================================================
# 3. 字体设置
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 4. 工具函数
# ============================================================

def clean_var_name(filename):
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
    如果解释变量与聚类图同网格，直接读取；
    如果是 0.1°，则用 average 聚合到 1° 聚类网格。
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
    适配新解释变量命名：
    R_Agriculture_mean
    I_log1p_Agriculture_mean
    Entericshare_mean
    """
    v = name.lower()
    sec = sector_short_name(name)

    if "entericshare" in v:
        if "trend" in v:
            return "Enteric share trend"
        return "Enteric share mean"

    if "livestockshare" in v:
        if "trend" in v:
            return "Livestock share trend"
        return "Livestock share mean"

    if v.startswith("r_"):
        if sec:
            if "trend" in v:
                return f"R {sec} trend"
            return f"R {sec} mean"
        return name

    if v.startswith("i_"):
        if sec:
            if "trend" in v:
                return f"I {sec} trend"
            return f"I {sec} mean"
        return name

    label_map = {
        "MAT_mean": "MAT mean",
        "MAT_trend": "MAT trend",
        "MAP_mean": "MAP mean",
        "MAP_trend": "MAP trend",
        "ST_mean": "ST mean",
        "ST_trend": "ST trend",
        "SM_mean": "SM mean",
        "SM_trend": "SM trend",
        "GPP_mean": "GPP mean",
        "GPP_trend": "GPP trend",
        "Texture_PC1": "Texture PC1",
        "Texture_PC2": "Texture PC2",
        "SOC": "SOC",
        "BD": "BD"
    }

    return label_map.get(name, name)


def get_variable_cn(var, var_df=None):
    """
    优先使用变量清单中的中文名；没有则按变量名简单生成。
    """
    if var_df is not None and "selected_variable" in var_df.columns and "variable_cn" in var_df.columns:
        sub = var_df[var_df["selected_variable"] == var]
        if len(sub) > 0:
            x = sub["variable_cn"].iloc[0]
            if isinstance(x, str) and x.strip() != "":
                return x

    v = var.lower()
    sec = sector_short_name(var)

    sec_cn_map = {
        "Agriculture": "农业",
        "Energy": "能源",
        "Industrial": "工业",
        "Waste": "废弃物",
        "Wetland": "湿地",
        "Livestock": "牲畜"
    }

    sec_cn = sec_cn_map.get(sec, "")

    if "entericshare" in v or "livestockshare" in v:
        return "牲畜排放占比趋势" if "trend" in v else "牲畜排放占比"

    if v.startswith("r_"):
        if sec_cn:
            return f"{sec_cn}排放结构趋势" if "trend" in v else f"{sec_cn}排放结构"
        return var

    if v.startswith("i_"):
        if sec_cn:
            return f"{sec_cn}排放强度趋势" if "trend" in v else f"{sec_cn}排放强度"
        return var

    if v.startswith("mat"):
        return "MAT趋势" if "trend" in v else "MAT"
    if v.startswith("map"):
        return "MAP趋势" if "trend" in v else "MAP"
    if v.startswith("st"):
        return "ST趋势" if "trend" in v else "ST"
    if v.startswith("sm"):
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


def extract_shap_values_positive(shap_values_raw):
    """
    兼容不同 shap 版本：
    - list: [class0, class1]
    - ndarray: n_samples × n_features × n_classes
    - ndarray: n_samples × n_features
    """
    if isinstance(shap_values_raw, list):
        return shap_values_raw[1]

    if hasattr(shap_values_raw, "values"):
        arr = np.asarray(shap_values_raw.values)
    else:
        arr = np.asarray(shap_values_raw)

    if arr.ndim == 3:
        if arr.shape[2] >= 2:
            return arr[:, :, 1]
        return arr[:, :, 0]

    if arr.ndim == 2:
        return arr

    raise ValueError(f"无法识别 SHAP 输出维度：{arr.shape}")


# ============================================================
# 5. 检查输入
# ============================================================

if not os.path.exists(factor_dir):
    raise FileNotFoundError(f"解释变量文件夹不存在：{factor_dir}")

if not os.path.exists(cluster_tif):
    raise FileNotFoundError(f"聚类图不存在：{cluster_tif}")

if not os.path.exists(final_var_file):
    raise FileNotFoundError(f"最终变量清单不存在：{final_var_file}")


# ============================================================
# 6. 读取最终变量
# ============================================================

var_df = pd.read_csv(final_var_file)

if "selected_variable" in var_df.columns:
    feature_vars = var_df["selected_variable"].dropna().astype(str).tolist()
elif "variable" in var_df.columns:
    feature_vars = var_df["variable"].dropna().astype(str).tolist()
else:
    raise ValueError("最终变量清单缺少 selected_variable 或 variable 列。")

feature_vars = list(dict.fromkeys(feature_vars))

if len(feature_vars) < 2:
    raise ValueError("最终变量少于 2 个，无法进行 RF/SHAP。")

print("最终变量清单数量：", len(feature_vars))
for v in feature_vars:
    print(" -", v)


# ============================================================
# 7. 建立变量 tif 映射
# ============================================================

tif_lookup = build_tif_lookup(factor_dir)

# 如果 Spearman 文件夹中有显著变量映射表，则优先读取 source/tif 信息
mapping_df = None
if os.path.exists(sig_tif_mapping_file):
    mapping_df = pd.read_csv(sig_tif_mapping_file)

var_to_tif = {}

for v in feature_vars:

    # 1) 优先用 mapping 表里的 tif_path
    if mapping_df is not None and "selected_variable" in mapping_df.columns and "tif_path" in mapping_df.columns:
        sub = mapping_df[mapping_df["selected_variable"] == v]
        if len(sub) > 0:
            p = sub["tif_path"].iloc[0]
            if isinstance(p, str) and os.path.exists(p):
                var_to_tif[v] = p
                continue

    # 2) 再用文件名匹配
    if v in tif_lookup:
        var_to_tif[v] = tif_lookup[v]

missing_vars = [v for v in feature_vars if v not in var_to_tif]

if len(missing_vars) > 0:
    missing_file = os.path.join(out_dir, "00_missing_final_variables_no_tif.csv")
    pd.DataFrame({"missing_variable": missing_vars}).to_csv(
        missing_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("[警告] 以下最终变量没有找到对应 tif，将跳过：")
    for v in missing_vars:
        print(" -", v)

feature_vars = [v for v in feature_vars if v in var_to_tif]

if len(feature_vars) < 2:
    raise ValueError("匹配到 tif 的最终变量少于 2 个，无法进行 RF/SHAP。")

pd.DataFrame({
    "selected_variable": feature_vars,
    "variable_cn": [get_variable_cn(v, var_df) for v in feature_vars],
    "tif_path": [var_to_tif[v] for v in feature_vars]
}).to_csv(
    os.path.join(out_dir, "00_final_variable_tif_mapping.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 读取聚类图，建立样本表
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
    "grid_id": np.arange(1, len(rows) + 1),
    "row": rows,
    "col": cols,
    "lon": xs,
    "lat": ys,
    "cluster": cluster_arr[rows, cols].astype(int)
})

df["response_type"] = np.where(
    df["cluster"] == positive_cluster,
    positive_label,
    negative_label
)

df["Y"] = np.where(
    df["cluster"] == positive_cluster,
    1,
    0
)

print("\n样本量：", len(df))
print("类别分布：")
print(df["Y"].value_counts())


# ============================================================
# 9. 从 0.1° tif 提取最终变量，并聚合到 1°
# ============================================================

print("\n正在从 0.1° tif 提取最终变量，并聚合到 1° 聚类网格...")

for v in feature_vars:

    tif = var_to_tif[v]
    print("提取变量：", v)

    arr = read_factor_to_cluster_grid(
        tif,
        template_profile
    )

    df[v] = arr[rows, cols]

factor_table_file = os.path.join(
    out_dir,
    "01_RF_SHAP_factor_table_from_tif_aggregated_to_1deg.csv"
)

df.to_csv(
    factor_table_file,
    index=False,
    encoding="utf-8-sig"
)

print("RF/SHAP 输入因子表已输出：", factor_table_file)


# ============================================================
# 10. 变量有效性过滤
# ============================================================

quality_rows = []
valid_feature_vars = []

for v in feature_vars:

    x = pd.to_numeric(df[v], errors="coerce").replace([np.inf, -np.inf], np.nan)

    n_total = int(x.notna().sum())
    n_pos = int(x[df["Y"] == 1].notna().sum())
    n_neg = int(x[df["Y"] == 0].notna().sum())
    n_unique = int(x.dropna().nunique())

    keep = (
        n_total >= min_valid_total and
        n_pos >= min_valid_per_class and
        n_neg >= min_valid_per_class and
        n_unique >= 2
    )

    quality_rows.append({
        "variable": v,
        "variable_cn": get_variable_cn(v, var_df),
        "n_total_valid": n_total,
        "n_positive_valid": n_pos,
        "n_negative_valid": n_neg,
        "n_unique": n_unique,
        "keep_for_RF_SHAP": keep
    })

    if keep:
        valid_feature_vars.append(v)

quality_df = pd.DataFrame(quality_rows)

quality_file = os.path.join(out_dir, "02_feature_data_quality_check.csv")
quality_df.to_csv(
    quality_file,
    index=False,
    encoding="utf-8-sig"
)

feature_vars = valid_feature_vars

if len(feature_vars) < 2:
    raise ValueError("经过有效性过滤后，最终变量少于 2 个，无法进行 RF/SHAP。")

print("\n进入 RF/SHAP 的变量数量：", len(feature_vars))
for v in feature_vars:
    print(" -", v, "｜", get_variable_cn(v, var_df))


# ============================================================
# 11. 构建建模数据
# ============================================================

data = df[["Y"] + feature_vars].copy()
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=["Y"])

X = data[feature_vars]
y = data["Y"].astype(int)

if y.nunique() != 2:
    raise ValueError("Y 不是二分类，请检查正向/反向样本。")

class_counts = y.value_counts()
min_class_n = int(class_counts.min())

if min_class_n < 5:
    raise ValueError("某一类样本过少，无法稳定建模。")

print("\n建模样本量：", len(data))
print("建模类别分布：")
print(class_counts)


# ============================================================
# 12. 划分训练集和测试集
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

print("\n训练集类别分布：")
print(y_train.value_counts())

print("\n测试集类别分布：")
print(y_test.value_counts())


# ============================================================
# 13. 建立 RF 模型
# ============================================================

rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    max_features=max_features,
    class_weight="balanced",
    random_state=random_state,
    n_jobs=-1,
    oob_score=True
)

model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", rf)
])

model.fit(X_train, y_train)

rf_model = model.named_steps["rf"]


# ============================================================
# 14. 模型预测与评价
# ============================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Negative livestock-dominant",
        "Positive livestock-dominant"
    ],
    output_dict=True,
    zero_division=0
)

# 交叉验证 AUC
cv_splits = min(n_splits_cv, min_class_n)

if cv_splits >= 2:
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    cv_auc = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    cv_auc_mean = float(cv_auc.mean())
    cv_auc_std = float(cv_auc.std())

else:
    cv_auc_mean = np.nan
    cv_auc_std = np.nan

performance_df = pd.DataFrame([{
    "n_samples": len(data),
    "n_features": len(feature_vars),
    "test_size": test_size,
    "accuracy": acc,
    "auc": auc,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "cv_auc_mean": cv_auc_mean,
    "cv_auc_std": cv_auc_std,
    "oob_score": rf_model.oob_score_
}])

performance_file = os.path.join(out_dir, "03_RF_model_performance.csv")
performance_df.to_csv(
    performance_file,
    index=False,
    encoding="utf-8-sig"
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_0_negative", "Actual_1_positive"],
    columns=["Pred_0_negative", "Pred_1_positive"]
)

cm_file = os.path.join(out_dir, "04_RF_confusion_matrix.csv")
cm_df.to_csv(
    cm_file,
    encoding="utf-8-sig"
)

report_file = os.path.join(out_dir, "05_RF_classification_report.csv")
pd.DataFrame(report).T.to_csv(
    report_file,
    encoding="utf-8-sig"
)

print("\n模型性能：")
print(performance_df)


# ============================================================
# 15. 绘制 ROC 曲线
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve: Positive vs Negative livestock-dominant type")
plt.legend()
plt.tight_layout()

roc_pdf = os.path.join(out_dir, "06_RF_ROC_curve.pdf")
roc_png = os.path.join(out_dir, "06_RF_ROC_curve.png")

plt.savefig(roc_pdf, format="pdf", bbox_inches="tight")
plt.savefig(roc_png, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 16. RF Gini 变量重要性
# ============================================================

importance_df = pd.DataFrame({
    "variable": feature_vars,
    "variable_cn": [get_variable_cn(v, var_df) for v in feature_vars],
    "variable_label": [short_label(v) for v in feature_vars],
    "gini_importance": rf_model.feature_importances_
}).sort_values(
    by="gini_importance",
    ascending=False
).reset_index(drop=True)

importance_file = os.path.join(out_dir, "07_RF_gini_feature_importance.csv")
importance_df.to_csv(
    importance_file,
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(7, max(4, len(feature_vars) * 0.35)))
plt.barh(
    importance_df["variable_label"][::-1],
    importance_df["gini_importance"][::-1]
)
plt.xlabel("Gini importance")
plt.title("Random Forest feature importance")
plt.tight_layout()

gini_pdf = os.path.join(out_dir, "07_RF_gini_feature_importance.pdf")
gini_png = os.path.join(out_dir, "07_RF_gini_feature_importance.png")

plt.savefig(gini_pdf, format="pdf", bbox_inches="tight")
plt.savefig(gini_png, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 17. Permutation importance
# ============================================================

perm = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=30,
    random_state=random_state,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "variable": feature_vars,
    "variable_cn": [get_variable_cn(v, var_df) for v in feature_vars],
    "variable_label": [short_label(v) for v in feature_vars],
    "permutation_importance_mean": perm.importances_mean,
    "permutation_importance_std": perm.importances_std
}).sort_values(
    by="permutation_importance_mean",
    ascending=False
).reset_index(drop=True)

perm_file = os.path.join(out_dir, "08_RF_permutation_importance.csv")
perm_df.to_csv(
    perm_file,
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(7, max(4, len(feature_vars) * 0.35)))
plt.barh(
    perm_df["variable_label"][::-1],
    perm_df["permutation_importance_mean"][::-1],
    xerr=perm_df["permutation_importance_std"][::-1]
)
plt.xlabel("Permutation importance based on AUC")
plt.title("Permutation feature importance")
plt.tight_layout()

perm_pdf = os.path.join(out_dir, "08_RF_permutation_importance.pdf")
perm_png = os.path.join(out_dir, "08_RF_permutation_importance.png")

plt.savefig(perm_pdf, format="pdf", bbox_inches="tight")
plt.savefig(perm_png, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 18. SHAP 解释
# ============================================================

imputer = model.named_steps["imputer"]

X_train_imp = pd.DataFrame(
    imputer.transform(X_train),
    columns=feature_vars,
    index=X_train.index
)

X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=feature_vars,
    index=X_test.index
)

explainer = shap.TreeExplainer(rf_model)
shap_values_raw = explainer.shap_values(X_test_imp)
shap_values_pos = extract_shap_values_positive(shap_values_raw)

mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)

shap_importance_df = pd.DataFrame({
    "variable": feature_vars,
    "variable_cn": [get_variable_cn(v, var_df) for v in feature_vars],
    "variable_label": [short_label(v) for v in feature_vars],
    "mean_abs_SHAP": mean_abs_shap
}).sort_values(
    by="mean_abs_SHAP",
    ascending=False
).reset_index(drop=True)

shap_importance_file = os.path.join(out_dir, "09_SHAP_mean_abs_importance.csv")
shap_importance_df.to_csv(
    shap_importance_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. SHAP bar plot
# ============================================================

plt.figure()
shap.summary_plot(
    shap_values_pos,
    X_test_imp,
    feature_names=[short_label(v) for v in feature_vars],
    plot_type="bar",
    show=False
)
plt.title("SHAP feature importance")
plt.tight_layout()

shap_bar_pdf = os.path.join(out_dir, "10_SHAP_bar_importance.pdf")
shap_bar_png = os.path.join(out_dir, "10_SHAP_bar_importance.png")

plt.savefig(shap_bar_pdf, format="pdf", bbox_inches="tight")
plt.savefig(shap_bar_png, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 20. SHAP beeswarm summary plot
# ============================================================

plt.figure()
shap.summary_plot(
    shap_values_pos,
    X_test_imp,
    feature_names=[short_label(v) for v in feature_vars],
    show=False
)
plt.title("SHAP summary plot")
plt.tight_layout()

shap_summary_pdf = os.path.join(out_dir, "11_SHAP_summary_beeswarm.pdf")
shap_summary_png = os.path.join(out_dir, "11_SHAP_summary_beeswarm.png")

plt.savefig(shap_summary_pdf, format="pdf", bbox_inches="tight")
plt.savefig(shap_summary_png, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 21. SHAP dependence plots
# ============================================================

top_vars = shap_importance_df["variable"].head(top_n_dependence).tolist()

dep_dir = os.path.join(out_dir, "12_SHAP_dependence_plots")
os.makedirs(dep_dir, exist_ok=True)

for var in top_vars:

    plt.figure()

    shap.dependence_plot(
        var,
        shap_values_pos,
        X_test_imp,
        feature_names=feature_vars,
        show=False,
        interaction_index=None
    )

    plt.title(f"SHAP dependence: {short_label(var)}")
    plt.tight_layout()

    safe_var = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fa5]+", "_", var)

    out_pdf = os.path.join(dep_dir, f"SHAP_dependence_{safe_var}.pdf")
    out_png = os.path.join(dep_dir, f"SHAP_dependence_{safe_var}.png")

    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 22. 输出测试集预测结果
# ============================================================

pred_cols = [
    "grid_id",
    "row",
    "col",
    "lon",
    "lat",
    "cluster",
    "response_type",
    "Y"
]

pred_df = df.loc[X_test.index, pred_cols].copy()

pred_df["y_true"] = y_test.values
pred_df["y_pred"] = y_pred
pred_df["prob_positive"] = y_prob

pred_file = os.path.join(out_dir, "13_RF_test_prediction_results.csv")
pred_df.to_csv(
    pred_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 23. 输出最终建模变量清单
# ============================================================

final_model_vars_file = os.path.join(out_dir, "14_RF_SHAP_final_model_variables.csv")

pd.DataFrame({
    "selected_variable": feature_vars,
    "variable_cn": [get_variable_cn(v, var_df) for v in feature_vars],
    "short_label": [short_label(v) for v in feature_vars],
    "tif_path": [var_to_tif[v] for v in feature_vars]
}).to_csv(
    final_model_vars_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 24. 完成
# ============================================================

print("\n全部完成！输出文件夹：")
print(out_dir)

print("\n主要输出：")
print("0. 变量 tif 映射：", os.path.join(out_dir, "00_final_variable_tif_mapping.csv"))
print("1. RF/SHAP 输入因子表：", factor_table_file)
print("2. 变量有效性检查：", quality_file)
print("3. 模型性能：", performance_file)
print("4. 混淆矩阵：", cm_file)
print("5. 分类报告：", report_file)
print("6. ROC 曲线：", roc_pdf)
print("7. RF Gini 重要性：", importance_file)
print("8. Permutation 重要性：", perm_file)
print("9. SHAP 重要性：", shap_importance_file)
print("10. SHAP bar：", shap_bar_pdf)
print("11. SHAP summary：", shap_summary_pdf)
print("12. SHAP dependence plots：", dep_dir)
print("13. 测试集预测结果：", pred_file)
print("14. 最终建模变量：", final_model_vars_file)

print("\n解释方向：")
print("SHAP 值 > 0：推动模型判定为正向牲畜主导型 Y=1。")
print("SHAP 值 < 0：推动模型判定为反向牲畜主导型 Y=0。")