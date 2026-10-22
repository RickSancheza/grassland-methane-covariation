# Methods ↔ Code Mapping (Round 2)

Status: **FOUND** / **PARTIAL** / **AUDIT-ONLY** / **RESULTS-ONLY** / **MISSING** / **OLD-VERSION**.
“RESULTS-ONLY” means outputs and/or parameter tables exist but original source was not found in `test2` (or parent).

## 1. Grassland classification and data preparation

| Methods step | Code / evidence | Status |
|--------------|-----------------|--------|
| WWF 6 grassland eco-types | `草地类型图/生态群系/*`, maps | RESULTS-ONLY |
| JRC grassland mask | `0_data/草地掩膜.tif` | RESULTS-ONLY |
| 0.1° alignment of grassland + emissions | `resample_grassland_0p1_and_reclip_emissions*.py`, `clip_emission_by_grasslandtype_mask.py` (nearest; codes 1–6) | PARTIAL (FOUND for clip/resample of analysis stacks) |
| Six-sector emissions 2005–2020 | `0_data/2_排放/*` | RESULTS-ONLY |
| XCH₄ / ERA5 / GPP / SoilGrids prep | `0_data/4_均值`, `5_趋势`, `6_soil` | RESULTS-ONLY |

## 2. Atmospheric CH4 growth rate

| Methods step | Evidence | Status |
|--------------|----------|--------|
| XCH4 annual mean | `无比例/ch4/年均甲烷浓度` | RESULTS-ONLY |
| GR | `无比例/ch4/GR/2004–*` | RESULTS-ONLY |
| ΔGR (2004 base) | `1_Delta_GR`, `无比例/ch4/Delta_GR`, `22/.../1_Delta_GR` (copy) | RESULTS-ONLY |

## 3. Emission change and sector composition

| Methods step | Code / evidence | Status |
|--------------|-----------------|--------|
| ΔE vs 2004 | `3_change/*` | RESULTS-ONLY |
| Sector composition donuts | `plot_donut_groups_for_folder1_folder2.py` + method note | FOUND |
| Enteric share / 40% | legacy `ratio>=0.4` fragment; `grasslandtype_stat_40percent` products | PARTIAL |

## 4. Regression and covariance contribution

| Methods step | Code / evidence | Status |
|--------------|-----------------|--------|
| ΔEnteric vs ΔGR regression | `04_regression_covariance/linear_relationship_deltaGR_livestock.R` | FOUND (path caveat `3_分部门变化`) |
| Grassland-type covariance / SC% | `covariance_by_grassland_type.R` | FOUND |
| Grid SC | `signed_contribution_grid.py` (`cov/Σ\|cov\|×100`) | FOUND |
| (Alt line) 4-sector covariance | `22/2_contribution/3_covariance_result/*` | RESULTS-ONLY, not manuscript 6-sector |

## 5. Response-type clustering

| Methods step | Code / evidence | Status |
|--------------|-----------------|--------|
| 1° aggregation | `agg1deg_SC_*.tif`; `22` 4SC analog | RESULTS-ONLY |
| K-means + DBI K=2–10 | `K2_K10_cell_results_method.txt`, `K2_K10_DBI_summary.xlsx` | RESULTS-ONLY (params known) |
| K=6 selection narrative | K6_selection*.txt, why_K6_not_K7_or_K9.txt | RESULTS-ONLY (text) |
| **union-fill K=6 final raster** | `K6_clusters.tif` + centers xlsx + consistency_check xlsx | RESULTS-ONLY (algorithm text absent) |
| Cluster proportion chart | `cluster_proportion_stats.R` | FOUND but **ID labels legacy** (not final 2/4 semantics) |
| Broad-class area (2/4/Other) | `K2_K10_broad_class_area.csv`, 类簇占比 CSVs | RESULTS-ONLY |

## 6. Explanatory variables

| Methods step | Evidence | Status |
|--------------|----------|--------|
| Enteric share mean/trend | tifs + model_input columns | RESULTS-ONLY |
| Intensity `I_log1p_*`, structure `R_*` | tifs | RESULTS-ONLY |
| Climate/GPP means & trends | tifs | RESULTS-ONLY; Theil–Sen **待确认** |
| Soil texture PCA | `Texture_PCA_*.csv` loadings match check summary | RESULTS-ONLY (code confirmed absent) |
| Match to 1° grid | logic inside old 07/08 scripts (average resample) | PARTIAL |

## 7. Statistical tests and variable screening

| Methods step | Code / evidence | Status |
|--------------|-----------------|--------|
| MW U + FDR (old 1/3) | `mannwhitney_u_test_cluster1_3.py` + 1_秩和检验 CSVs | OLD-VERSION |
| MW final (2/4) | `new_unionfill_results/秩和检验` (n=1912/1183, 21 sig) | RESULTS-ONLY |
| Spearman hub-drop (old) | `spearman_corr_filter_cluster1_3.py` | OLD-VERSION |
| Final screening | Table_S7 (pairwise RF AUC), Table_S8 (forward RF CV AUC), Table_S9 (11 vars) | RESULTS-ONLY |
| Univariate RF AUC | embedded in Table_S7 columns `RF CV AUC1/2` | RESULTS-ONLY |

## 8. Random Forest and SHAP

| Methods step | Code / evidence | Status |
|--------------|-----------------|--------|
| RF/SHAP old 1/3 | `random_forest_shap_cluster1_3.py` (3390×13) | OLD-VERSION |
| RF final 2/4 | `RF_model_performance.xlsx` (3095×11; AUC 0.955) | RESULTS-ONLY |
| SHAP final | `SHAP_feature_importance.xlsx`, summary PDF; MC interventional 64 orderings | RESULTS-ONLY |
| Model input freeze | `new_unionfill_model_input.csv` (3095 rows, 27 factors + labels) | RESULTS-ONLY |

## 9. Figures

| Type | Status |
|------|--------|
| Maps (classification / emissions / clusters) | ArcGIS `.mxd` only — MISSING as code |
| Donut / regression / covariance panels | FOUND (embedded in 03–05 scripts) |
| Final RF/SHAP panels | RESULTS-ONLY (exports 2026-07-19) |
| Country top-10 bars | MISSING code |
