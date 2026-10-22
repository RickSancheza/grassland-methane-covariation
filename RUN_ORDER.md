# Run Order (Round 2)

Dependency order from **inputs/outputs and result timestamps**, not folder names.
Status marks: FOUND / PARTIAL / RESULTS-ONLY / MISSING / OLD / AUDIT.

## Stage 01 — Classification & preprocessing

| # | Step | Status |
|---|------|--------|
| 1.1 | WWF → grassland type 1–6 | RESULTS-ONLY |
| 1.2 | JRC grassland mask | RESULTS-ONLY |
| 1.3 | Multi-source 0.1° harmonization | RESULTS-ONLY (analysis clip/resample scripts exist for emissions×grassland only) |
| 1.4 | `clip_emission_by_grasslandtype_mask.py` | FOUND |
| 1.5 | `resample_grassland_0p1_and_reclip_emissions.py` (folder1) | FOUND |
| 1.6 | `resample_grassland_0p1_and_reclip_emissions_folder2.py` (folder2) | FOUND |
| 1.7 | Six-sector annual emission build | RESULTS-ONLY |
| 1.8 | Enteric share / 40% filter | PARTIAL (legacy donut diagnostics) |

## Stage 02 — CH4 growth rate

| # | Step | Status |
|---|------|--------|
| 2.1–2.3 | XCH4 mean → GR → ΔGR(2004) | RESULTS-ONLY (`无比例/ch4`, `1_Delta_GR`) |

## Stage 03 — Emissions & composition

| # | Step | Status |
|---|------|--------|
| 3.1 | ΔE vs 2004 | RESULTS-ONLY |
| 3.2 | `plot_donut_groups_for_folder1_folder2.py` | FOUND |
| 3.3 | Share / 40% paper tables | MISSING |

## Stage 04 — Regression & covariance (code-backed)

| # | Step | Status |
|---|------|--------|
| 4.1 | `signed_contribution_grid.py` → `signed_contrib_*.tif` | FOUND — **required before 05** |
| 4.2 | `covariance_by_grassland_type.R` | FOUND |
| 4.3 | `linear_relationship_deltaGR_livestock.R` | FOUND (fix ΔE folder name) |

## Stage 05 — Clustering

| # | Step | Status |
|---|------|--------|
| 5.1 | Aggregate SC to 1° | RESULTS-ONLY |
| 5.2 | K-means K=2–10 + DBI | RESULTS-ONLY (method table + DBI xlsx) |
| 5.3 | Select K=6 (narrative) | RESULTS-ONLY (txt) |
| 5.4 | **union-fill / final `K6_clusters.tif`** | RESULTS-ONLY — **must precede 07–08 final** |
| 5.5 | `cluster_proportion_stats.R` | FOUND (legacy ID names) |
| 5.6 | Broad-class area stats | RESULTS-ONLY |

**Note:** `22/3_julei` is a parallel **4SC / K=8** earlier line; do not run in this order.

## Stage 06 — Explanatory variables

| # | Step | Status |
|---|------|--------|
| 6.1–6.4 | Share/intensity/structure, env means/trends, soil PCA | RESULTS-ONLY |
| 6.5 | Match factors to 1° cluster grid | PARTIAL (inside 07/08 old scripts) |

## Stage 07 — Tests & screening

| # | Step | Status |
|---|------|--------|
| 7.1 | MW old cluster 1/3 | OLD (`mannwhitney_u_test_cluster1_3.py`) |
| 7.2 | Spearman hub-drop old | OLD (`spearman_corr_filter_cluster1_3.py`) |
| 7.3 | **Final MW cluster 2/4** | RESULTS-ONLY (`new_unionfill_results/秩和检验`) |
| 7.4 | **Final high-corr + RF-AUC screening** | RESULTS-ONLY (Table S7–S9) |

## Stage 08 — RF & SHAP

| # | Step | Status |
|---|------|--------|
| 8.1 | RF/SHAP old 1/3 | OLD |
| 8.2 | **Final RF + MC interventional SHAP (2/4)** | RESULTS-ONLY (`new_unionfill_results/RF_SHAP`) |

## Stage 09 — Figures

| # | Step | Status |
|---|------|--------|
| 9.1 | Map figures | ArcGIS only |
| 9.2 | Analytic panels | FOUND where scripts exist (03–05) |
| 9.3 | Final RF/SHAP panel exports | RESULTS-ONLY (7/19) |
| 9.4 | Country bars | MISSING |

## Stage 90 — Validation (not Methods)

1. `audit_test2_results_and_logic.py`
2. `data_locator_and_second_round_audit.py`
3. `thirdpart_recalculation_audit.py`

## Stage 99 — Legacy / other lines

- Donut diagnostics & near-duplicate donut script
- **`22\` 4-sector / 4SC / K=8 alternative** (results only)

---

## Practical execution if reconstructing

Only 01 (partial), 03 donuts, 04, 05 stats chart, and OLD 07–08 can be re-run from this archive.  
Everything else requires new reconstruction code plus frozen inputs (`new_unionfill_model_input.csv` can freeze the final factor table).
