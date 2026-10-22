# Reproducibility status (concise)

| Stage | Script(s) | Status |
|-------|-----------|--------|
| 01 Preprocess | `align_and_mask_by_grassland.py` | Recovered logic (clip/resample) |
| 02 Growth rate | — | **Missing** (no original XCH4→GR→ΔGR code) |
| 03 Emissions | `sector_composition_donuts.py` | Recovered formula |
| 04 Covariance | `signed_contribution_grid.py`, `grassland_type_covariance.R`, `livestock_deltaGR_regression.R` | Recovered formulas |
| 05 Clustering | `kmeans_response_types.py` | Reconstructed; DBI/labels verified |
| 06 Explanatory | `soil_texture_pca.py`, `theilsen_trends.py` | Reconstructed; verified vs tables/trends |
| 07 Statistics | `mannwhitney_pos_vs_neg.py` | Reconstructed; p-values verified |
| 08 RF/SHAP | `rf_shap_final_mc64.py` | Reconstructed 11-var RF; SHAP MC-64 (not bit-identical) |

Still absent as original code: GR/ΔGR construction, ΔE build from 2004,
1° aggregation of SC, full explanatory-factor construction from raw ERA5/GPP,
and the paper’s forward RF-AUC screening tables (Table S7–S8).

Cluster **1/3** scripts from earlier drafts are **not** included and must not
be used for the final analysis (final frame is cluster **2/4**).
