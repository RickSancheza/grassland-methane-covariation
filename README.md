# Grassland Methane Analysis — Publication Code Archive

Reproducible analysis code inventory for the GCB manuscript on global grassland
CH4 emissions, atmospheric CH4 growth rate (GR / ΔGR), sectoral covariance
contributions, response-type clustering, and Random Forest / SHAP interpretation.

**Archive root:** `F:\DOCTOR\graduating\test2\40\_publication_code`  
**Source project:** `F:\DOCTOR\graduating\test2\` (analysis results under `40\`)  
**Organized:** content-preserving copies only. Originals were not modified, moved,
or deleted. No paper results were recalculated or overwritten.

## What this archive is (and is not)

This archive contains **scripts found on disk** after a recursive content-based
scan of `.py`, `.txt`, `.R`, `.r`, and `.ipynb` files (many analysis programs were
saved as `.txt` and are converted here to `.py` / `.R` with identical bytes and
recorded SHA256). It is **not** yet a fully closed, one-command reproducible
pipeline: several manuscript-critical steps have **no original source code** on
disk (see `MISSING_SCRIPTS.md`), and some surviving scripts implement an
**older labeling scheme** that does not match the final union-fill results
(see `VERSION_CONFLICTS.md`).

## Stage layout (paper calculation order)

| Folder | Stage | Content status |
|--------|-------|----------------|
| `00_inventory/` | Inventory, notes, hashes | Complete |
| `01_grassland_classification_preprocessing/` | Grassland mask / 0.1° clip of emissions | Partial |
| `02_ch4_growth_rate/` | XCH4 annual mean, GR, ΔGR (2004 baseline) | **Empty — code missing** |
| `03_emission_change_sector/` | Sectoral emission composition (donuts) | Partial |
| `04_regression_covariance/` | ΔEnteric–ΔGR regression; covariance / SC | Main scripts present |
| `05_response_clustering/` | 1° aggregation, K-means, DBI, K=6 types | Partial (stats only) |
| `06_explanatory_variables/` | Share/intensity/climate trends, soil PCA | **Empty — code missing** |
| `07_statistical_tests_screening/` | Mann–Whitney, Spearman, screening | Old cluster 1/3 versions |
| `08_random_forest_shap/` | RF + SHAP | Old cluster 1/3 version |
| `09_figures/` | Main and supplementary figures | Embedded / ArcGIS only |
| `90_validation/` | Audit and recalculation checks | Audit scripts only |
| `99_legacy/` | Duplicates, diagnostics, non-final donuts | Present |

## Data requirements (not redistributed)

Scripts assume local raster stacks that are **not** included in this archive
(large GeoTIFFs, original mission products). Typical inputs used by the found
scripts:

- `0_data/1_Delta_GR/YYYY.tif` — annual ΔGR (ppb yr⁻¹ relative to 2004)
- `0_data/2_排放/<Sector>/YYYY.tif` — six sector emissions
- `0_data/3_change/<Sector>/YYYY.tif` — six sector emission changes vs 2004
- `0_data/4_均值/`, `0_data/5_趋势/`, `0_data/6_soil/` — means, trends, soil
- `0_data/grasslandtype.tif`, `grassland.tif` — WWF-derived grassland types 1–6
- `3_thirdpart/3_解释/0_解释变量/*.tif` — 0.1° explanatory variables
- Cluster rasters under `3_thirdpart/2_聚类/` (K6, 1° aggregation, etc.)

External source products referenced by methods/notes (code to build them is
missing): WWF terrestrial ecoregions, JRC grassland mask, XCH₄, ERA5, MODIS GPP,
SoilGrids.

## How to run (once data paths are set)

All found scripts use **absolute Windows paths** (`F:\...` or `G:\...`). Before
running, edit the path block at the top of each script or set a project root
variable. Do **not** treat those absolute paths as the public data locator.

Suggested order for the steps that have code (see `RUN_ORDER.md` for evidence):

```text
01  resample/clip emissions to grasslandtype
03  plot_donut_groups_for_folder1_folder2.py
04  signed_contribution_grid.py
    covariance_by_grassland_type.R
    linear_relationship_deltaGR_livestock.R
05  [missing: 1° aggregation + K-means/DBI] then cluster_proportion_stats.R
07  mannwhitney_u_test_cluster1_3.py
    spearman_corr_filter_cluster1_3.py
08  random_forest_shap_cluster1_3.py   # OLD labels only
90  audit_* scripts (validation, not paper pipeline)
```

Python dependencies: see `requirements.txt`.  
R dependencies: `terra`, `dplyr`, `stringr`, `ggplot2`, `readr`, `tidyr`, optional `showtext`.

## Provenance and checksums

- `00_inventory/CODE_INVENTORY.csv` — every candidate code/note file
- `00_inventory/SHA256SUMS.csv` — source path, destination, SHA256, identity check
- Converted `.txt` → `.py`/`.R` files are byte-identical to the originals
  (`content_identical_to_source=YES` in the hash table)

## Publication readiness (GitHub / Zenodo)

**Not ready as a complete public reproducibility package yet.** Remaining blockers:

1. Missing scripts for stages 02, 05 (clustering core), 06, parts of 03/09.
2. Final paper results use **union-fill K=6 cluster 2/4**; found analysis scripts
   use **cluster 1/3** and older cluster raster paths.
3. Hard-coded local absolute paths (`F:\`, `G:\`) must be replaced by relative
   `data/` layout before release.
4. Audit scripts must stay clearly separated from manuscript methods code
   (they are already under `90_validation/`).
5. Clarify and freeze the 40% enteric-share / grassland-fraction filter definition.

Do not push raw rasters or personal paths. See `.gitignore`.
