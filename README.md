# Grassland CH4 Analysis — GitHub Release

Reproducible analysis code for the global grassland methane study (GCB),
organized by calculation stage. This package contains **analysis scripts and
configuration only**. Large GeoTIFF inputs are not redistributed.

## Important provenance note

Scripts fall into three categories (see `REPRODUCIBILITY.md`):

1. **Recovered analysis logic** — formulas and parameters taken from project
   source recovered on disk (English rewrite + relative paths; math unchanged).
2. **Reconstructed (verified)** — missing original programs rebuilt from
   method tables and result fingerprints, then checked against published
   numbers (K-means, Mann–Whitney, soil PCA, Theil–Sen, RF/SHAP).
3. **Intentionally absent** — steps with no original code and no verified
   rebuild (e.g. XCH₄ → GR → ΔGR). They are documented, not invented.

Reconstructed SHAP is **not** a bit-for-bit reproduction of the original
mean |SHAP| values.

## Layout

```text
config.py                 # paths and verified parameters
raster_utils.py           # shared raster I/O helpers
01_preprocess/            # grassland mask + grid alignment
02_growth_rate/           # missing (documented)
03_emissions/             # sector composition donuts
04_covariance/            # SC grid + grassland-type covariance + regression
05_clustering/            # K-means response types (verified parameters)
06_explanatory/           # soil texture PCA + Theil-Sen trends
07_statistics/            # Mann-Whitney U + FDR (cluster 2 vs 4)
08_rf_shap/               # final 11-variable RF + MC-64 SHAP (reconstructed)
```

## Environment

Python 3.10+ (see `requirements.txt`) and R 4.x with `terra`, `dplyr`,
`stringr`, `ggplot2`, `readr`.

```bash
export GRASSLAND_CH4_DATA_ROOT=/path/to/data
export GRASSLAND_CH4_OUTPUT_ROOT=/path/to/results
python -m compileall -q .
```

Edit `config.py` if your data folder names differ from the defaults
(`delta_gr`, `emissions`, `emission_changes`, `grasslandtype.tif`, …).

## Suggested run order

```bash
python 01_preprocess/align_and_mask_by_grassland.py
python 03_emissions/sector_composition_donuts.py
python 04_covariance/signed_contribution_grid.py
Rscript 04_covariance/grassland_type_covariance.R
Rscript 04_covariance/livestock_deltaGR_regression.R
python 05_clustering/kmeans_response_types.py
python 06_explanatory/soil_texture_pca.py
python 06_explanatory/theilsen_trends.py
python 07_statistics/mannwhitney_pos_vs_neg.py
python 08_rf_shap/rf_shap_final_mc64.py
```

Provide a frozen model table at `DATA_ROOT/model_input_final.csv`
(3095 cells; columns `lon, lat, cluster, response_binary` + factor columns)
for stages 07–08.

## Verified parameters (do not alter)

| Step | Parameters |
|------|------------|
| SC | `SC_k = cov(dE_k,dGR)/sum\|cov\|*100`; min 5 valid years; ton→Tg ÷1e6 |
| K-means | z-score (pop. SD), missing→0, k-means++/Lloyd, rs=42, n_init=50, max_iter=500 |
| Labels | Enteric-positive = cluster **2**, Enteric-negative = cluster **4** |
| Trends | Theil–Sen slope on year index |
| PCA | Sand/Silt/Clay complete cases, z-score, SVD |
| RF | 11 predictors; 1000 trees; min_leaf=3; min_split=6; max_features=sqrt; rs=42 |
| SHAP | MC interventional permutation; 64 orderings; train-median baseline; class-1 probability |

## License

Add a `LICENSE` file when you choose one (e.g. MIT, CC-BY-4.0 for docs).
No license is bundled in this package.
