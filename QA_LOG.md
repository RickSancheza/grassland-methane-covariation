# Release QA log (2026-10-22)

## Checks performed

| Check | Result |
|-------|--------|
| `python -m compileall` on all `.py` | PASS |
| `Rscript parse()` on both `.R` | PASS |
| Hardcoded `F:/` `G:/` `DOCTOR\` in sources | **None** |
| Cluster 1/3 as final analysis code | **None** |
| 9-variable RF model script | **None** (only 11-feature `FINAL_RF_FEATURES`) |
| CJK in Python/R sources | **None** |
| Config via `os.environ` / `PROJECT_ROOT` | PASS |
| Smoke: SC formula sum\|SC\|≈100 | PASS |
| Smoke: KMeans params importable | PASS |

## Full data end-to-end runs

Not executed in this pass (no large data mounted under `DATA_ROOT`).  
Logic-level verification against paper results was done in
`_publication_code/reconstructed/` (K-means, MW, PCA, Theil–Sen, RF/SHAP).

## Intentionally omitted from this package

- Legacy Cluster 1/3 scripts
- Audit / recalculation scripts
- `verify_outputs`, method reports, Chinese plot diagnostics
- Large TIFF inputs
