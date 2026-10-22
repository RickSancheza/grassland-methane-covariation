# Central path configuration for the grassland CH4 analysis release.
# Override the data root with the environment variable GRASSLAND_CH4_DATA_ROOT
# or by editing DATA_ROOT below. No drive letters are hard-coded.

from __future__ import annotations

import os
from pathlib import Path

# Repository root (this file's directory)
PROJECT_ROOT = Path(__file__).resolve().parent

# Root of input data (GeoTIFF stacks, tables). Default: PROJECT_ROOT/data
DATA_ROOT = Path(os.environ.get("GRASSLAND_CH4_DATA_ROOT", PROJECT_ROOT / "data"))

# Root for analysis outputs. Default: PROJECT_ROOT/results
OUTPUT_ROOT = Path(os.environ.get("GRASSLAND_CH4_OUTPUT_ROOT", PROJECT_ROOT / "results"))

# --- Input layout (relative to DATA_ROOT) ---
# Adjust these subpaths if your local archive uses different names.

DELTA_GR_DIR = DATA_ROOT / "delta_gr"                    # yearly Delta GR 2005-2020
EMISSION_DIR = DATA_ROOT / "emissions"                   # <Sector>/<year>.tif
DELTA_EMISSION_DIR = DATA_ROOT / "emission_changes"      # <Sector>/<year>.tif vs 2004
GRASSLAND_TYPE_TIF = DATA_ROOT / "grasslandtype.tif"     # codes 1-6
ANNUAL_GPP_DIR = DATA_ROOT / "annual_gpp"                # 2005.tif ... 2020.tif
ANNUAL_CLIMATE_DIRS = {
    "MAT": DATA_ROOT / "annual_mat",
    "MAP": DATA_ROOT / "annual_map",
    "SM": DATA_ROOT / "annual_sm",
    "ST": DATA_ROOT / "annual_st",
}
SOIL_DIR = DATA_ROOT / "soil"                            # Sand/Silt/Clay or Texture_PCA_*.csv
SC_GRID_DIR = DATA_ROOT / "signed_contribution"          # signed_contrib_<Sector>.tif
FACTOR_DIR = DATA_ROOT / "explanatory_factors"           # 0.1 deg factor GeoTIFFs
MODEL_INPUT_CSV = DATA_ROOT / "model_input_final.csv"    # frozen 3095-row table (optional)

# --- Output layout (relative to OUTPUT_ROOT) ---
OUT_PREPROCESS = OUTPUT_ROOT / "01_preprocess"
OUT_GROWTH = OUTPUT_ROOT / "02_growth_rate"
OUT_EMISSIONS = OUTPUT_ROOT / "03_emissions"
OUT_COVARIANCE = OUTPUT_ROOT / "04_covariance"
OUT_CLUSTERING = OUTPUT_ROOT / "05_clustering"
OUT_FACTORS = OUTPUT_ROOT / "06_explanatory"
OUT_STATS = OUTPUT_ROOT / "07_statistics"
OUT_RF_SHAP = OUTPUT_ROOT / "08_rf_shap"

# Analysis period
YEARS = list(range(2005, 2021))
BASELINE_YEAR = 2004

# Six CH4 source sectors (folder names under EMISSION_DIR / DELTA_EMISSION_DIR)
SECTORS = ["Livestock", "Agriculture", "Energy", "Industrial", "Waste", "Wetland"]
# Folder aliases when data use Enteric instead of Livestock
SECTOR_ALIASES = {
    "Livestock": ["Livestock", "Enteric"],
    "Agriculture": ["Agriculture"],
    "Energy": ["Energy"],
    "Industrial": ["Industrial"],
    "Waste": ["Waste"],
    "Wetland": ["Wetland"],
}

# WWF-derived grassland eco-type codes (1-6) used throughout
GRASSLAND_CODES = (1, 2, 3, 4, 5, 6)
GRASSLAND_LABELS_EN = {
    1: "tropical_subtropical_grassland",
    2: "temperate_grassland",
    3: "flooded_wet_grassland",
    4: "montane_grassland",
    5: "tundra",
    6: "arid_desert_grassland",
}

# Emission unit conversion: ton CH4 / cell / yr -> Tg CH4 / cell / yr
EMISSION_TO_TG_DIVISOR = 1.0e6

# Covariance / SC parameters (verified against published SC rasters)
MIN_VALID_YEARS = 5
SC_NODATA = -9999.0

# K-means (verified: DBI K=6 = 1.139228, 100% label match)
KMEANS_K = 6
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 500

# Final RF / SHAP (11 predictors; MC-64 reconstructed implementation)
RF_PARAMS = {
    "n_estimators": 1000,
    "max_depth": None,
    "min_samples_leaf": 3,
    "min_samples_split": 6,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}
RF_TEST_SIZE = 0.30
RF_RANDOM_STATE = 42
SHAP_N_ORDERINGS = 64
SHAP_SEED = 42

# Final 11 RF predictors (Table S9 / SHAP_feature_importance.xlsx 2026-07-19)
FINAL_RF_FEATURES = [
    "Enteric_share_trend",
    "ST_mean",
    "I_log1p_Enteric_mean",
    "R_Agriculture_mean",
    "Texture_PC2",
    "MAT_trend",
    "MAP_trend",
    "SM_mean",
    "I_log1p_Industrial_mean",
    "GPP_trend",
    "I_log1p_Wetland_mean",
]

# Cluster labels for Enteric-positive / negative (union-fill K=6)
POSITIVE_CLUSTER = 2
NEGATIVE_CLUSTER = 4

NODATA_VALUES = (-9999.0, -3.4e38, 3.4e38)


def ensure_output_dirs() -> None:
    for p in (
        OUT_PREPROCESS,
        OUT_GROWTH,
        OUT_EMISSIONS,
        OUT_COVARIANCE,
        OUT_CLUSTERING,
        OUT_FACTORS,
        OUT_STATS,
        OUT_RF_SHAP,
    ):
        p.mkdir(parents=True, exist_ok=True)


def resolve_sector_dir(root: Path, sector: str) -> Path:
    """Return the directory for a sector, accepting documented folder aliases."""
    for name in SECTOR_ALIASES.get(sector, [sector]):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Sector folder not found for {sector!r} under {root}")


def year_tif(folder: Path, year: int) -> Path:
    """Locate <year>.tif under folder."""
    matches = sorted(folder.glob(f"*{year}*.tif"))
    if not matches:
        raise FileNotFoundError(f"No GeoTIFF for year {year} in {folder}")
    return matches[0]
