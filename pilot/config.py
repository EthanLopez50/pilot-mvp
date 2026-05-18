"""Paths and constants for the pilot pipeline. No logic."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_cols.json"
FEATURE_MEDIANS_PATH = PROJECT_ROOT / "models" / "feature_medians.json"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
GRID_PATH = PROJECT_ROOT / "data" / "grid" / "features_v2.gpkg"
AOI_DIR = PROJECT_ROOT / "data" / "aoi"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# CRS of the precomputed Nevada feature grid (reproject AOIs to this).
GRID_CRS = "EPSG:3857"

# Spatial buffer around the AOI when extracting grid cells (kilometres).
BUFFER_KM = 10

# Fixed seed for any sampling in explanations or plots.
RANDOM_SEED = 42

# Number of top-ranked targets in the report and explanation step.
TOP_N = 10

# Minimum fraction of AOI area covered by the grid to treat coverage as adequate.
COVERAGE_MIN_FRACTION = 0.8
