"""Acceptance tests for Task 2 — model service."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot import config
from pilot.model_service import ModelServiceError, predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS: list[str] = json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8"))
FEATURE_MEDIANS: dict[str, float] = json.loads(
    config.FEATURE_MEDIANS_PATH.read_text(encoding="utf-8")
)

# Fixed regression input: two rows, explicit overrides, NaNs to exercise imputation.
_REGRESSION_ROW_0 = {col: FEATURE_MEDIANS[col] for col in FEATURE_COLS}
_REGRESSION_ROW_0.update(
    {
        "geochem_Li_median": 50.0,
        "geochem_Li_max": 60.0,
        "dominant_lith": "lacustrine",
        "closed_basin_indicator": 1,
        "bouguer_mean": -220.0,
        "dist_to_lacustrine_m": 500.0,
        "geochem_sample_count": float("nan"),
        "fault_count": float("nan"),
    }
)
_REGRESSION_ROW_1 = {col: FEATURE_MEDIANS[col] for col in FEATURE_COLS}
_REGRESSION_ROW_1.update(
    {
        "geochem_Li_median": 10.0,
        "geochem_Li_max": 15.0,
        "dominant_lith": "rhyolite",
        "closed_basin_indicator": 0,
        "bouguer_mean": -180.0,
        "dist_to_lacustrine_m": 50000.0,
    }
)
REGRESSION_FEATURES_DF = pd.DataFrame([_REGRESSION_ROW_0, _REGRESSION_ROW_1])[FEATURE_COLS]

# Pinned against models/model.pkl on the provided artifact set.
EXPECTED_REGRESSION_PROBS = np.array([0.4005, 0.0])


def test_regression_sample_yields_fixed_probabilities() -> None:
    probs = predict(REGRESSION_FEATURES_DF)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2,)
    np.testing.assert_allclose(probs, EXPECTED_REGRESSION_PROBS, rtol=0, atol=1e-12)

    # Deterministic across repeated calls in the same process.
    again = predict(REGRESSION_FEATURES_DF)
    np.testing.assert_allclose(again, probs, rtol=0, atol=1e-12)


def test_missing_model_pkl_raises_error_naming_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "model.pkl"
    assert not missing.is_file()
    monkeypatch.setattr(config, "MODEL_PATH", missing)
    # Force reload of cached artifacts if the implementation caches them.
    from pilot import model_service

    model_service._load_artifacts.cache_clear()

    with pytest.raises(ModelServiceError, match=r"model\.pkl"):
        predict(REGRESSION_FEATURES_DF)

    model_service._load_artifacts.cache_clear()


def test_missing_required_feature_column_raises_error_naming_columns() -> None:
    df = REGRESSION_FEATURES_DF.drop(columns=["geochem_Li_median", "fault_count"])
    with pytest.raises(ModelServiceError, match=r"geochem_Li_median"):
        predict(df)
    with pytest.raises(ModelServiceError, match=r"fault_count"):
        predict(df)


def test_extra_column_raises_specific_error() -> None:
    df = REGRESSION_FEATURES_DF.copy()
    df["unexpected_extra"] = 1.0
    with pytest.raises(ModelServiceError, match=r"unexpected_extra"):
        predict(df)


def test_wrong_column_order_raises_specific_error() -> None:
    reversed_cols = list(reversed(FEATURE_COLS))
    df = REGRESSION_FEATURES_DF[reversed_cols]
    with pytest.raises(ModelServiceError, match=r"order|column"):
        predict(df)
