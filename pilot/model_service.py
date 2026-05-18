"""Load the frozen model and expose predict()."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pilot import config


class ModelServiceError(ValueError):
    """Raised when model artifacts are missing or features are invalid."""


@dataclass(frozen=True)
class _Artifacts:
    model: object
    label_encoder: LabelEncoder
    feature_cols: tuple[str, ...]
    feature_medians: dict[str, float]


@lru_cache(maxsize=1)
def _load_artifacts() -> _Artifacts:
    """Load model artifacts from disk (cached)."""
    _require_file(config.MODEL_PATH)
    _require_file(config.FEATURE_COLS_PATH)
    _require_file(config.FEATURE_MEDIANS_PATH)
    _require_file(config.LABEL_ENCODER_PATH)

    feature_cols = tuple(
        json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8"))
    )
    feature_medians = json.loads(
        config.FEATURE_MEDIANS_PATH.read_text(encoding="utf-8")
    )
    model = joblib.load(config.MODEL_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)

    return _Artifacts(
        model=model,
        label_encoder=label_encoder,
        feature_cols=feature_cols,
        feature_medians=feature_medians,
    )


def predict(features_df: pd.DataFrame) -> np.ndarray:
    """Return a prospectivity probability for each feature row.

    Args:
        features_df: DataFrame whose columns exactly match
            ``models/feature_cols.json`` in name and order. Missing numeric
            values are filled from ``models/feature_medians.json``.
            ``dominant_lith`` may be a lithology label string; it is encoded
            with the saved label encoder before inference.

    Returns:
        One-dimensional array of probabilities, one per row.

    Raises:
        ModelServiceError: If a model artifact file is missing, columns do not
            match the expected feature list, or ``dominant_lith`` cannot be
            encoded.
    """
    artifacts = _load_artifacts()
    _validate_feature_columns(features_df, artifacts.feature_cols)

    prepared = features_df.copy()
    for col in artifacts.feature_cols:
        if col == "dominant_lith":
            continue
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(
            artifacts.feature_medians[col]
        )

    try:
        prepared["dominant_lith"] = artifacts.label_encoder.transform(
            prepared["dominant_lith"].astype(str)
        )
    except ValueError as exc:
        raise ModelServiceError(
            f"Unknown dominant_lith value(s) in column dominant_lith: {exc}"
        ) from exc

    matrix = prepared[list(artifacts.feature_cols)]
    return np.asarray(artifacts.model.predict(matrix), dtype=float)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ModelServiceError(f"Required model artifact not found: {path.name}")


def _validate_feature_columns(
    features_df: pd.DataFrame, expected_cols: tuple[str, ...]
) -> None:
    actual_cols = list(features_df.columns)
    expected_list = list(expected_cols)

    if actual_cols == expected_list:
        return

    missing = [col for col in expected_list if col not in actual_cols]
    if missing:
        raise ModelServiceError(
            f"Missing feature column(s): {', '.join(missing)}"
        )

    extra = [col for col in actual_cols if col not in expected_list]
    if extra:
        raise ModelServiceError(
            f"Unexpected feature column(s): {', '.join(extra)}"
        )

    raise ModelServiceError(
        "Feature columns do not match the expected order in feature_cols.json"
    )
