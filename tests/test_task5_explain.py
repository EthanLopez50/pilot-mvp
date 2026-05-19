"""Acceptance tests for Task 5 — SHAP explanations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot import config
from pilot.aoi import parse_aoi
from pilot.explain import feature_mapping_keys, rationale, top_shap_features
from pilot.features import extract
from pilot.model_service import predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample.geojson"

FEATURE_COLS: list[str] = json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8"))
FEATURE_MEDIANS: dict[str, float] = json.loads(
    config.FEATURE_MEDIANS_PATH.read_text(encoding="utf-8")
)

# Fixed two-row input (same overrides as Task 2 regression), with stable cell ids.
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
FIXED_EXPLAIN_DF = pd.DataFrame(
    [
        {**_REGRESSION_ROW_0, "cell_id": "reg-cell-0"},
        {**_REGRESSION_ROW_1, "cell_id": "reg-cell-1"},
    ]
)[["cell_id"] + FEATURE_COLS]

_DISTANCE_FEATURES = [col for col in FEATURE_COLS if col.startswith("dist_to_")]
_CITED_FEATURE_RE = re.compile(r"(?:^|; )([\w]+) \(")
_NEAR_PROXIMITY_PHRASES = (
    "close to",
    "sits on or near",
    "within regional distance",
    "near a mapped",
    "intermediate distance",
    "the mapped feature",
)


def _cited_features(rationale_text: str) -> list[str]:
    return _CITED_FEATURE_RE.findall(rationale_text)


def _rationale_clause(text: str, feature: str) -> str:
    for clause in text.split("; "):
        if clause.startswith(f"{feature} ("):
            return clause
    raise AssertionError(f"Feature {feature!r} not cited in rationale: {text!r}")


def _fixed_probabilities() -> np.ndarray:
    return predict(FIXED_EXPLAIN_DF[FEATURE_COLS])


@pytest.fixture(scope="module")
def sample_ranked() -> tuple[pd.DataFrame, np.ndarray]:
    aoi = parse_aoi(SAMPLE_GEOJSON)
    features_df, coverage = extract(aoi)
    assert coverage.adequate
    probabilities = predict(features_df[FEATURE_COLS])
    return features_df, probabilities


def test_every_feature_in_feature_cols_has_mapping_entry() -> None:
    mapped = feature_mapping_keys()
    missing = [col for col in FEATURE_COLS if col not in mapped]
    extra = [col for col in mapped if col not in FEATURE_COLS]
    assert not missing, f"Features missing geological mapping: {missing}"
    assert not extra, f"Mapping entries not in feature_cols.json: {extra}"


def test_fixed_input_yields_fixed_rationale_text() -> None:
    probs = _fixed_probabilities()
    first = rationale(FIXED_EXPLAIN_DF, probs)
    second = rationale(FIXED_EXPLAIN_DF, probs)
    assert first == second
    assert first == {
        "reg-cell-0": (
            "geochem_Li_max (60): elevated lithium in nearby stream-sediment "
            "geochemistry; pct_lacustrine (0.0%): little to no lacustrine sediments "
            "mapped in the cell. Lacustrine sediments are the host setting for "
            "brine-type lithium.; dist_to_hotspring_m (21740 m): distant from "
            "thermal springs"
        ),
        "reg-cell-1": (
            "geochem_Li_max (15): depressed lithium in nearby stream-sediment "
            "geochemistry; dist_to_hotspring_m (21740 m): distant from thermal "
            "springs; hotsprings_temp_max (35.0 C): near elevated geothermal "
            "temperatures, a lithium-mobilizing heat source"
        ),
    }
    assert "geochem_Li_median" not in first["reg-cell-0"]


def test_every_top_n_cell_has_nonempty_rationale(sample_ranked: tuple[pd.DataFrame, np.ndarray]) -> None:
    features_df, probabilities = sample_ranked
    rationales = rationale(features_df, probabilities, top_n=config.TOP_N)

    assert len(rationales) == config.TOP_N
    for cell_id, text in rationales.items():
        assert cell_id
        assert isinstance(text, str)
        assert text.strip(), f"Empty rationale for cell {cell_id}"


def test_each_rationale_references_actual_top_shap_features(
    sample_ranked: tuple[pd.DataFrame, np.ndarray],
) -> None:
    features_df, probabilities = sample_ranked
    rationales = rationale(features_df, probabilities, top_n=config.TOP_N)

    for cell_id, text in rationales.items():
        cited = _cited_features(text)
        assert cited, f"No cited features parsed from rationale for {cell_id}"
        top_pool = top_shap_features(features_df, probabilities, cell_id, n=10)
        assert any(feature in top_pool for feature in cited), (
            f"Rationale for {cell_id} cites {cited} but none appear in the top "
            f"|SHAP| pool {top_pool}: {text!r}"
        )


def test_rationales_never_use_generic_mapped_feature_wording(
    sample_ranked: tuple[pd.DataFrame, np.ndarray],
) -> None:
    _, probabilities = sample_ranked
    features_df, _ = sample_ranked
    rationales = rationale(features_df, probabilities, top_n=config.TOP_N)

    for cell_id, text in rationales.items():
        assert "the mapped feature" not in text, (
            f"Generic distance wording in rationale for {cell_id}: {text!r}"
        )
        assert "intermediate distance" not in text.lower(), (
            f"Intermediate distance wording in rationale for {cell_id}: {text!r}"
        )


def test_distance_feature_phrasing_reflects_proximity_value(
    sample_ranked: tuple[pd.DataFrame, np.ndarray],
) -> None:
    features_df, probabilities = sample_ranked
    rationales = rationale(features_df, probabilities, top_n=config.TOP_N)

    for cell_id, text in rationales.items():
        row = features_df.loc[
            features_df["cell_id"].astype(str) == str(cell_id)
        ].iloc[0]
        for feature in _cited_features(text):
            if feature not in _DISTANCE_FEATURES:
                continue
            distance = float(row[feature])
            median = FEATURE_MEDIANS[feature]
            clause = _rationale_clause(text, feature)

            if distance <= median * 0.5:
                assert any(
                    phrase in clause.lower() for phrase in _NEAR_PROXIMITY_PHRASES[:4]
                ) or "sits on or near" in clause.lower(), (
                    f"Near distance {feature}={distance} should use proximity-favorable "
                    f"phrasing: {clause!r}"
                )
            else:
                assert "distant from" in clause.lower() or "remote from" in clause.lower(), (
                    f"Non-near distance {feature}={distance} (median {median}) should "
                    f"use absence/distant phrasing, not favorable proximity: {clause!r}"
                )
                for phrase in _NEAR_PROXIMITY_PHRASES:
                    assert phrase not in clause.lower(), (
                        f"Favorable proximity phrase {phrase!r} in non-near clause: "
                        f"{clause!r}"
                    )
