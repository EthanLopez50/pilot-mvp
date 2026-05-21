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
from pilot.explain import (
    FEATURE_LABELS,
    feature_mapping_keys,
    rationale,
    top_shap_features,
    _format_rationale_part,
    _format_feature_value,
)
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
_BULLET_RE = re.compile(r"^- (.+)$", re.MULTILINE)
_NEAR_PROXIMITY_PHRASES = (
    "close to",
    "sits on or near",
    "within regional distance",
    "near a mapped",
    "intermediate distance",
    "the mapped feature",
)
_DRILL_LANGUAGE_RE = re.compile(
    r"drill[- ]?site|drill[- ]?ready|recommend(?:ed)? drilling",
    re.IGNORECASE,
)
_GEOCHEM_PPM_RE = re.compile(r"\d+\.\d ppm")
_RAW_FEATURE_NAMES = frozenset(FEATURE_COLS)


def _bullets(rationale_text: str) -> list[str]:
    return _BULLET_RE.findall(rationale_text)


def _bullet_for_label(text: str, label: str) -> str:
    for bullet in _bullets(text):
        if bullet.startswith(f"{label} ("):
            return bullet
    raise AssertionError(f"Label {label!r} not found in rationale bullets: {text!r}")


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
            "- maximum stream-sediment lithium (60.0 ppm): elevated lithium in "
            "nearby stream-sediment geochemistry\n"
            "- lacustrine sediment coverage (0.0%): little to no lacustrine "
            "sediments mapped in the cell. Lacustrine sediments are the host "
            "setting for brine-type lithium.\n"
            "- distance to hotspring (21.7 km): distant from thermal springs"
        ),
        "reg-cell-1": (
            "- maximum stream-sediment lithium (15.0 ppm): depressed lithium in "
            "nearby stream-sediment geochemistry\n"
            "- distance to hotspring (21.7 km): distant from thermal springs\n"
            "- Hotsprings Temp Max (35.0 C): near elevated geothermal "
            "temperatures, a lithium-mobilizing heat source"
        ),
    }
    assert "geochem_Li_median" not in first["reg-cell-0"]


def test_geochem_values_use_fixed_one_decimal_ppm() -> None:
    assert _format_feature_value("geochem_Li_max", 60) == "60.0 ppm"
    assert _format_feature_value("geochem_Li_median", 214.2) == "214.2 ppm"


def test_rationales_never_use_drill_site_or_drill_ready_language(
    sample_ranked: tuple[pd.DataFrame, np.ndarray],
) -> None:
    features_df, probabilities = sample_ranked
    rationales = rationale(features_df, probabilities, top_n=config.TOP_N)

    for cell_id, text in rationales.items():
        assert not _DRILL_LANGUAGE_RE.search(text), (
            f"Drill-site/drill-ready wording in rationale for {cell_id}: {text!r}"
        )


def test_aoi_percentile_context_when_enough_cells() -> None:
    """Extreme values in a multi-cell AOI earn decile context from real column data."""
    base = {col: FEATURE_MEDIANS[col] for col in FEATURE_COLS}
    rows = []
    for idx in range(12):
        row = dict(base)
        row["geochem_Li_max"] = float(idx * 10)
        row["dominant_lith"] = "lacustrine"
        row["cell_id"] = f"pct-cell-{idx}"
        rows.append(row)
    features_df = pd.DataFrame(rows)[["cell_id"] + FEATURE_COLS]
    probabilities = np.linspace(0.9, 0.1, len(rows))
    rationales = rationale(features_df, probabilities, top_n=1)

    top_cell = features_df.loc[features_df["cell_id"] == next(iter(rationales))]
    text = next(iter(rationales.values()))
    assert "highest decile in this AOI" in text or "lowest decile in this AOI" in text
    assert _DRILL_LANGUAGE_RE.search(text) is None


def test_rendered_rationale_uses_readable_labels_not_raw_names(
    sample_ranked: tuple[pd.DataFrame, np.ndarray],
) -> None:
    features_df, probabilities = sample_ranked
    rationales = rationale(features_df, probabilities, top_n=config.TOP_N)

    for cell_id, text in rationales.items():
        for raw_name in _RAW_FEATURE_NAMES:
            assert raw_name not in text, (
                f"Raw feature name {raw_name!r} in rationale for {cell_id}: {text!r}"
            )
        assert text.startswith("- "), f"Rationale must use bullet lines: {text!r}"
        assert "\n- " in text or text.count("\n") == 0, (
            f"Expected newline-separated bullets for {cell_id}"
        )


def test_rendered_rationale_includes_units(
    sample_ranked: tuple[pd.DataFrame, np.ndarray],
) -> None:
    features_df, probabilities = sample_ranked
    rationales = rationale(features_df, probabilities, top_n=config.TOP_N)

    unit_pattern = re.compile(
        r"\b(\d[\d,]*(?:\.\d+)?\s*(?:ppm|%|km|m)|\d+\.\d ppm|\d+\.\d%)\b"
    )
    for cell_id, text in rationales.items():
        assert unit_pattern.search(text), (
            f"No unit-bearing values in rationale for {cell_id}: {text!r}"
        )
        if "ppm" in text:
            assert _GEOCHEM_PPM_RE.search(text), (
                f"Geochemistry should use one decimal before ppm for {cell_id}: {text!r}"
            )


def test_pct_lacustrine_directional_phrasing_high_vs_marginal() -> None:
    high = _format_rationale_part("pct_lacustrine", 0.92)
    marginal_45 = _format_rationale_part("pct_lacustrine", 0.456)
    marginal_29 = _format_rationale_part("pct_lacustrine", 0.293)

    assert high is not None
    assert marginal_45 is not None
    assert marginal_29 is not None

    assert "dominated by lacustrine sediments" in high
    assert "dominated by" not in marginal_45
    assert "substantial lacustrine sediments cover" in marginal_45
    assert "dominated by" not in marginal_29
    assert "substantial lacustrine sediments cover" in marginal_29


def test_feature_labels_dict_covers_required_mappings() -> None:
    required = {
        "geochem_Li_max": "maximum stream-sediment lithium",
        "geochem_Rb_median": "median stream-sediment rubidium",
        "geochem_As_median": "median stream-sediment arsenic",
        "pct_lacustrine": "lacustrine sediment coverage",
        "dist_to_caldera_m": "distance to nearest caldera margin",
        "dist_to_geothermal_m": "distance to mapped geothermal field",
        "dist_to_geochem_anom_m": "distance to nearest geochemical anomaly",
        "elev_min": "minimum elevation",
    }
    for feature, label in required.items():
        assert FEATURE_LABELS[feature] == label


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
        bullets = _bullets(text)
        assert bullets, f"No bullets parsed from rationale for {cell_id}"
        top_pool = top_shap_features(features_df, probabilities, cell_id, n=10)
        labels_in_text = [
            FEATURE_LABELS.get(f, f.replace("_", " ").title())
            for f in top_pool
        ]
        assert any(
            bullet.startswith(f"{label} (") for label in labels_in_text for bullet in bullets
        ), (
            f"Rationale for {cell_id} does not reference a top-|SHAP| feature label "
            f"from {top_pool}: {text!r}"
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
        top_pool = top_shap_features(features_df, probabilities, cell_id, n=10)
        for feature in top_pool:
            if feature not in _DISTANCE_FEATURES:
                continue
            label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
            if not any(bullet.startswith(f"{label} (") for bullet in _bullets(text)):
                continue
            distance = float(row[feature])
            median = FEATURE_MEDIANS[feature]
            clause = _bullet_for_label(text, label)

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
