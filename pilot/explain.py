"""SHAP-based per-target geological rationale."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import shap

from pilot import config
from pilot.model_service import _load_artifacts

# Strongest contributors included in each cell's rationale text.
N_TOP_SHAP_FEATURES = 3

FeatureRationaleFn = Callable[[Any], str | None]


def feature_mapping_keys() -> frozenset[str]:
    """Return the set of feature names with a geological rationale mapping."""
    return frozenset(FEATURE_RATIONALE)


def rationale(
    features_df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model: object | None = None,
    top_n: int | None = None,
) -> dict[str, str]:
    """Build plain-language rationales for the top-ranked cells by probability.

    Args:
        features_df: Grid cells with ``cell_id``, model feature columns, and any
            extra columns (for example geometry).
        probabilities: One probability per row, aligned with ``features_df``.
        model: Optional pre-loaded sklearn model; if omitted, the frozen model
            from ``model_service`` is used.
        top_n: Number of highest-probability cells to explain; defaults to
            ``config.TOP_N``.

    Returns:
        ``cell_id`` -> non-empty rationale string referencing that cell's
        strongest SHAP contributors and their values.

    Raises:
        ValueError: If ``cell_id`` is missing, lengths disagree, or no cells
            are available to explain.
    """
    if "cell_id" not in features_df.columns:
        raise ValueError("features_df must include a cell_id column")

    n_rows = len(features_df)
    if len(probabilities) != n_rows:
        raise ValueError(
            f"probabilities length ({len(probabilities)}) does not match "
            f"features_df rows ({n_rows})"
        )
    if n_rows == 0:
        raise ValueError("features_df is empty; nothing to explain")

    limit = top_n if top_n is not None else config.TOP_N
    ranked = _rank_top_cells(features_df, probabilities, limit)
    if ranked.empty:
        raise ValueError("No cells available for explanation after ranking")

    np.random.seed(config.RANDOM_SEED)
    shap_values, feature_cols, raw_values = _shap_for_rows(ranked, model=model)

    rationales: dict[str, str] = {}
    for row_idx, cell_id in enumerate(ranked["cell_id"].astype(str)):
        parts = _collect_rationale_parts(
            shap_values[row_idx],
            feature_cols,
            raw_values[row_idx],
        )
        rationales[cell_id] = "; ".join(parts)

    return rationales


def top_shap_features(
    features_df: pd.DataFrame,
    probabilities: np.ndarray,
    cell_id: str,
    *,
    n: int = N_TOP_SHAP_FEATURES,
    model: object | None = None,
) -> list[str]:
    """Return the top-|SHAP| feature names for one cell.

    Args:
        features_df: Same contract as :func:`rationale`.
        probabilities: Same contract as :func:`rationale`.
        cell_id: Target cell identifier present in ``features_df``.
        n: How many strongest contributors to return.
        model: Optional pre-loaded model; uses the frozen artifact when omitted.

    Returns:
        Feature names ordered by descending absolute SHAP value.

    Raises:
        ValueError: If ``cell_id`` is not found in ``features_df``.
    """
    if "cell_id" not in features_df.columns:
        raise ValueError("features_df must include a cell_id column")

    positions = np.flatnonzero(
        features_df["cell_id"].astype(str).to_numpy() == str(cell_id)
    )
    if len(positions) == 0:
        raise ValueError(f"cell_id not found in features_df: {cell_id}")

    row = features_df.iloc[int(positions[0]) : int(positions[0]) + 1]

    np.random.seed(config.RANDOM_SEED)
    shap_values, feature_cols, _raw = _shap_for_rows(row, model=model)
    top_indices = _top_shap_indices(shap_values[0], n=n)
    return [feature_cols[idx] for idx in top_indices]


def _rank_top_cells(
    features_df: pd.DataFrame,
    probabilities: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    ranked = features_df.copy()
    ranked["_probability"] = probabilities
    ranked = ranked.sort_values("_probability", ascending=False)
    return ranked.head(min(top_n, len(ranked))).drop(columns="_probability")


def _shap_for_rows(
    features_df: pd.DataFrame,
    *,
    model: object | None,
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    artifacts = _load_artifacts()
    estimator = model if model is not None else artifacts.model
    feature_cols = artifacts.feature_cols

    feature_cols_json = json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8"))
    matrix, raw_values = _prepare_model_matrix(features_df, feature_cols_json, artifacts)

    explainer = shap.TreeExplainer(estimator)
    shap_values = np.asarray(explainer.shap_values(matrix), dtype=float)
    if shap_values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape {shap_values.shape}")

    return shap_values, feature_cols, raw_values


def _prepare_model_matrix(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    artifacts: object,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    prepared_rows: list[dict[str, float]] = []
    raw_values: list[dict[str, Any]] = []

    for _, row in features_df.iterrows():
        raw: dict[str, Any] = {}
        numeric_row: dict[str, float] = {}
        for col in feature_cols:
            if col == "dominant_lith":
                raw[col] = row[col]
                numeric_row[col] = float(
                    artifacts.label_encoder.transform([str(row[col])])[0]
                )
                continue
            value = row[col]
            raw[col] = value
            numeric = pd.to_numeric(value, errors="coerce")
            if pd.isna(numeric):
                numeric = artifacts.feature_medians[col]
            numeric_row[col] = float(numeric)
        raw_values.append(raw)
        prepared_rows.append(numeric_row)

    matrix = np.asarray(
        [[prepared[col] for col in feature_cols] for prepared in prepared_rows],
        dtype=float,
    )
    return matrix, raw_values


def _top_shap_indices(shap_row: np.ndarray, *, n: int) -> list[int]:
    order = np.argsort(np.abs(shap_row))[::-1]
    return [int(idx) for idx in order[:n]]


def _collect_rationale_parts(
    shap_row: np.ndarray,
    feature_cols: tuple[str, ...],
    raw_row: dict[str, Any],
) -> list[str]:
    order = np.argsort(np.abs(shap_row))[::-1]
    parts: list[str] = []
    seen_geological: set[str] = set()

    for col_idx in order:
        if len(parts) >= N_TOP_SHAP_FEATURES:
            break
        feature = feature_cols[int(col_idx)]
        part = _format_rationale_part(feature, raw_row[feature])
        if part is None:
            continue
        geological = _geological_sentence(part)
        if geological in seen_geological:
            continue
        seen_geological.add(geological)
        parts.append(part)

    return parts


def _geological_sentence(rationale_part: str) -> str:
    return rationale_part.split("): ", 1)[1]


def _format_rationale_part(feature: str, raw_value: Any) -> str | None:
    if feature not in FEATURE_RATIONALE:
        raise ValueError(f"No geological mapping for feature: {feature}")
    geological = FEATURE_RATIONALE[feature](raw_value)
    if geological is None:
        return None
    return f"{feature} ({_format_feature_value(feature, raw_value)}): {geological}"


def _format_feature_value(feature: str, raw_value: Any) -> str:
    if feature == "dominant_lith":
        return str(raw_value)
    if feature == "closed_basin_indicator":
        return "yes" if int(float(raw_value)) == 1 else "no"
    if feature.startswith("pct_"):
        return f"{float(raw_value) * 100:.1f}%"
    if feature.endswith("_m"):
        return f"{float(raw_value):.0f} m"
    if feature.endswith("_t") or feature == "hotsprings_temp_max":
        return f"{float(raw_value):.1f} C"
    if feature.endswith("_count") or feature.endswith("_count_10km"):
        return f"{float(raw_value):.0f}"
    if pd.isna(raw_value):
        return "missing"
    value = float(raw_value)
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def _geochem_median_text(element: str, value: Any) -> str:
    level = _level_vs_median(float(value), FEATURE_MEDIANS[f"geochem_{element}_median"])
    if level == "high":
        if element == "Li":
            return "elevated lithium in nearby stream-sediment geochemistry"
        return f"elevated {element} in nearby stream-sediment geochemistry"
    if level == "low":
        name = "lithium" if element == "Li" else element
        return f"depressed {name} in nearby stream-sediment geochemistry"
    name = "lithium" if element == "Li" else element
    return f"{name} in nearby stream-sediment geochemistry near typical regional levels"


def _geochem_count_text(element: str, value: Any) -> str:
    count = float(value)
    if count >= 3:
        return f"multiple nearby stream-sediment samples with {element} measurements"
    if count >= 1:
        return f"limited nearby stream-sediment {element} measurements"
    return f"no nearby stream-sediment {element} measurements in the cell"


def _distance_text(
    feature: str,
    value: Any,
    *,
    near: str,
    far: str,
) -> str | None:
    """Proximity features: low distance is favorable; otherwise absence wording."""
    distance = float(value)
    median = FEATURE_MEDIANS[feature]
    if distance <= median * 0.5:
        return near
    return far


def _pct_text(lithology: str, value: Any) -> str:
    pct = float(value) * 100.0
    if pct >= 50.0:
        return f"dominated by {lithology}."
    if pct >= 20.0:
        return f"substantial {lithology} cover in the cell."
    if pct > 0.0:
        return f"minor {lithology} component in the cell."
    return f"little to no {lithology} mapped in the cell."


def _level_vs_median(value: float, median: float) -> str:
    if value >= median * 1.25:
        return "high"
    if value <= median * 0.75:
        return "low"
    return "typical"


FEATURE_MEDIANS: dict[str, float] = json.loads(
    config.FEATURE_MEDIANS_PATH.read_text(encoding="utf-8")
)

FEATURE_RATIONALE: dict[str, FeatureRationaleFn] = {
    "geochem_sample_count": lambda v: (
        "dense stream-sediment sampling nearby"
        if float(v) >= FEATURE_MEDIANS["geochem_sample_count"] * 1.5
        else "sparse stream-sediment sampling nearby"
    ),
    "geochem_Li_median": lambda v: _geochem_median_text("Li", v),
    "geochem_Li_max": lambda v: _geochem_median_text("Li", v),
    "geochem_Li_count": lambda v: _geochem_count_text("lithium", v),
    "geochem_As_median": lambda v: _geochem_median_text("As", v),
    "geochem_As_count": lambda v: _geochem_count_text("arsenic", v),
    "geochem_Ba_median": lambda v: _geochem_median_text("Ba", v),
    "geochem_Ba_count": lambda v: _geochem_count_text("barium", v),
    "geochem_Rb_median": lambda v: _geochem_median_text("Rb", v),
    "geochem_Rb_count": lambda v: _geochem_count_text("rubidium", v),
    "geochem_Cs_median": lambda v: _geochem_median_text("Cs", v),
    "geochem_Cs_count": lambda v: _geochem_count_text("cesium", v),
    "geochem_Mo_median": lambda v: _geochem_median_text("Mo", v),
    "geochem_Mo_count": lambda v: _geochem_count_text("molybdenum", v),
    "geochem_Sb_median": lambda v: _geochem_median_text("Sb", v),
    "geochem_Sb_count": lambda v: _geochem_count_text("antimony", v),
    "geochem_K_median": lambda v: _geochem_median_text("K", v),
    "geochem_K_count": lambda v: _geochem_count_text("potassium", v),
    "geochem_Mg_median": lambda v: _geochem_median_text("Mg", v),
    "geochem_Mg_count": lambda v: _geochem_count_text("magnesium", v),
    "geochem_Sr_median": lambda v: _geochem_median_text("Sr", v),
    "geochem_Sr_count": lambda v: _geochem_count_text("strontium", v),
    "geochem_Mn_median": lambda v: _geochem_median_text("Mn", v),
    "geochem_Mn_count": lambda v: _geochem_count_text("manganese", v),
    "geochem_Fe_median": lambda v: _geochem_median_text("Fe", v),
    "geochem_Fe_count": lambda v: _geochem_count_text("iron", v),
    "geochem_Cu_median": lambda v: _geochem_median_text("Cu", v),
    "geochem_Cu_count": lambda v: _geochem_count_text("copper", v),
    "dist_to_fault_m": lambda v: _distance_text(
        "dist_to_fault_m",
        v,
        near="structurally complex, faulting provides fluid pathways",
        far="remote from mapped faults",
    ),
    "fault_count": lambda v: (
        "structurally complex, faulting provides fluid pathways"
        if float(v) >= max(1.0, FEATURE_MEDIANS["fault_count"] + 1.0)
        else "limited fault density in the cell"
    ),
    "dominant_lith": lambda v: (
        f"dominant mapped lithology is {v}, influencing host-rock prospectivity"
    ),
    "pct_tuff_ignimbrite": lambda v: _pct_text("volcanic tuff/ignimbrite cover", v)
    + " Volcanic tuff/ignimbrite cover is the host setting for clay-hosted lithium.",
    "pct_rhyolite": lambda v: _pct_text("rhyolite cover", v)
    + " Rhyolite cover is the host setting for clay-hosted lithium.",
    "pct_lacustrine": lambda v: _pct_text("lacustrine sediments", v)
    + " Lacustrine sediments are the host setting for brine-type lithium.",
    "pct_alluvium": lambda v: _pct_text("alluvial cover", v),
    "pct_mafic_volcanic": lambda v: _pct_text("mafic volcanic rock", v),
    "pct_intrusive": lambda v: _pct_text("intrusive rock", v),
    "dist_to_geothermal_m": lambda v: _distance_text(
        "dist_to_geothermal_m",
        v,
        near="near a mapped geothermal field, a lithium-mobilizing heat source",
        far="distant from mapped geothermal fields",
    ),
    "nearest_geo_max_meas_t": lambda v: (
        "near elevated geothermal temperatures, a lithium-mobilizing heat source"
        if float(v) >= FEATURE_MEDIANS["nearest_geo_max_meas_t"]
        else "moderate nearby geothermal temperatures"
    ),
    "nearest_geo_geotherm_t": lambda v: (
        "near elevated geothermal temperatures, a lithium-mobilizing heat source"
        if float(v) >= FEATURE_MEDIANS["nearest_geo_geotherm_t"]
        else "moderate nearby geothermal gradient"
    ),
    "nearest_geo_max_temp": lambda v: (
        "near elevated geothermal temperatures, a lithium-mobilizing heat source"
        if float(v) >= FEATURE_MEDIANS["nearest_geo_max_temp"]
        else "moderate nearby geothermal temperature indicators"
    ),
    "nearest_geo_mw_e": lambda v: (
        "elevated geothermal heat flow nearby"
        if float(v) >= FEATURE_MEDIANS["nearest_geo_mw_e"]
        else "typical regional geothermal heat flow nearby"
    ),
    "gravity_station_count": lambda v: (
        "well-sampled gravity coverage in the cell"
        if float(v) >= FEATURE_MEDIANS["gravity_station_count"]
        else "sparse gravity station coverage in the cell"
    ),
    "bouguer_mean": lambda v: (
        "gravity low consistent with thick low-density basin fill"
        if float(v) <= FEATURE_MEDIANS["bouguer_mean"] * 0.9
        else "gravity signature not strongly basin-like"
    ),
    "isostatic_mean": lambda v: (
        "strong negative isostatic gravity anomaly, indicating a deep sediment-filled basin"
        if float(v) <= FEATURE_MEDIANS["isostatic_mean"] * 0.9
        else "isostatic gravity not strongly basin-like"
    ),
    "dist_to_nearest_station_m": lambda v: _distance_text(
        "dist_to_nearest_station_m",
        v,
        near="close to a gravity station, improving gravity constraint",
        far="far from the nearest gravity station",
    ),
    "elev_min": lambda v: (
        "low-relief basin floor, favorable for brine accumulation"
        if float(v) <= FEATURE_MEDIANS["elev_min"]
        else "higher minimum elevation, less basin-floor-like"
    ),
    "elev_max": lambda v: (
        "moderate local relief"
        if float(v) - FEATURE_MEDIANS["elev_max"] <= 200.0
        else "higher maximum elevation nearby"
    ),
    "elev_mean": lambda v: (
        "low mean elevation, basin-like setting"
        if float(v) <= FEATURE_MEDIANS["elev_mean"]
        else "higher mean elevation"
    ),
    "elev_std": lambda v: (
        "low topographic variability"
        if float(v) <= FEATURE_MEDIANS["elev_std"]
        else "higher topographic variability"
    ),
    "elev_range": lambda v: (
        "low-relief basin floor, favorable for brine accumulation"
        if float(v) <= FEATURE_MEDIANS["elev_range"]
        else "greater local relief"
    ),
    "dist_to_hotspring_m": lambda v: _distance_text(
        "dist_to_hotspring_m",
        v,
        near="close to a thermal spring, indicating active fluid systems",
        far="distant from thermal springs",
    ),
    "dist_to_li_hotspring_m": lambda v: _distance_text(
        "dist_to_li_hotspring_m",
        v,
        near="close to a lithium-bearing spring, indicating active fluid systems",
        far="distant from lithium-bearing springs",
    ),
    "hotsprings_count_10km": lambda v: (
        "cluster of thermal springs within 10 km, indicating active fluid systems"
        if float(v) >= 1.0
        else "no thermal springs within 10 km"
    ),
    "hotsprings_li_count_10km": lambda v: (
        "lithium-bearing thermal springs within 10 km, indicating active fluid systems"
        if float(v) >= 1.0
        else "no lithium-bearing thermal springs within 10 km"
    ),
    "hotsprings_li_max": lambda v: (
        "elevated lithium in nearby hot-spring chemistry"
        if float(v) >= FEATURE_MEDIANS["hotsprings_li_max"]
        else "moderate lithium in nearby hot-spring chemistry"
    ),
    "hotsprings_li_median": lambda v: (
        "elevated lithium in nearby hot-spring chemistry"
        if float(v) >= FEATURE_MEDIANS["hotsprings_li_median"]
        else "moderate lithium in nearby hot-spring chemistry"
    ),
    "hotsprings_b_max": lambda v: (
        "elevated boron in nearby hot-spring chemistry, consistent with brine systems"
        if float(v) >= FEATURE_MEDIANS["hotsprings_b_max"]
        else "moderate boron in nearby hot-spring chemistry"
    ),
    "hotsprings_temp_max": lambda v: (
        "near elevated geothermal temperatures, a lithium-mobilizing heat source"
        if float(v) >= FEATURE_MEDIANS["hotsprings_temp_max"]
        else "moderate nearby hot-spring temperatures"
    ),
    "dist_to_geochem_anom_m": lambda v: _distance_text(
        "dist_to_geochem_anom_m",
        v,
        near="close to a stream-sediment geochemical anomaly",
        far="distant from stream-sediment geochemical anomalies",
    ),
    "dist_to_caldera_m": lambda v: _distance_text(
        "dist_to_caldera_m",
        v,
        near="close to a caldera margin, a common volcanic lithium setting",
        far="distant from caldera margins",
    ),
    "dist_to_tuff_m": lambda v: _distance_text(
        "dist_to_tuff_m",
        v,
        near="close to mapped tuff, the host setting for clay-hosted lithium",
        far="distant from mapped tuff",
    ),
    "dist_to_lacustrine_m": lambda v: _distance_text(
        "dist_to_lacustrine_m",
        v,
        near="sits on or near lacustrine basin sediments, the host setting for brine-type lithium",
        far="distant from lacustrine basin sediments",
    ),
    "closed_basin_indicator": lambda v: (
        "flagged as a closed-basin setting analogous to Clayton Valley"
        if int(float(v)) == 1
        else "not flagged as a closed-basin setting"
    ),
}
