"""Acceptance tests for Task 3 — feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from pilot import config
from pilot.aoi import ParsedAoi, parse_aoi
from pilot.features import extract

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample.geojson"
OUT_OF_COVERAGE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample_out_of_coverage.geojson"

FEATURE_COLS: list[str] = json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8"))
EXPECTED_COLUMNS = FEATURE_COLS + ["cell_id", "geometry"]

# Pinned against data/grid/features_v2.gpkg with BUFFER_KM=10 on sample.geojson.
EXPECTED_SAMPLE_CELL_COUNT = 90


@pytest.fixture(scope="module")
def grid_bounds() -> tuple[float, float, float, float]:
    grid = gpd.read_file(config.GRID_PATH)
    return tuple(grid.total_bounds)


def _partial_coverage_aoi(grid_bounds: tuple[float, float, float, float]) -> ParsedAoi:
    """AOI straddling the eastern grid edge (~one third inside coverage)."""
    xmin, ymin, xmax, ymax = grid_bounds
    polygon = box(
        xmax - 50_000,
        (ymin + ymax) / 2 - 50_000,
        xmax + 100_000,
        (ymin + ymax) / 2 + 50_000,
    )
    return ParsedAoi(polygon=polygon, crs=config.GRID_CRS)


def test_in_coverage_aoi_returns_expected_cell_count() -> None:
    aoi = parse_aoi(SAMPLE_GEOJSON)
    features_df, coverage = extract(aoi)

    assert len(features_df) == EXPECTED_SAMPLE_CELL_COUNT
    assert coverage.adequate is True
    assert coverage.fraction == pytest.approx(1.0)


def test_returned_columns_match_feature_cols_plus_id_and_geometry() -> None:
    aoi = parse_aoi(SAMPLE_GEOJSON)
    features_df, _coverage = extract(aoi)

    assert list(features_df.columns) == EXPECTED_COLUMNS


def test_out_of_coverage_aoi_reports_inadequate_coverage() -> None:
    aoi = parse_aoi(OUT_OF_COVERAGE_GEOJSON)
    features_df, coverage = extract(aoi)

    assert len(features_df) == 0
    assert coverage.adequate is False
    assert coverage.fraction == pytest.approx(0.0)


def test_partially_covered_aoi_returns_correct_coverage_fraction(
    grid_bounds: tuple[float, float, float, float],
) -> None:
    aoi = _partial_coverage_aoi(grid_bounds)
    _features_df, coverage = extract(aoi)

    assert coverage.fraction == pytest.approx(1 / 3)
    assert coverage.adequate is False
