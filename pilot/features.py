"""Extract feature rows for an AOI from the precomputed grid."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from pilot import config
from pilot.aoi import ParsedAoi


class FeatureExtractionError(ValueError):
    """Raised when the feature grid cannot be loaded or used."""


@dataclass(frozen=True)
class CoverageResult:
    """How much of the AOI lies inside the precomputed grid."""

    fraction: float
    adequate: bool


@lru_cache(maxsize=1)
def _load_grid() -> gpd.GeoDataFrame:
    if not config.GRID_PATH.is_file():
        raise FeatureExtractionError(
            f"Required feature grid not found: {config.GRID_PATH.name}"
        )
    return gpd.read_file(config.GRID_PATH)


@lru_cache(maxsize=1)
def _feature_columns() -> tuple[str, ...]:
    return tuple(json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _grid_coverage_geometry() -> BaseGeometry:
    grid = _load_grid()
    return unary_union(grid.geometry)


def extract(aoi: ParsedAoi) -> tuple[pd.DataFrame, CoverageResult]:
    """Return grid feature rows for an AOI and a coverage assessment.

    Spatially filters the precomputed grid to cells intersecting the AOI polygon
    expanded by :data:`pilot.config.BUFFER_KM`. Coverage is the fraction of AOI
    area overlapping the grid, without the buffer.

    Args:
        aoi: Validated AOI in the grid CRS (EPSG:3857) from :func:`pilot.aoi.parse_aoi`.

    Returns:
        A tuple of ``(features_df, coverage)``. ``features_df`` columns are the
        model feature list in order, then ``cell_id`` and centroid ``geometry``.
        ``coverage.adequate`` is true when the covered fraction is at least
        :data:`pilot.config.COVERAGE_MIN_FRACTION`.

    Raises:
        FeatureExtractionError: If the grid or feature column list is missing.
    """
    if not config.FEATURE_COLS_PATH.is_file():
        raise FeatureExtractionError(
            f"Required feature column list not found: {config.FEATURE_COLS_PATH.name}"
        )

    grid = _load_grid()
    feature_cols = _feature_columns()
    buffer_m = config.BUFFER_KM * 1000
    search_region = aoi.polygon.buffer(buffer_m)
    selected = grid[grid.intersects(search_region)].copy()

    if selected.empty:
        features_df = _empty_features_frame(feature_cols)
    else:
        selected["geometry"] = selected.geometry.centroid
        output_cols = list(feature_cols) + ["cell_id", "geometry"]
        features_df = pd.DataFrame(selected[output_cols])

    coverage = _assess_coverage(aoi.polygon)
    return features_df, coverage


def _empty_features_frame(feature_cols: tuple[str, ...]) -> pd.DataFrame:
    columns = list(feature_cols) + ["cell_id", "geometry"]
    return pd.DataFrame(columns=columns)


def _assess_coverage(aoi_polygon: BaseGeometry) -> CoverageResult:
    grid_geom = _grid_coverage_geometry()
    aoi_area = aoi_polygon.area
    if aoi_area <= 0:
        fraction = 0.0
    else:
        covered_area = aoi_polygon.intersection(grid_geom).area
        fraction = covered_area / aoi_area
    adequate = fraction >= config.COVERAGE_MIN_FRACTION
    return CoverageResult(fraction=fraction, adequate=adequate)
