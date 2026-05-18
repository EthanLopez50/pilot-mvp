"""Acceptance tests for Task 1 — AOI parsing and validation."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from pilot import config
from pilot.aoi import AoiError, parse_aoi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample.geojson"

# Bounding box covering the sample polygon (WGS84 lon/lat).
SAMPLE_BBOX = "-117.65,37.68,-117.45,37.88"


@pytest.fixture(scope="module")
def sample_shapefile(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shapefile with the same footprint as data/aoi/sample.geojson."""
    gdf = gpd.read_file(SAMPLE_GEOJSON)
    out_dir = tmp_path_factory.mktemp("aoi_shp")
    shp_path = out_dir / "sample.shp"
    gdf.to_file(shp_path)
    return shp_path


def test_valid_geojson_parses_to_polygon_in_grid_crs() -> None:
    result = parse_aoi(SAMPLE_GEOJSON)

    assert result.crs == config.GRID_CRS
    assert not result.polygon.is_empty
    assert result.polygon.is_valid
    assert result.polygon.area > 0


def test_valid_shapefile_parses_equivalently(sample_shapefile: Path) -> None:
    from_geojson = parse_aoi(SAMPLE_GEOJSON)
    from_shp = parse_aoi(sample_shapefile)

    assert from_shp.crs == config.GRID_CRS
    overlap = from_shp.polygon.intersection(from_geojson.polygon).area
    union = from_shp.polygon.union(from_geojson.polygon).area
    assert overlap / union > 0.99


def test_valid_bbox_parses_to_correct_rectangular_polygon() -> None:
    from_bbox = parse_aoi(SAMPLE_BBOX)
    from_geojson = parse_aoi(SAMPLE_GEOJSON)

    assert from_bbox.crs == config.GRID_CRS
    assert from_bbox.polygon.geom_type == "Polygon"
    assert from_bbox.polygon.is_valid
    assert from_bbox.polygon.area > 0
    # Bbox should cover at least the sample polygon footprint.
    assert from_bbox.polygon.contains(from_geojson.polygon) or from_bbox.polygon.intersects(
        from_geojson.polygon
    )
    expected = gpd.GeoSeries(
        [box(-117.65, 37.68, -117.45, 37.88)], crs="EPSG:4326"
    ).to_crs(config.GRID_CRS)[0]
    assert from_bbox.polygon.equals_exact(expected, tolerance=1.0)


def test_malformed_geojson_raises_error_naming_problem(tmp_path: Path) -> None:
    invalid_json = tmp_path / "broken.geojson"
    invalid_json.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(AoiError, match=r"broken\.geojson.*json|JSON|parse"):
        parse_aoi(invalid_json)

    no_geometry = tmp_path / "no_geometry.geojson"
    no_geometry.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )
    with pytest.raises(AoiError, match=r"no_geometry\.geojson.*(feature|geometry|empty)"):
        parse_aoi(no_geometry)

    point_only = tmp_path / "point.geojson"
    point_only.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "Point", "coordinates": [-117.5, 37.75]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AoiError, match=r"point\.geojson.*(polygon|area|type)"):
        parse_aoi(point_only)


def test_nonexistent_path_raises_error_naming_missing_file() -> None:
    missing = PROJECT_ROOT / "data" / "aoi" / "does_not_exist.geojson"
    with pytest.raises(
        AoiError,
        match=r"(not found|missing|exist).*does_not_exist\.geojson|does_not_exist\.geojson.*(not found|missing|exist)",
    ):
        parse_aoi(missing)


def test_zero_area_polygon_raises_specific_error(tmp_path: Path) -> None:
    degenerate = tmp_path / "zero_area.geojson"
    degenerate.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-117.5, 37.75],
                                    [-117.5, 37.75],
                                    [-117.5, 37.75],
                                    [-117.5, 37.75],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AoiError, match=r"zero_area\.geojson.*(area|zero|degenerate)"):
        parse_aoi(degenerate)


def test_invalid_bbox_raises_specific_error() -> None:
    with pytest.raises(AoiError, match=r"bounding box|bbox|four"):
        parse_aoi("1,2,3")

    with pytest.raises(AoiError, match=r"bounding box|bbox|min|max|order"):
        parse_aoi("10,20,5,15")
