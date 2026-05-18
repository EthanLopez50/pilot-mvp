"""Parse and validate an area of interest into a polygon."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from pilot.config import GRID_CRS

# Default CRS for comma-separated bounding boxes (min_lon,min_lat,max_lon,max_lat).
BBOX_CRS = "EPSG:4326"

_BBOX_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$"
)


class AoiError(ValueError):
    """Raised when an AOI source cannot be parsed or validated."""


@dataclass(frozen=True)
class ParsedAoi:
    """Validated AOI polygon reprojected to the feature grid CRS."""

    polygon: Polygon
    crs: str


def parse_aoi(source: str | Path) -> ParsedAoi:
    """Parse and validate an AOI from GeoJSON, shapefile, or bounding box.

    Args:
        source: Path to a ``.geojson`` / ``.json`` / ``.shp`` file, or a
            comma-separated bounding box ``min_lon,min_lat,max_lon,max_lat``
            in WGS84 (EPSG:4326).

    Returns:
        A :class:`ParsedAoi` with a non-empty polygon in :data:`GRID_CRS`.

    Raises:
        AoiError: If the source is missing, malformed, or has zero area.
    """
    source_text = str(source).strip()
    bbox_match = _BBOX_PATTERN.match(source_text)
    if bbox_match is not None:
        geometry, source_crs = _bbox_to_geometry(bbox_match.groups())
        return _finalize(geometry, source_crs, label="bounding box")

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json", ".shp"}:
        if not path.is_file():
            raise AoiError(f"AOI file not found: {path.name}")
        geometry, source_crs = _load_vector_file(path)
        return _finalize(geometry, source_crs, label=str(path))

    if path.exists():
        geometry, source_crs = _load_vector_file(path)
        return _finalize(geometry, source_crs, label=str(path))

    raise AoiError(
        f"Unrecognized AOI source {source!r}: expected a GeoJSON path, "
        "shapefile path, or bounding box min_lon,min_lat,max_lon,max_lat"
    )


def _bbox_to_geometry(groups: tuple[str, ...]) -> tuple[BaseGeometry, str]:
    min_lon, min_lat, max_lon, max_lat = (float(value) for value in groups)
    if min_lon >= max_lon:
        raise AoiError(
            "Invalid bounding box: min_lon must be less than max_lon "
            f"(got min_lon={min_lon}, max_lon={max_lon})"
        )
    if min_lat >= max_lat:
        raise AoiError(
            "Invalid bounding box: min_lat must be less than max_lat "
            f"(got min_lat={min_lat}, max_lat={max_lat})"
        )
    return box(min_lon, min_lat, max_lon, max_lat), BBOX_CRS


def _load_vector_file(path: Path) -> tuple[BaseGeometry, str]:
    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json"}:
        try:
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AoiError(f"Malformed GeoJSON in {path.name}: invalid JSON") from exc
        if payload.get("type") == "FeatureCollection" and not payload.get("features"):
            raise AoiError(f"Malformed GeoJSON in {path.name}: no features")

    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        if suffix in {".geojson", ".json"}:
            raise AoiError(f"Malformed GeoJSON in {path.name}: {exc}") from exc
        raise AoiError(f"Failed to read shapefile {path.name}: {exc}") from exc

    if gdf.empty:
        raise AoiError(f"Malformed GeoJSON in {path.name}: no features")

    geometry = unary_union(gdf.geometry)
    if geometry is None or geometry.is_empty:
        raise AoiError(f"Malformed GeoJSON in {path.name}: empty geometry")

    source_crs = gdf.crs.to_string() if gdf.crs is not None else BBOX_CRS
    return geometry, source_crs


def _finalize(geometry: BaseGeometry, source_crs: str, *, label: str) -> ParsedAoi:
    polygon = _as_polygon(geometry, label)
    if polygon.area <= 0:
        raise AoiError(f"AOI in {label} has zero area")
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.area <= 0:
        raise AoiError(f"AOI in {label} has zero area")

    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=source_crs)
    reprojected = gdf.to_crs(GRID_CRS)
    result_geom = reprojected.geometry.iloc[0]
    if not isinstance(result_geom, Polygon):
        raise AoiError(f"AOI in {label} did not produce a polygon after reprojection")
    return ParsedAoi(polygon=result_geom, crs=GRID_CRS)


def _as_polygon(geometry: BaseGeometry, label: str) -> Polygon:
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type == "MultiPolygon":
        if len(geometry.geoms) == 0:
            raise AoiError(f"Malformed AOI in {label}: empty MultiPolygon")
        return max(geometry.geoms, key=lambda geom: geom.area)
    raise AoiError(
        f"Malformed AOI in {label}: expected a polygon, got {geometry.geom_type}"
    )
