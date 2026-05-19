"""Render the prospectivity heatmap PNG."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from pilot import config

# Grid cells in features_v2.gpkg are 5 km squares in EPSG:3857.
_CELL_SIZE_M = 5000.0
_HALF_CELL_M = _CELL_SIZE_M / 2.0

HEATMAP_FILENAME = "prospectivity_heatmap.png"


def heatmap_path(client: str) -> Path:
    """Return the deterministic heatmap PNG path for a client."""
    return _client_output_dir(client) / HEATMAP_FILENAME


def render(
    features_df: pd.DataFrame,
    client: str,
    aoi_polygon: BaseGeometry,
) -> Path:
    """Write a prospectivity heatmap PNG for grid cells in the AOI.

    Args:
        features_df: Grid rows with centroid ``geometry`` and a ``probability`` column.
        client: Client name used for the per-client output folder.
        aoi_polygon: AOI boundary in the grid CRS (EPSG:3857).

    Returns:
        Path to the written PNG under :data:`pilot.config.OUTPUTS_DIR`.

    Raises:
        ValueError: If ``probability`` or ``geometry`` is missing.
    """
    if "probability" not in features_df.columns:
        raise ValueError("features_df must include a 'probability' column")
    if "geometry" not in features_df.columns:
        raise ValueError("features_df must include a 'geometry' column")

    output_path = heatmap_path(client)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    plot_gdf = _plot_geometries(features_df, aoi_polygon)

    if not features_df.empty:
        patches, colors = _cell_patches(plot_gdf)
        collection = PatchCollection(
            patches,
            cmap="viridis",
            edgecolors="none",
            linewidths=0,
        )
        collection.set_array(np.asarray(colors, dtype=float))
        collection.set_clim(0.0, 1.0)
        ax.add_collection(collection)

    plot_gdf[plot_gdf["kind"] == "aoi"].boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.5,
        zorder=3,
    )

    _set_map_extent(ax, plot_gdf)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Lithium prospectivity")

    if not features_df.empty:
        colorbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Prospectivity probability")

    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=100)
    plt.close(fig)

    return output_path


def _client_output_dir(client: str) -> Path:
    safe_name = client.strip()
    for char in ("/", "\\", ":"):
        safe_name = safe_name.replace(char, "_")
    if not safe_name:
        raise ValueError("client name must not be empty")
    return config.OUTPUTS_DIR / safe_name


def _plot_geometries(
    features_df: pd.DataFrame,
    aoi_polygon: BaseGeometry,
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = [
        gpd.GeoDataFrame(
            {"kind": ["aoi"], "probability": [np.nan]},
            geometry=[aoi_polygon],
            crs=config.GRID_CRS,
        )
    ]
    if not features_df.empty:
        cell_gdf = gpd.GeoDataFrame(
            features_df[["probability"]].copy(),
            geometry=features_df["geometry"].values,
            crs=config.GRID_CRS,
        )
        cell_gdf["kind"] = "cell"
        cell_gdf["geometry"] = cell_gdf.geometry.apply(_cell_polygon)
        frames.append(cell_gdf)
    plot_gdf = pd.concat(frames, ignore_index=True)
    plot_gdf = gpd.GeoDataFrame(plot_gdf, geometry="geometry", crs=config.GRID_CRS)
    return plot_gdf.to_crs("EPSG:4326")


def _cell_polygon(centroid: object) -> Polygon:
    return box(
        centroid.x - _HALF_CELL_M,
        centroid.y - _HALF_CELL_M,
        centroid.x + _HALF_CELL_M,
        centroid.y + _HALF_CELL_M,
    )


def _cell_patches(plot_gdf: gpd.GeoDataFrame) -> tuple[list[MplPolygon], list[float]]:
    patches: list[MplPolygon] = []
    colors: list[float] = []
    cells = plot_gdf[plot_gdf["kind"] == "cell"]
    for row in cells.itertuples(index=False):
        geom = row.geometry
        if not isinstance(geom, Polygon):
            continue
        x, y = geom.exterior.xy
        patches.append(MplPolygon(np.column_stack([x, y]), closed=True))
        colors.append(float(row.probability))
    return patches, colors


def _set_map_extent(ax: plt.Axes, plot_gdf: gpd.GeoDataFrame) -> None:
    xmin, ymin, xmax, ymax = plot_gdf.total_bounds
    pad_x = (xmax - xmin) * 0.05 or 0.01
    pad_y = (ymax - ymin) * 0.05 or 0.01
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
