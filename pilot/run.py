"""CLI entry point and orchestration."""

from __future__ import annotations

import argparse
import json
import sys

import geopandas as gpd
import pandas as pd

from pilot import config
from pilot.aoi import parse_aoi
from pilot.features import CoverageResult, extract
from pilot.model_service import predict

RANKED_TABLE_HEADER = "probability  cell_id  latitude  longitude"
COVERAGE_WARNING_PREFIX = "Coverage warning:"


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run inference for a pilot client AOI."""
    parser = argparse.ArgumentParser(
        description="Generate a lithium prospectivity report for a pilot client AOI.",
    )
    parser.add_argument("--client", required=True, help="Client name for the report.")
    parser.add_argument(
        "--aoi",
        required=True,
        help="Path to AOI GeoJSON, shapefile, or bounding box.",
    )
    args = parser.parse_args(argv)

    aoi = parse_aoi(args.aoi)
    features_df, coverage = extract(aoi)

    if not coverage.adequate:
        _print_coverage_warning(coverage)
        return 0

    feature_cols = json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8"))
    probabilities = predict(features_df[feature_cols])
    _print_ranked_table(features_df, probabilities)
    return 0


def _print_coverage_warning(coverage: CoverageResult) -> None:
    minimum_pct = config.COVERAGE_MIN_FRACTION * 100
    covered_pct = coverage.fraction * 100
    print(
        f"{COVERAGE_WARNING_PREFIX} only {covered_pct:.1f}% of the AOI is covered by "
        f"the precomputed grid (minimum required: {minimum_pct:.0f}%). "
        "Skipping predictions.",
        file=sys.stdout,
    )


def _print_ranked_table(features_df: pd.DataFrame, probabilities: object) -> None:
    ranked = features_df.copy()
    ranked["probability"] = probabilities
    gdf = gpd.GeoDataFrame(
        ranked,
        geometry="geometry",
        crs=config.GRID_CRS,
    )
    gdf = gdf.to_crs("EPSG:4326")
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    gdf = gdf.sort_values("probability", ascending=False)

    print(RANKED_TABLE_HEADER, file=sys.stdout)
    for row in gdf.itertuples(index=False):
        print(
            f"{row.probability:.6f}  {row.cell_id}  {row.latitude:.6f}  {row.longitude:.6f}",
            file=sys.stdout,
        )


if __name__ == "__main__":
    raise SystemExit(main())
