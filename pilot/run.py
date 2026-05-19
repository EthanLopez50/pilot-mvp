"""CLI entry point and orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from pilot import config
from pilot.aoi import parse_aoi
from pilot.features import CoverageResult, extract
from pilot.maps import heatmap_path
from pilot.model_service import predict
from pilot.report import build

RANKED_TARGETS_FILENAME = "ranked_targets.csv"
RANKED_TABLE_HEADER = "probability  cell_id  latitude  longitude"
COVERAGE_WARNING_PREFIX = "Coverage warning:"
CSV_HEADER = "probability,cell_id,latitude,longitude,dominant_lithology\n"


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the full pilot pipeline for a client AOI."""
    parser = argparse.ArgumentParser(
        description="Generate a lithium prospectivity report for a pilot client AOI.",
    )
    parser.add_argument("--client", required=True, help="Client name for the report.")
    parser.add_argument(
        "--aoi",
        required=True,
        help="Path to AOI GeoJSON, shapefile, or bounding box.",
    )
    parser.add_argument(
        "--report-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Report cover date for reproducible output (default: today).",
    )
    args = parser.parse_args(argv)

    report_date = _parse_report_date(args.report_date)
    aoi = parse_aoi(args.aoi)
    features_df, coverage = extract(aoi)

    if not coverage.adequate:
        _print_coverage_warning(coverage)
        build(
            features_df=features_df,
            probabilities=np.asarray([], dtype=float),
            client=args.client,
            aoi_polygon=aoi.polygon,
            coverage=coverage,
            report_date=report_date,
        )
        _write_empty_ranked_targets_csv(args.client)
        return 0

    feature_cols = json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8"))
    probabilities = predict(features_df[feature_cols])
    _print_ranked_table(features_df, probabilities)
    _write_ranked_targets_csv(args.client, features_df, probabilities)
    build(
        features_df=features_df,
        probabilities=probabilities,
        client=args.client,
        aoi_polygon=aoi.polygon,
        coverage=coverage,
        report_date=report_date,
    )
    return 0


def ranked_targets_path(client: str) -> Path:
    """Return the deterministic ranked-targets CSV path for a client."""
    return heatmap_path(client).parent / RANKED_TARGETS_FILENAME


def _parse_report_date(value: str | None) -> date:
    if value is None:
        return date.today()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(
            f"Invalid --report-date {value!r}: expected YYYY-MM-DD"
        ) from exc


def _print_coverage_warning(coverage: CoverageResult) -> None:
    minimum_pct = config.COVERAGE_MIN_FRACTION * 100
    covered_pct = coverage.fraction * 100
    print(
        f"{COVERAGE_WARNING_PREFIX} only {covered_pct:.1f}% of the AOI is covered by "
        f"the precomputed grid (minimum required: {minimum_pct:.0f}%). "
        "Skipping predictions.",
        file=sys.stdout,
    )


def _ranked_rows(
    features_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> gpd.GeoDataFrame:
    """Return AOI cells ranked by probability descending, then cell_id ascending."""
    rounded_prob = np.round(np.asarray(probabilities, dtype=float), 6)
    cell_ids = features_df["cell_id"].astype(str).to_numpy()
    order = np.lexsort((cell_ids, -rounded_prob))

    ranked = features_df.iloc[order].reset_index(drop=True)
    ranked = ranked.copy()
    ranked["probability"] = rounded_prob[order]

    gdf = gpd.GeoDataFrame(
        ranked,
        geometry="geometry",
        crs=config.GRID_CRS,
    ).to_crs("EPSG:4326")
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    return gdf


def _print_ranked_table(features_df: pd.DataFrame, probabilities: np.ndarray) -> None:
    gdf = _ranked_rows(features_df, probabilities)

    print(RANKED_TABLE_HEADER, file=sys.stdout)
    for row in gdf.itertuples(index=False):
        print(
            f"{row.probability:.6f}  {row.cell_id}  {row.latitude:.6f}  {row.longitude:.6f}",
            file=sys.stdout,
        )


def _write_ranked_targets_csv(
    client: str,
    features_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> None:
    output_path = ranked_targets_path(client)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = _ranked_rows(features_df, probabilities)
    lines = [CSV_HEADER]
    for row in gdf.itertuples(index=False):
        lines.append(
            f"{row.probability:.6f},{row.cell_id},{row.latitude:.6f},"
            f"{row.longitude:.6f},{row.dominant_lith}\n"
        )
    output_path.write_text("".join(lines), encoding="utf-8", newline="")


def _write_empty_ranked_targets_csv(client: str) -> None:
    output_path = ranked_targets_path(client)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(CSV_HEADER, encoding="utf-8", newline="")


if __name__ == "__main__":
    raise SystemExit(main())
