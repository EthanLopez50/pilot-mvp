"""CLI entry point and orchestration (stub until Task 4)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the pipeline (not yet implemented)."""
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
    print(
        f"Pipeline not yet implemented (client={args.client!r}, aoi={args.aoi!r}).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
