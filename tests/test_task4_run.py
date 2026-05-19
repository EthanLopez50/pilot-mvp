"""Acceptance tests for Task 4 — end-to-end inference CLI."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample.geojson"
OUT_OF_COVERAGE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample_out_of_coverage.geojson"

RANKED_TABLE_HEADER = "probability  cell_id  latitude  longitude"
COVERAGE_WARNING_MARKER = "Coverage warning:"

# Pinned against data/grid/features_v2.gpkg with BUFFER_KM=10 on sample.geojson.
EXPECTED_SAMPLE_CELL_COUNT = 90

_PROBABILITY_LINE = re.compile(
    r"^\s*(\d+\.\d+)\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)


def _run_cli(aoi_path: Path, *, client: str = "Acceptance Test Client") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pilot.run",
            "--client",
            client,
            "--aoi",
            str(aoi_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _parse_ranked_table(stdout: str) -> list[tuple[float, str, float, float]]:
    lines = stdout.splitlines()
    try:
        header_index = lines.index(RANKED_TABLE_HEADER)
    except ValueError as exc:
        raise AssertionError(
            f"Expected ranked table header {RANKED_TABLE_HEADER!r} in stdout"
        ) from exc

    rows: list[tuple[float, str, float, float]] = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            continue
        match = _PROBABILITY_LINE.match(line)
        if not match:
            break
        prob, cell_id, lat, lon = match.groups()
        rows.append((float(prob), cell_id, float(lat), float(lon)))
    return rows


def test_in_coverage_sample_prints_ranked_table_with_probability_and_coordinates() -> None:
    result = _run_cli(SAMPLE_GEOJSON)

    assert result.returncode == 0, result.stderr or result.stdout
    assert COVERAGE_WARNING_MARKER not in result.stdout

    rows = _parse_ranked_table(result.stdout)
    assert len(rows) == EXPECTED_SAMPLE_CELL_COUNT

    probabilities = [row[0] for row in rows]
    assert probabilities == sorted(probabilities, reverse=True)

    for prob, _cell_id, lat, lon in rows:
        assert 0.0 <= prob <= 1.0
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0


def test_out_of_coverage_sample_prints_warning_without_ranked_table() -> None:
    result = _run_cli(OUT_OF_COVERAGE_GEOJSON)

    assert result.returncode == 0, result.stderr or result.stdout
    assert COVERAGE_WARNING_MARKER in result.stdout
    assert RANKED_TABLE_HEADER not in result.stdout
    assert _parse_probability_lines(result.stdout) == []


def _parse_probability_lines(stdout: str) -> list[str]:
    """Return stdout lines that look like ranked-table data rows."""
    return [line for line in stdout.splitlines() if _PROBABILITY_LINE.match(line)]
