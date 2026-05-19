"""Acceptance tests for Task 8 — end-to-end CLI and reproducibility."""

from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pilot import config
from pilot.aoi import parse_aoi
from pilot.features import extract
from pilot.maps import HEATMAP_FILENAME, heatmap_path
from pilot.report import COVERAGE_HEADING, REPORT_FILENAME, report_path
from pilot.run import RANKED_TARGETS_FILENAME, main
from tests.test_task7_report import _extract_pdf_text, _normalize_pdf_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample.geojson"
OUT_OF_COVERAGE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample_out_of_coverage.geojson"

IN_COVERAGE_CLIENT = "Task 8 E2E Client"
OUT_OF_COVERAGE_CLIENT = "Task 8 OOC Client"
FIXED_REPORT_DATE = "2026-05-19"

RANKED_TABLE_HEADER = "probability  cell_id  latitude  longitude"
COVERAGE_WARNING_MARKER = "Coverage warning:"

EXPECTED_SAMPLE_CELL_COUNT = 90

CSV_COLUMNS = ("probability", "cell_id", "latitude", "longitude", "dominant_lithology")

_PROBABILITY_LINE = re.compile(
    r"^\s*(\d+\.\d+)\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)


@pytest.fixture
def outputs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    monkeypatch.setattr(config, "OUTPUTS_DIR", out)
    return out


def _argv(
    aoi_path: Path,
    *,
    client: str,
    report_date: str | None = FIXED_REPORT_DATE,
) -> list[str]:
    cmd = ["--client", client, "--aoi", str(aoi_path)]
    if report_date is not None:
        cmd.extend(["--report-date", report_date])
    return cmd


def _client_dir(outputs_dir: Path, client: str) -> Path:
    return outputs_dir / client


def _read_ranked_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_in_coverage_cli_writes_all_three_deliverables(outputs_dir: Path) -> None:
    result = main(_argv(SAMPLE_GEOJSON, client=IN_COVERAGE_CLIENT))
    assert result == 0

    client_dir = _client_dir(outputs_dir, IN_COVERAGE_CLIENT)
    assert (client_dir / REPORT_FILENAME).is_file()
    assert (client_dir / HEATMAP_FILENAME).is_file()
    assert (client_dir / RANKED_TARGETS_FILENAME).is_file()
    assert (client_dir / REPORT_FILENAME).stat().st_size > 500
    assert (client_dir / HEATMAP_FILENAME).stat().st_size > 100


def test_out_of_coverage_cli_writes_report_without_ranked_targets(outputs_dir: Path) -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = main(_argv(OUT_OF_COVERAGE_GEOJSON, client=OUT_OF_COVERAGE_CLIENT))
    assert result == 0

    stdout = buffer.getvalue()
    assert COVERAGE_WARNING_MARKER in stdout
    assert RANKED_TABLE_HEADER not in stdout
    assert not [line for line in stdout.splitlines() if _PROBABILITY_LINE.match(line)]

    client_dir = _client_dir(outputs_dir, OUT_OF_COVERAGE_CLIENT)
    pdf_path = client_dir / REPORT_FILENAME
    assert pdf_path.is_file()

    text = _extract_pdf_text(pdf_path)
    assert COVERAGE_HEADING in text
    normalized = _normalize_pdf_text(text)
    assert normalized.find(COVERAGE_HEADING) < normalized.find("Methodology")

    csv_path = client_dir / RANKED_TARGETS_FILENAME
    if csv_path.is_file():
        assert _read_ranked_csv(csv_path) == []


def test_ranked_csv_is_complete_sorted_and_well_formed(outputs_dir: Path) -> None:
    main(_argv(SAMPLE_GEOJSON, client=IN_COVERAGE_CLIENT))

    rows = _read_ranked_csv(_client_dir(outputs_dir, IN_COVERAGE_CLIENT) / RANKED_TARGETS_FILENAME)
    assert tuple(rows[0].keys()) == CSV_COLUMNS
    assert len(rows) == EXPECTED_SAMPLE_CELL_COUNT

    probabilities: list[float] = []
    seen_cell_ids: set[str] = set()
    for row in rows:
        prob = float(row["probability"])
        probabilities.append(prob)
        assert 0.0 <= prob <= 1.0
        assert -90.0 <= float(row["latitude"]) <= 90.0
        assert -180.0 <= float(row["longitude"]) <= 180.0
        assert row["dominant_lithology"].strip()
        assert row["cell_id"] not in seen_cell_ids
        seen_cell_ids.add(row["cell_id"])

    assert probabilities == sorted(probabilities, reverse=True)


def test_ranked_csv_matches_extracted_grid_cells(outputs_dir: Path) -> None:
    main(_argv(SAMPLE_GEOJSON, client=IN_COVERAGE_CLIENT))

    aoi = parse_aoi(SAMPLE_GEOJSON)
    features_df, coverage = extract(aoi)
    assert coverage.adequate
    expected_ids = set(features_df["cell_id"].astype(str))

    rows = _read_ranked_csv(_client_dir(outputs_dir, IN_COVERAGE_CLIENT) / RANKED_TARGETS_FILENAME)
    assert {row["cell_id"] for row in rows} == expected_ids


def test_identical_inputs_and_report_date_produce_identical_csv_and_pdf_text(
    outputs_dir: Path,
) -> None:
    argv = _argv(SAMPLE_GEOJSON, client=IN_COVERAGE_CLIENT)
    assert main(argv) == 0
    first_csv = (_client_dir(outputs_dir, IN_COVERAGE_CLIENT) / RANKED_TARGETS_FILENAME).read_bytes()
    first_pdf_text = _normalize_pdf_text(
        _extract_pdf_text(_client_dir(outputs_dir, IN_COVERAGE_CLIENT) / REPORT_FILENAME)
    )

    assert main(argv) == 0
    second_csv = (_client_dir(outputs_dir, IN_COVERAGE_CLIENT) / RANKED_TARGETS_FILENAME).read_bytes()
    second_pdf_text = _normalize_pdf_text(
        _extract_pdf_text(_client_dir(outputs_dir, IN_COVERAGE_CLIENT) / REPORT_FILENAME)
    )

    assert first_csv == second_csv
    assert first_pdf_text == second_pdf_text


def test_deliverable_paths_match_report_and_maps_helpers(outputs_dir: Path) -> None:
    main(_argv(SAMPLE_GEOJSON, client=IN_COVERAGE_CLIENT))

    assert report_path(IN_COVERAGE_CLIENT) == _client_dir(outputs_dir, IN_COVERAGE_CLIENT) / REPORT_FILENAME
    assert heatmap_path(IN_COVERAGE_CLIENT) == _client_dir(outputs_dir, IN_COVERAGE_CLIENT) / HEATMAP_FILENAME


def test_task4_subprocess_in_coverage_still_prints_ranked_table() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pilot.run",
            "--client",
            "Task 8 Subprocess Client",
            "--aoi",
            str(SAMPLE_GEOJSON),
            "--report-date",
            FIXED_REPORT_DATE,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert COVERAGE_WARNING_MARKER not in result.stdout
    assert RANKED_TABLE_HEADER in result.stdout
    rows = [
        line
        for line in result.stdout.splitlines()
        if _PROBABILITY_LINE.match(line)
    ]
    assert len(rows) == EXPECTED_SAMPLE_CELL_COUNT
