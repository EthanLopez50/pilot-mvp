"""Acceptance tests for Task 7 — PDF report assembly."""

from __future__ import annotations

import base64
import json
import re
import zlib
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from pilot import config
from pilot.aoi import parse_aoi
from pilot.explain import rationale
from pilot.features import CoverageResult, extract
from pilot.maps import HEATMAP_FILENAME, heatmap_path
from pilot.model_service import predict
from pilot.report import (
    LIMITATIONS_TEXT,
    LOGO_PATH,
    METHODOLOGY_SUMMARY,
    REPORT_FILENAME,
    REPORT_STATUS_TEXT,
    SECTION_REPORT_STATUS,
    build,
    format_lithology_label,
    report_path,
)

# Minimal 1×1 JPEG for logo-embedding assertions (valid JFIF structure).
_MINIMAL_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00" + bytes([8] * 64)
    + b"\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    + b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08"
    + b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xaa\xff\xd9"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample.geojson"
OUT_OF_COVERAGE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample_out_of_coverage.geojson"

FEATURE_COLS: list[str] = json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8"))
CLIENT_NAME = "Task 7 Report Test Client"
OUT_OF_COVERAGE_CLIENT = "Task 7 Out Of Coverage Client"
FIXED_REPORT_DATE = date(2026, 5, 19)
PDF_MAGIC = b"%PDF-"

COVERAGE_HEADING = "Coverage limitation"
SECTION_METHODOLOGY = "Methodology"
SECTION_HEATMAP = "Prospectivity heatmap"
SECTION_RANKED_TARGETS = "Ranked targets"
SECTION_TARGET_RATIONALES = "Target rationales"
SECTION_LIMITATIONS = "Limitations"
COMPANY_NAME = "X-Ray Geoanalytics"
REPORT_TITLE = "Lithium Prospectivity Assessment"
FOOTER_TEXT = f"Produced by {COMPANY_NAME}"


@pytest.fixture
def outputs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    monkeypatch.setattr(config, "OUTPUTS_DIR", out)
    return out


@pytest.fixture
def sample_build_input() -> tuple[object, object, object, object]:
    aoi = parse_aoi(SAMPLE_GEOJSON)
    features_df, coverage = extract(aoi)
    assert coverage.adequate
    probabilities = predict(features_df[FEATURE_COLS])
    return features_df, probabilities, aoi.polygon, coverage


def _decode_pdf_literal(raw: bytes) -> str:
    decoded = (
        raw.replace(rb"\\(", b"(")
        .replace(rb"\\)", b")")
        .replace(rb"\\n", b"\n")
        .replace(rb"\\r", b"\r")
        .replace(rb"\\t", b"\t")
    )
    return decoded.decode("latin-1", errors="replace")


def _decode_pdf_stream(stream: bytes, filters: list[str]) -> bytes | None:
    data = stream
    for filt in filters:
        if filt == "ASCII85Decode":
            try:
                data = base64.a85decode(data, adobe=True)
            except ValueError:
                return None
        elif filt == "FlateDecode":
            for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
                try:
                    data = zlib.decompress(data, wbits)
                    break
                except zlib.error:
                    continue
            else:
                return None
        else:
            return None
    return data


def _stream_filters(header: bytes) -> list[str]:
    filters: list[str] = []
    if b"/ASCII85Decode" in header:
        filters.append("ASCII85Decode")
    if b"/FlateDecode" in header:
        filters.append("FlateDecode")
    return filters


def _iter_decoded_pdf_streams(data: bytes) -> list[bytes]:
    blobs = [data]
    search_from = 0
    while True:
        marker = data.find(b"stream", search_from)
        if marker < 0:
            break
        line_end = data.find(b"\n", marker)
        if line_end < 0:
            break
        content_start = line_end + 1
        content_end = data.find(b"endstream", content_start)
        if content_end < 0:
            break
        header = data[max(0, marker - 400) : marker]
        filters = _stream_filters(header)
        if filters:
            decoded = _decode_pdf_stream(data[content_start:content_end], filters)
            if decoded is not None:
                blobs.append(decoded)
        search_from = content_end + len(b"endstream")
    return blobs


def _extract_pdf_text(path: Path) -> str:
    """Extract visible text from a ReportLab PDF for assertions."""
    chunks: list[str] = []
    for blob in _iter_decoded_pdf_streams(path.read_bytes()):
        for match in re.finditer(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)", blob):
            text = _decode_pdf_literal(match.group(1))
            if text.strip():
                chunks.append(text)

        for match in re.finditer(rb"<([0-9A-Fa-f]+)>", blob):
            hex_str = match.group(1).decode("ascii")
            if len(hex_str) % 2 != 0:
                continue
            try:
                hex_bytes = bytes.fromhex(hex_str)
            except ValueError:
                continue
            if hex_bytes.startswith(b"\xfe\xff"):
                try:
                    chunks.append(hex_bytes.decode("utf-16-be"))
                except UnicodeDecodeError:
                    continue

    return "\n".join(chunks)


def _assert_valid_pdf(path: Path) -> None:
    assert path.is_file(), f"Expected PDF at {path}"
    data = path.read_bytes()
    assert data.startswith(PDF_MAGIC), f"Not a PDF file: {path}"
    assert len(data) > 500
    assert b"/Type /Page" in data
    assert b"%%EOF" in data[-2048:]


def _normalize_pdf_text(text: str) -> str:
    unescaped = text.replace(r"\(", "(").replace(r"\)", ")")
    return re.sub(r"\s+", " ", unescaped).strip()


def _rationale_bullet_lines(rationale_text: str) -> list[str]:
    lines: list[str] = []
    for line in rationale_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        lines.append(stripped)
    return lines


def _assert_no_orphaned_bullet_markers(text: str, rationale_text: str) -> None:
    """Bullet glyphs must sit on the same line as the start of their text."""
    for bullet_line in _rationale_bullet_lines(rationale_text):
        snippet = bullet_line[:40]
        assert snippet, "expected non-empty rationale bullet line"
        assert not re.search(
            rf"•\s*(?:\n|\r\n)\s*{re.escape(snippet[:20])}",
            text,
        ), f"Orphaned bullet marker before: {snippet[:20]!r}"
        normalized_snippet = _normalize_pdf_text(snippet[:30])
        assert (
            f"• {_normalize_pdf_text(snippet[:30])}" in _normalize_pdf_text(text)
            or normalized_snippet in _normalize_pdf_text(text)
        ), f"Expected contiguous bullet text for: {snippet[:30]!r}"


def _write_minimal_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_MINIMAL_JPEG)


def _assert_report_status_section(text: str, *, after_coverage: bool) -> None:
    normalized = _normalize_pdf_text(text)
    heading_pos = normalized.find(SECTION_REPORT_STATUS)
    para_positions = [
        normalized.find(_normalize_pdf_text(paragraph)) for paragraph in REPORT_STATUS_TEXT
    ]
    methodology_heading_pos = normalized.find(SECTION_METHODOLOGY)
    methodology_body_pos = normalized.find(_normalize_pdf_text(METHODOLOGY_SUMMARY))

    assert heading_pos >= 0, "Expected 'Report status and scope' heading in PDF text"
    assert all(pos >= 0 for pos in para_positions), (
        "Expected full Report status and scope section text in PDF"
    )
    assert methodology_heading_pos >= 0
    assert methodology_body_pos >= 0

    assert heading_pos < methodology_heading_pos < methodology_body_pos, (
        "Report status and scope must appear before the Methodology section"
    )
    assert heading_pos < min(para_positions), (
        "Report status and scope heading must precede its body paragraphs"
    )
    assert max(para_positions) < methodology_heading_pos, (
        "Report status and scope body must appear before the Methodology heading"
    )

    if after_coverage:
        coverage_pos = normalized.find(COVERAGE_HEADING)
        assert coverage_pos >= 0, "Expected coverage limitation heading in PDF text"
        assert coverage_pos < heading_pos, (
            "Coverage limitation must appear before Report status and scope"
        )


def _assert_section_order(text: str) -> None:
    normalized = _normalize_pdf_text(text)
    report_status_positions = {
        "report_status_heading": normalized.find(SECTION_REPORT_STATUS),
        "report_status_para_1": normalized.find(_normalize_pdf_text(REPORT_STATUS_TEXT[0])),
        "report_status_para_2": normalized.find(_normalize_pdf_text(REPORT_STATUS_TEXT[1])),
    }
    positions = {
        **report_status_positions,
        "methodology_heading": normalized.find(SECTION_METHODOLOGY),
        "methodology_body": normalized.find(_normalize_pdf_text(METHODOLOGY_SUMMARY)),
        "heatmap_heading": normalized.find(SECTION_HEATMAP),
        "ranked_heading": normalized.find(SECTION_RANKED_TARGETS),
        "rationales_heading": normalized.find(SECTION_TARGET_RATIONALES),
        "limitations_heading": normalized.find(SECTION_LIMITATIONS),
        "limitations_body": normalized.find(_normalize_pdf_text(LIMITATIONS_TEXT)),
    }
    missing = [name for name, pos in positions.items() if pos < 0]
    assert not missing, f"Missing required report section text: {missing}"

    ordered = [
        positions["report_status_heading"],
        positions["report_status_para_1"],
        positions["report_status_para_2"],
        positions["methodology_heading"],
        positions["methodology_body"],
        positions["heatmap_heading"],
        positions["ranked_heading"],
        positions["rationales_heading"],
        positions["limitations_heading"],
        positions["limitations_body"],
    ]
    assert ordered == sorted(ordered), (
        "Report sections are out of order: "
        + ", ".join(f"{name}@{pos}" for name, pos in zip(positions, ordered, strict=True))
    )


def test_sample_aoi_writes_pdf_to_expected_outputs_path(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input
    expected = report_path(CLIENT_NAME)

    result = build(
        features_df=features_df,
        probabilities=probabilities,
        client=CLIENT_NAME,
        aoi_polygon=aoi_polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )

    assert result == expected
    assert result.parent == outputs_dir / CLIENT_NAME
    assert result.name == REPORT_FILENAME
    _assert_valid_pdf(result)
    assert heatmap_path(CLIENT_NAME).is_file()
    assert heatmap_path(CLIENT_NAME).name == HEATMAP_FILENAME


def test_logo_path_points_at_pilot_assets_logo_jpg() -> None:
    expected = PROJECT_ROOT / "pilot" / "assets" / "logo.jpg"
    assert LOGO_PATH == expected


def test_pdf_embeds_logo_jpeg_when_logo_file_present(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input
    logo_file = tmp_path / "logo.jpg"
    _write_minimal_jpeg(logo_file)
    monkeypatch.setattr("pilot.report.LOGO_PATH", logo_file)

    pdf_path = build(
        features_df=features_df,
        probabilities=probabilities,
        client=CLIENT_NAME,
        aoi_polygon=aoi_polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )

    assert b"/DCTDecode" in pdf_path.read_bytes()


def test_pdf_cover_has_no_jpeg_when_logo_file_missing(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input
    missing_logo = tmp_path / "nonexistent" / "logo.jpg"
    assert not missing_logo.is_file()
    monkeypatch.setattr("pilot.report.LOGO_PATH", missing_logo)

    pdf_path = build(
        features_df=features_df,
        probabilities=probabilities,
        client=CLIENT_NAME,
        aoi_polygon=aoi_polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )

    assert b"/DCTDecode" not in pdf_path.read_bytes()


def test_ranked_targets_table_uses_readable_lithology_labels(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input

    pdf_path = build(
        features_df=features_df,
        probabilities=probabilities,
        client=CLIENT_NAME,
        aoi_polygon=aoi_polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )
    text = _extract_pdf_text(pdf_path)
    normalized = _normalize_pdf_text(text)

    ranked = features_df.copy()
    ranked["_probability"] = probabilities
    ranked = ranked.sort_values("_probability", ascending=False).head(config.TOP_N)
    table_codes = set(ranked["dominant_lith"].astype(str))

    assert "tuff_ignimbrite" in table_codes
    assert "tuff_ignimbrite" not in text
    assert format_lithology_label("tuff_ignimbrite") in normalized
    assert format_lithology_label("lacustrine") in normalized
    assert format_lithology_label("carbonate") in normalized
    for code in table_codes:
        assert code not in text or "_" not in code
        assert format_lithology_label(code) in normalized


def test_target_rationale_bullets_are_not_orphaned_from_text(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input

    pdf_path = build(
        features_df=features_df,
        probabilities=probabilities,
        client=CLIENT_NAME,
        aoi_polygon=aoi_polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )
    text = _extract_pdf_text(pdf_path)

    rationales = rationale(features_df, probabilities)
    for _cell_id, rationale_text in rationales.items():
        _assert_no_orphaned_bullet_markers(text, rationale_text)


def test_pdf_cover_page_includes_company_name_and_title(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input

    pdf_path = build(
        features_df=features_df,
        probabilities=probabilities,
        client=CLIENT_NAME,
        aoi_polygon=aoi_polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )
    text = _extract_pdf_text(pdf_path)
    normalized = _normalize_pdf_text(text)

    assert COMPANY_NAME in text
    assert REPORT_TITLE in text
    assert CLIENT_NAME in text
    assert FIXED_REPORT_DATE.isoformat() in text
    assert normalized.find(COMPANY_NAME) < normalized.find(SECTION_REPORT_STATUS), (
        "Cover company name should appear before Report status and scope"
    )


def test_pdf_content_pages_include_producer_footer(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input

    pdf_path = build(
        features_df=features_df,
        probabilities=probabilities,
        client=CLIENT_NAME,
        aoi_polygon=aoi_polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )
    text = _extract_pdf_text(pdf_path)

    assert FOOTER_TEXT in text


def test_pdf_contains_every_required_section_verbatim(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input

    pdf_path = build(
        features_df=features_df,
        probabilities=probabilities,
        client=CLIENT_NAME,
        aoi_polygon=aoi_polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )
    text = _extract_pdf_text(pdf_path)
    normalized = _normalize_pdf_text(text)

    assert CLIENT_NAME in text
    assert FIXED_REPORT_DATE.isoformat() in text
    assert _normalize_pdf_text(METHODOLOGY_SUMMARY) in normalized
    assert _normalize_pdf_text(LIMITATIONS_TEXT) in normalized
    for paragraph in REPORT_STATUS_TEXT:
        assert _normalize_pdf_text(paragraph) in normalized
    _assert_report_status_section(text, after_coverage=False)
    _assert_section_order(text)

    ranked = features_df.copy()
    ranked["_probability"] = probabilities
    ranked = ranked.sort_values("_probability", ascending=False).head(config.TOP_N)
    for cell_id in ranked["cell_id"].astype(str):
        assert str(cell_id) in text

    rationales = rationale(features_df, probabilities)
    for _cell_id, rationale_text in rationales.items():
        for line in rationale_text.splitlines():
            bullet = line.strip()
            if bullet.startswith("- "):
                bullet = bullet[2:].strip()
            if bullet:
                assert _normalize_pdf_text(bullet) in normalized


def test_out_of_coverage_shows_coverage_limitation_before_methodology(
    outputs_dir: Path,
) -> None:
    aoi = parse_aoi(OUT_OF_COVERAGE_GEOJSON)
    features_df, coverage = extract(aoi)
    assert not coverage.adequate

    pdf_path = build(
        features_df=features_df,
        probabilities=np.asarray([], dtype=float),
        client=OUT_OF_COVERAGE_CLIENT,
        aoi_polygon=aoi.polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )
    text = _extract_pdf_text(pdf_path)
    normalized = _normalize_pdf_text(text)

    _assert_report_status_section(text, after_coverage=True)
    _assert_section_order(text)
    assert _normalize_pdf_text(METHODOLOGY_SUMMARY) in normalized
    assert _normalize_pdf_text(LIMITATIONS_TEXT) in normalized
    for paragraph in REPORT_STATUS_TEXT:
        assert _normalize_pdf_text(paragraph) in normalized


def test_pdf_opens_without_errors(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input

    pdf_path = build(
        features_df=features_df,
        probabilities=probabilities,
        client=CLIENT_NAME,
        aoi_polygon=aoi_polygon,
        coverage=coverage,
        report_date=FIXED_REPORT_DATE,
    )

    _assert_valid_pdf(pdf_path)
    text = _extract_pdf_text(pdf_path)
    normalized = _normalize_pdf_text(text)
    assert _normalize_pdf_text(METHODOLOGY_SUMMARY) in normalized
    assert _normalize_pdf_text(LIMITATIONS_TEXT) in normalized


def test_identical_inputs_produce_identical_pdf_content(
    outputs_dir: Path,
    sample_build_input: tuple[object, object, object, object],
) -> None:
    features_df, probabilities, aoi_polygon, coverage = sample_build_input
    kwargs = {
        "features_df": features_df,
        "probabilities": probabilities,
        "client": CLIENT_NAME,
        "aoi_polygon": aoi_polygon,
        "coverage": coverage,
        "report_date": FIXED_REPORT_DATE,
    }

    first = build(**kwargs)
    second = build(**kwargs)

    assert _normalize_pdf_text(_extract_pdf_text(first)) == _normalize_pdf_text(
        _extract_pdf_text(second)
    )
