"""Acceptance tests for Task 6 — prospectivity heatmap maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pilot import config
from pilot.aoi import parse_aoi
from pilot.features import extract
from pilot.maps import (
    HEATMAP_FILENAME,
    _cell_patches,
    _plot_geometries,
    heatmap_path,
    render,
)
from pilot.model_service import predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_GEOJSON = PROJECT_ROOT / "data" / "aoi" / "sample.geojson"

FEATURE_COLS: list[str] = json.loads(config.FEATURE_COLS_PATH.read_text(encoding="utf-8"))
CLIENT_NAME = "Task 6 Map Test Client"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def outputs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    monkeypatch.setattr(config, "OUTPUTS_DIR", out)
    return out


@pytest.fixture
def sample_render_input() -> tuple[object, object]:
    aoi = parse_aoi(SAMPLE_GEOJSON)
    features_df, coverage = extract(aoi)
    assert coverage.adequate
    frame = features_df.copy()
    frame["probability"] = predict(features_df[FEATURE_COLS])
    return frame, aoi.polygon


def _assert_valid_png(path: Path) -> None:
    assert path.is_file(), f"Expected PNG at {path}"
    data = path.read_bytes()
    assert data.startswith(PNG_MAGIC), f"Not a PNG file: {path}"
    assert len(data) > 100


def _assert_colored_cells_in_image(path: Path, *, min_unique_colors: int = 8) -> None:
    """Heatmap must contain filled cell colors, not only background and outline."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    flat = rgb.reshape(-1, 3)
    unique = np.unique(flat, axis=0)
    assert len(unique) >= min_unique_colors, (
        f"Expected colored grid cells in {path}, found only {len(unique)} distinct colors"
    )
    assert rgb.std() >= 12.0, (
        f"Expected non-uniform pixel values from filled cells in {path}, std={rgb.std():.2f}"
    )


def _assert_cell_patch_count(
    features_df: object,
    aoi_polygon: object,
    *,
    expected: int,
) -> None:
    plot_gdf = _plot_geometries(features_df, aoi_polygon)
    patches, colors = _cell_patches(plot_gdf)
    assert len(patches) == expected
    assert len(colors) == expected


def test_sample_aoi_writes_png_to_expected_outputs_path(
    outputs_dir: Path,
    sample_render_input: tuple[object, object],
) -> None:
    features_df, aoi_polygon = sample_render_input
    expected = heatmap_path(CLIENT_NAME)

    result = render(features_df, CLIENT_NAME, aoi_polygon)

    assert result == expected
    assert result.parent == outputs_dir / CLIENT_NAME
    assert result.name == HEATMAP_FILENAME
    _assert_valid_png(result)
    _assert_cell_patch_count(features_df, aoi_polygon, expected=len(features_df))
    _assert_colored_cells_in_image(result)


def test_rerender_overwrites_without_error(
    outputs_dir: Path,
    sample_render_input: tuple[object, object],
) -> None:
    features_df, aoi_polygon = sample_render_input

    first = render(features_df, CLIENT_NAME, aoi_polygon)
    _assert_valid_png(first)

    second = render(features_df, CLIENT_NAME, aoi_polygon)
    _assert_valid_png(second)
    assert second == first


def test_few_cells_still_produces_valid_png(
    outputs_dir: Path,
    sample_render_input: tuple[object, object],
) -> None:
    features_df, aoi_polygon = sample_render_input
    small = features_df.head(3).copy()
    client = "Task 6 Small AOI Client"

    path = render(small, client, aoi_polygon)

    assert path.parent == outputs_dir / client
    _assert_valid_png(path)
    _assert_cell_patch_count(small, aoi_polygon, expected=3)
    _assert_colored_cells_in_image(path, min_unique_colors=4)
