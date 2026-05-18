"""Acceptance tests for Task 0 — skeleton and config."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODULES = (
    "pilot",
    "pilot.config",
    "pilot.run",
    "pilot.aoi",
    "pilot.features",
    "pilot.model_service",
    "pilot.explain",
    "pilot.maps",
    "pilot.report",
)

PATH_ATTRS = (
    "PROJECT_ROOT",
    "MODEL_PATH",
    "FEATURE_COLS_PATH",
    "LABEL_ENCODER_PATH",
    "GRID_PATH",
    "AOI_DIR",
    "OUTPUTS_DIR",
)

CONSTANT_ATTRS = (
    "GRID_CRS",
    "BUFFER_KM",
    "RANDOM_SEED",
    "TOP_N",
    "COVERAGE_MIN_FRACTION",
)


def test_all_package_modules_import() -> None:
    for name in MODULES:
        importlib.import_module(name)


def test_config_paths_are_paths_under_project_root() -> None:
    from pilot import config

    assert config.PROJECT_ROOT == PROJECT_ROOT
    for attr in PATH_ATTRS:
        if attr == "PROJECT_ROOT":
            continue
        path = getattr(config, attr)
        assert isinstance(path, Path), f"{attr} must be a pathlib.Path"
        assert path.is_absolute(), f"{attr} must be absolute"
        try:
            path.relative_to(config.PROJECT_ROOT)
        except ValueError as exc:
            raise AssertionError(f"{attr} must live under PROJECT_ROOT") from exc


def test_config_constants_are_set() -> None:
    from pilot import config

    for attr in CONSTANT_ATTRS:
        assert hasattr(config, attr), f"missing config constant {attr}"
        assert getattr(config, attr) is not None

    assert isinstance(config.GRID_CRS, str) and config.GRID_CRS.startswith("EPSG:")
    assert config.BUFFER_KM > 0
    assert config.TOP_N > 0
    assert 0 < config.COVERAGE_MIN_FRACTION <= 1


def test_run_cli_accepts_required_args() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pilot.run",
            "--client",
            "Test Client",
            "--aoi",
            "data/aoi/sample.geojson",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
