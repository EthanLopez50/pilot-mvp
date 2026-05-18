# 01 — Architecture

## Shape of the system

A single Python package run as a command-line tool. No services, no network, no
persistence beyond writing output files. This is the correct architecture for an
internally-run pilot pipeline and is deliberately minimal.

```
pilot/
  __init__.py
  run.py              # CLI entry point and orchestration
  model_service.py    # loads frozen model, exposes predict()
  aoi.py              # parse + validate an area of interest into a polygon
  features.py         # extract feature rows for an AOI from the precomputed grid
  explain.py          # SHAP-based per-target geological rationale
  report.py           # build the PDF report
  maps.py             # render the heatmap PNG
  config.py           # paths and constants, no logic
```

## Data flow (strict, one direction)

```
CLI args
  -> aoi.parse_aoi()         : input -> validated shapely polygon (+ CRS)
  -> features.extract()      : polygon -> DataFrame of grid cells + features
  -> model_service.predict() : features -> probabilities (model frozen)
  -> explain.rationale()     : per top-cell SHAP -> plain-language reasons
  -> maps.render()           : cells + probs -> heatmap PNG
  -> report.build()          : everything -> PDF
  -> run.py writes all deliverables to outputs/<client>/
```

Each module does one thing and is independently testable. No module reaches
backward in the flow. No global mutable state.

## Key design decisions (do not deviate without flagging)

1. **The model is loaded behind one function.** `model_service.predict(df)`
   takes a feature DataFrame with a documented, fixed column order and returns
   probabilities. Nothing else in the codebase knows how the model works
   internally. The model artifact and its expected feature list are provided
   inputs in `models/`.

2. **Features come from the precomputed grid, not raw data.** `features.py`
   loads the provided `features_v2.gpkg`, spatially filters to the AOI plus a
   configurable buffer (default 10 km), and returns those rows. It does NOT run
   geopandas/rasterio feature engineering from raw datasets. That heavy pipeline
   is out of scope for the MVP.

3. **Coverage check is mandatory.** If the AOI is not substantially covered by
   the precomputed grid, `features.py` returns a coverage warning that `report.py`
   surfaces prominently. Never silently produce predictions for uncovered ground.

4. **Determinism.** Same inputs + same model artifact + pinned dependencies =
   byte-identical numeric outputs. No randomness in inference. Any sampling in
   explanation/plots uses a fixed seed from `config.py`.

5. **Explainability is not optional.** Every cell in the top-N target list must
   have a rationale string from `explain.py`. A target without a reason is a bug,
   not acceptable output.

## Inputs provided to the pipeline (not built by it)

- `models/model.pkl` — the frozen trained classifier.
- `models/feature_cols.json` — the exact ordered feature list the model expects.
- `models/label_encoder.pkl` — the fitted encoder for `dominant_lith` (if used).
- `data/grid/features_v2.gpkg` — the precomputed Nevada feature grid.

The build does not create these; it consumes them. If they are absent, the
pipeline must fail with a clear, specific error naming the missing file.

## Technology constraints

- Python, dependency-pinned, `uv`-managed virtual environment.
- Geospatial stack: geopandas / shapely / rasterio as already used in the model
  project. Pin versions; the GDAL/PROJ stack is fragile across environments.
- PDF generation: a single well-supported library; keep report layout simple
  and robust rather than elaborate.
- No cloud, no Lambda, no containers for the MVP. It runs on a workstation.

## Why no web app

Pilots are free and delivered by hand. A web app adds auth, hosting,
deployment, and the serverless/geospatial problems discussed elsewhere — all
pure risk and zero pilot value. The web layer is a deliberate future phase that
begins only after pilots prove the report is worth paying for. Building it now
would be the single biggest mistake available.
