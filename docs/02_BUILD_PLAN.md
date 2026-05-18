# 02 — Build Plan

Each task below is sized so its full diff is reviewable in ~10 minutes and its
acceptance test can be written before coding. Do tasks in order. Do not begin a
task until every prior task passes its acceptance test and is committed.

For every task: write/confirm the acceptance test first, implement the smallest
change that passes it, run the full test suite, commit with a clear message.

---

## Task 0 — Skeleton and config

Create the package structure from `01_ARCHITECTURE.md`, an empty test suite that
runs, the `requirements.txt` with pinned versions, and `config.py` containing
only paths and constants (no logic). Add a `make`/script target or documented
command to run tests.

**Done when:** `pytest` runs and reports zero tests failing (zero collected is
acceptable at this stage); the package imports cleanly; the README quick-start
commands exist even if downstream steps are stubbed.

---

## Task 1 — AOI parsing and validation (`aoi.py`)

Implement `parse_aoi(source)` returning a validated shapely polygon plus its
CRS, accepting: a GeoJSON file path, a shapefile path, or a bounding box.
Reject malformed input with a clear, specific error. Reproject to the grid CRS.

**Done when:** acceptance tests for all three input types pass, and at least
three deliberately malformed inputs each raise a clear, specific error (not a
generic stack trace).

---

## Task 2 — Model service (`model_service.py`)

Load the frozen model, encoder, and ordered feature list from `models/`. Expose
`predict(features_df)` returning a probability per row. Validate that incoming
columns match the expected feature list exactly; on mismatch raise a specific
error naming the offending columns. Add a regression test: a fixed sample input
yields fixed expected probabilities.

**Done when:** the regression test passes deterministically across repeated
runs; a missing model file produces a clear named error; a feature-column
mismatch produces a clear named error.

---

## Task 3 — Feature extraction (`features.py`)

Load `features_v2.gpkg`, spatially filter to the AOI polygon plus the buffer
from `config.py`, return a DataFrame with the model's feature columns plus cell
ids and centroid geometry. Implement the coverage check: return a structured
coverage result (fraction of AOI covered, boolean adequate flag).

**Done when:** for a sample in-coverage AOI it returns the expected cell count
and columns; for an out-of-coverage AOI the coverage result correctly flags
inadequate coverage; column order matches what `model_service.predict` expects.

---

## Task 4 — End-to-end inference path

Wire `aoi -> features -> predict` in `run.py` behind the CLI, producing an
in-memory ranked table (no report yet). Coverage warnings propagate.

**Done when:** one command on a sample AOI prints a ranked cell table with
probabilities and coordinates; an out-of-coverage AOI prints the coverage
warning instead of silently producing numbers.

---

## Task 5 — Explanations (`explain.py`)

For the top-N cells, compute SHAP contributions and translate the strongest
contributors into plain-language geological rationale strings using a templated
mapping from feature name to geological meaning (provided in
`03_CODING_STANDARDS.md`). Every top cell gets a non-empty rationale.

**Done when:** every cell in the top-N has a rationale string referencing its
actual dominant features; the mapping covers every feature in
`feature_cols.json`; a fixed input yields a fixed rationale (seeded).

---

## Task 6 — Maps (`maps.py`)

Render a static prospectivity heatmap PNG of the AOI with the cells colored by
probability and the AOI boundary drawn. Deterministic output path under
`outputs/<client>/`.

**Done when:** running on the sample AOI produces a readable PNG at the expected
path; re-running overwrites cleanly; no error when the AOI is small (few cells).

---

## Task 7 — Report (`report.py`)

Assemble the PDF: cover page (client name, date), one-paragraph methodology
summary (text provided in standards doc), the heatmap, the ranked target table
(top-N with coordinates, probability, dominant lithology), the per-target
rationale section, and a fixed limitations section (text provided). If coverage
was inadequate, the limitation is stated prominently near the top.

**Done when:** running the full command on the sample AOI produces a complete,
well-formed PDF containing every required section; the coverage warning appears
prominently when triggered; the PDF opens without errors.

---

## Task 8 — End-to-end acceptance and reproducibility

A single documented command produces all three deliverables in
`outputs/<client>/`. A teammate following only the README on a clean checkout
reproduces an identical run.

**Done when:** every acceptance test in `04_ACCEPTANCE_TESTS.md` passes; a clean
clone + README steps + sample AOI yields the deliverables with identical numeric
content.

---

## After the MVP (do NOT build now — listed only to prevent scope creep)

Web app, API, accounts, billing, customer-data upload + feature override,
consent layer, retraining flywheel. These are explicitly future phases and must
not be started, scaffolded, or stubbed during MVP work.
