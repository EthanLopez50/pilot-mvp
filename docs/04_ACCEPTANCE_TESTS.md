# 04 — Acceptance Tests

Each task in `02_BUILD_PLAN.md` is "done" only when its tests here pass and the
full suite is green. Tests are automated (pytest) unless marked [manual review].

## Task 1 — AOI parsing

- A valid GeoJSON polygon parses to a non-empty shapely polygon in the grid CRS.
- A valid shapefile parses equivalently.
- A valid bounding box parses to the correct rectangular polygon.
- A malformed GeoJSON raises an error whose message names the problem.
- A non-existent path raises an error naming the missing file.
- A polygon with zero area raises a specific error.

## Task 2 — Model service

- A fixed sample feature DataFrame yields probabilities equal to a stored
  expected array (regression test), repeatable across runs.
- A missing `models/model.pkl` raises an error naming that file.
- A DataFrame missing a required feature column raises an error naming the
  missing column(s).
- An extra/unexpected column raises a specific error.

## Task 3 — Feature extraction

- A sample in-coverage AOI returns the expected number of cells.
- Returned columns exactly match `models/feature_cols.json` order, plus cell id
  and geometry.
- An out-of-coverage AOI returns a coverage result with `adequate == False`.
- A partially covered AOI returns the correct coverage fraction.

## Task 4 — End-to-end inference

- One CLI command on the sample AOI prints a ranked table with probability and
  lat/lon per cell.
- The out-of-coverage sample AOI prints the coverage warning and does not print
  a ranked prediction table.

## Task 5 — Explanations

- Every cell in the top-N has a non-empty rationale string.
- Each rationale references at least one of that cell's actual top SHAP
  features.
- Every feature in `feature_cols.json` has a mapping entry (test asserts no
  feature is unmapped).
- Fixed input yields fixed rationale text (seeded).

## Task 6 — Maps

- Running on the sample AOI writes a PNG to the expected
  `outputs/<client>/` path.
- Re-running overwrites without error.
- A small AOI (few cells) still produces a valid PNG.

## Task 7 — Report

- Running the full command on the sample AOI writes a PDF to the expected path.
- [manual review] The PDF contains, in order: cover page, methodology
  paragraph (verbatim), heatmap, ranked target table, per-target rationale
  section, limitations section (verbatim).
- When the AOI is out of coverage, the coverage limitation appears prominently
  near the top of the report.
- The PDF opens without errors in a standard viewer.

## Task 8 — End-to-end & reproducibility

- A single documented command produces all three deliverables in
  `outputs/<client>/`.
- [manual review] On a fresh clone, following only the README with the sample
  AOI reproduces deliverables whose numeric content (CSV probabilities) is
  identical to a reference run.
- The full pytest suite passes with zero failures.

## Standing regression guard

The model-output regression test (Task 2) must remain green through all later
tasks. If any change makes it fail, stop and investigate before proceeding —
do not edit the expected values to make it pass.
