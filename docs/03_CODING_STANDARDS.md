# 03 — Coding Standards & Conventions

## General

- Python, type hints on all public functions, docstrings stating inputs,
  outputs, and raised errors.
- One responsibility per module per `01_ARCHITECTURE.md`. No module reaches
  backward in the data flow. No global mutable state.
- Fail loudly and specifically. Every error names the file, value, or column at
  fault. Never swallow exceptions to "keep going."
- Determinism: any randomness uses the single seed in `config.py`.
- All paths come from `config.py`. No hard-coded paths anywhere else.
- Pin every dependency in `requirements.txt` with `==`. Do not add a dependency
  without recording why in the commit message.

## Testing

- Every module has a matching test file. Tests are written before or alongside
  implementation, never after as an afterthought.
- Include negative tests: malformed AOI inputs, missing model files, feature
  column mismatch, out-of-coverage AOI. These are required deliverables, not
  optional.
- A regression test pins model output for a fixed input. If it changes, that is
  a red flag to investigate, not a test to update casually.

## Style

- Small functions, descriptive names, no clever one-liners.
- Comments explain *why*, not *what*.
- No premature abstraction. Two concrete cases before any generalization.

## Geological feature → plain-language mapping (for `explain.py`)

When SHAP identifies a feature as a strong contributor for a target cell,
translate it using language in this register. Adapt the numbers to the cell's
actual values; keep the geological framing.

- `dist_to_lacustrine_m` (low) → "sits on or near lacustrine basin sediments,
  the host setting for brine-type lithium."
- `isostatic_mean` (strongly negative) → "strong negative isostatic gravity
  anomaly, indicating a deep sediment-filled basin."
- `bouguer_mean` (strongly negative) → "gravity low consistent with thick
  low-density basin fill."
- `dist_to_li_hotspring_m` / `dist_to_hotspring_m` (low) → "close to a
  lithium-bearing or thermal spring, indicating active fluid systems."
- `pct_lacustrine` (high) → "dominated by lacustrine sediments."
- `pct_tuff_ignimbrite` / `pct_rhyolite` (high) → "volcanic tuff/rhyolite
  cover, the host setting for clay-hosted lithium."
- `geochem_Li_max` / `geochem_Li_median` (high) → "elevated lithium in nearby
  stream-sediment geochemistry."
- `fault_count` (high) / `dist_to_fault_m` (low) → "structurally complex,
  faulting provides fluid pathways."
- `elev_range` (low) + `elev_min` (low) → "low-relief basin floor, favorable
  for brine accumulation."
- `closed_basin_indicator` (1) → "flagged as a closed-basin setting analogous
  to Clayton Valley."
- Geothermal temperature features (high) → "near elevated geothermal
  temperatures, a lithium-mobilizing heat source."

The mapping must cover every feature in `models/feature_cols.json`. If a feature
has no entry, that is a gap to fill, not to skip.

## Fixed report text (use verbatim in `report.py`)

**Methodology summary paragraph:**

> This report was produced by a machine-learning prospectivity model trained on
> public geoscientific datasets for Nevada, including USGS deposit records,
> NURE-HSSR stream-sediment geochemistry, USGS gravity, the State Geologic Map,
> NBMG hot-spring chemistry, and SRTM elevation. Predictions are generated over
> a uniform grid and validated with spatial cross-validation. This is a
> regional prospectivity assessment intended for early-stage target generation.

**Limitations section:**

> This assessment identifies regional geological prospectivity and is intended
> for early-stage target generation, not drill-site selection. It is scoped to
> brine-type and clay-hosted lithium systems characteristic of Nevada and does
> not incorporate proprietary drilling, detailed geophysical surveys, or remote
> sensing. Stream-sediment geochemical signals carry spatial uncertainty, as the
> lithium source may lie upstream of the sample point. Predictions for areas
> with sparse public-data coverage are lower confidence and are flagged where
> applicable. This analysis does not constitute a mineral resource estimate or
> investment advice.

Do not paraphrase these. They are reviewed wording.

## Out-of-scope guardrail

If any task appears to require a web server, database, auth, billing, customer
upload, model training, or consent logic, stop and flag it in your response.
That signals a mis-scoped task, not a feature to add. See `00_PROJECT_BRIEF.md`.
