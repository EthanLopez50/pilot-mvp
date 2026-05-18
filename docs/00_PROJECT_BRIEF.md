# 00 — Project Brief

## Purpose

Build the leanest possible pipeline that converts a pilot client's area of
interest into a professional lithium prospectivity report. The report is what we
hand to junior mining companies during free pilots to demonstrate value and earn
trust.

## The single success criterion

A working exploration geologist looks at a generated report for their own land
package and finds it geologically credible and useful. Everything in this MVP
serves that one outcome. Nothing else matters yet.

## Who runs it

Our own team, internally, from the command line. This is not self-serve
software. Pilots are high-touch: we run the pipeline, review the output, and
deliver the report personally. The MVP optimizes for *quality and
reproducibility of the deliverable*, not for being a product a stranger can use.

## Inputs

- A client name (string, used in the report).
- An area of interest, supplied as one of: a GeoJSON polygon, an ESRI
  shapefile, or a bounding box of coordinates.
- That is the entire input surface. Do not build anything to accept other
  inputs.

## Outputs (the deliverables)

For each pilot run, the pipeline produces, in a per-client output folder:

1. **A professional PDF report** containing: cover page, one-paragraph
   methodology summary, a prospectivity heatmap of the AOI, a ranked table of
   the top targets with coordinates and probabilities, a short
   per-target geological rationale, and a limitations section.
2. **A ranked target CSV** — every cell in the AOI with its predicted
   probability and key feature values, sorted by probability, with lat/lon
   centroid coordinates.
3. **Map image files** — the static heatmap PNG used in the report, saved
   separately so we can reuse it in pitch decks.

## Hard constraints

- The trained model is **frozen and provided**. The pipeline loads it; it never
  trains or modifies it.
- The Nevada feature grid is **precomputed and provided** (the
  `features_v2.gpkg` artifact). For an AOI inside Nevada, producing features is
  a spatial filter of this grid plus a buffer — NOT recomputation from raw data.
- If an AOI falls partly or wholly outside the precomputed grid coverage, the
  pipeline must say so clearly in the report rather than silently extrapolating.
- Predictions must be explainable: every top target carries a plain-language
  reason derived from the model (SHAP), never an unexplained number.

## Explicit non-goals

Do not build, scaffold, or stub: a web server, REST API, frontend, database,
authentication, billing, customer data upload, model retraining, consent
tracking, or multi-user features. If a task seems to require any of these, stop
and flag it — it means the task is mis-scoped, not that these should be added.

## Definition of "done" for the MVP

The pipeline runs end to end from a single command on a sample AOI and produces
all three deliverables, the report is geologically coherent, every automated
acceptance test in `04_ACCEPTANCE_TESTS.md` passes, and a teammate can reproduce
an identical run from the README alone.
