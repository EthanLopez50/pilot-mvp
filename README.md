# Lithium Prospectivity — Pilot Delivery Pipeline (MVP)

This repository turns a client's area of interest into a professional lithium
prospectivity report using a pre-trained model and a precomputed Nevada feature
grid. It exists to deliver **free, high-touch pilot analyses** to junior mining
companies.

## What this is

A reproducible command-line pipeline:

```
input:  a client name + an area-of-interest polygon (GeoJSON / shapefile / bbox)
output: a professional PDF report + ranked target CSV + map images
```

## What this is NOT (deliberately out of scope for the MVP)

- No web application, frontend, or user accounts
- No billing or subscriptions
- No database
- No customer-data upload / feature override
- No consent or data-aggregation layer
- No model training (the model is pre-trained and frozen)

These are future phases. The MVP's only job is to produce a credible report a
geologist will respect, run by our own team for pilot clients.

## For AI coding agents

Before doing any work, read these documents in order:

1. `docs/00_PROJECT_BRIEF.md` — scope and goals
2. `docs/01_ARCHITECTURE.md` — the intended technical design
3. `docs/02_BUILD_PLAN.md` — the sequenced, agent-sized task list
4. `docs/03_CODING_STANDARDS.md` — conventions you must follow
5. `docs/04_ACCEPTANCE_TESTS.md` — how each task is verified
6. `.cursorrules` — persistent hard constraints (also enforced automatically)

Do exactly one task from the build plan at a time. Do not start a task whose
predecessors are not complete and tested.

## Quick start (once built)

```
uv venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pilot.run --client "Acme Lithium" --aoi data/aoi/acme.geojson
```
