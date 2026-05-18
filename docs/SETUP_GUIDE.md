# Project Setup Guide — Do This Before Using Agents

Follow these steps in order. Do not start agent work until every step is done.
The goal: a clean, locked, well-documented skeleton so agents build into
structure instead of inventing it.

## Step 1 — Create the project root

Create a NEW directory, separate from the model research repo. Keep the ML
research code and this pilot pipeline in different repositories. Example:

    C:\Users\Super\projects\lithium-pilot\

## Step 2 — Initialize git immediately

    cd C:\Users\Super\projects\lithium-pilot
    git init
    git branch -M main

Create a .gitignore before anything else. At minimum ignore: .venv/,
__pycache__/, *.pyc, outputs/, and any large data artifacts you don't want in
git history (the .gpkg grid and model .pkl files — track these with care or
store them outside git; they are large binaries).

## Step 3 — Create the directory structure

    lithium-pilot/
      docs/
      pilot/
      tests/
      models/
      data/
        grid/
        aoi/
      outputs/

`pilot/` is the package, `docs/` holds the context documents, `models/` and
`data/grid/` hold the provided frozen artifacts, `data/aoi/` holds sample and
client AOI files, `outputs/` is generated (gitignored).

## Step 4 — Drop in the context documents

Place these provided files exactly here:

- README.md                  -> project root
- .cursorrules               -> project root (note the leading dot)
- 00_PROJECT_BRIEF.md         -> docs/
- 01_ARCHITECTURE.md          -> docs/
- 02_BUILD_PLAN.md            -> docs/
- 03_CODING_STANDARDS.md      -> docs/
- 04_ACCEPTANCE_TESTS.md      -> docs/

These are the agent's instructions. They must exist before the agent starts.

## Step 5 — Place the frozen input artifacts

Copy from the model project into this repo:

- The trained model            -> models/model.pkl
- The ordered feature list     -> models/feature_cols.json
- The label encoder (if used)  -> models/label_encoder.pkl
- The precomputed grid         -> data/grid/features_v2.gpkg

Also place at least one sample AOI (a small GeoJSON polygon inside Nevada) at
data/aoi/sample.geojson, and one deliberately out-of-coverage AOI at
data/aoi/sample_out_of_coverage.geojson. The acceptance tests need both.

If you do not yet have a finalized trained model, create a placeholder model
with the same predict interface so the pipeline can be built against the
INTERFACE, not the specific model. Swap the real model in later. This decouples
the build from the still-evolving model.

## Step 6 — Set up and lock the Python environment

    uv venv .venv
    .venv\Scripts\python.exe -m pip install <the libraries you need>

Then freeze exact versions into requirements.txt:

    .venv\Scripts\python.exe -m pip freeze > requirements.txt

The geospatial stack (geopandas, rasterio, GDAL, PROJ) is fragile across
machines — the locked requirements.txt is your single most important
mistake-prevention artifact. Commit it.

## Step 7 — Verify the environment loads the artifacts

Before any agent work, manually open a Python shell in the venv and confirm:
the model file loads, the grid file opens, and the feature list reads. If these
fail now, they will fail for the agent too — fix environment issues here, by
hand, where you can reason about them.

## Step 8 — Initial commit

    git add .
    git commit -m "Project skeleton: context docs, structure, locked env, frozen artifacts"

This is your clean baseline. Every agent session starts from a committed state.

## Step 9 — Configure the agent workflow

In Cursor: confirm the agent can see .cursorrules and the docs/ folder. Your
standing instruction to the agent for every session is:

  "Read the docs in order per the README and .cursorrules. Do ONLY Task N from
  docs/02_BUILD_PLAN.md. Write the acceptance test first, implement the minimum
  to pass it, run the full test suite, then stop and show me the diff."

## Step 10 — The session loop (repeat per task)

1. git commit (clean state) before starting.
2. Tell the agent exactly one task number from the build plan.
3. Agent writes test, implements, runs suite.
4. You review the FULL diff against that task's spec and acceptance tests.
5. Reject anything outside the task's scope, including unrequested refactors.
6. When green and in-scope: git commit with a clear message.
7. Only then move to the next task.

## The two rules that prevent most damage

- One task per session, always reviewed against its spec before commit.
- Never let the agent change the model interface, dependencies, or anything
  outside the current task without you explicitly approving it in chat.
