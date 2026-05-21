# FastAPI to run the pilot pipeline behind a HTTP endpoint
from __future__ import annotations
 
import json
import tempfile
import os
from datetime import date
from pathlib import Path
 
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# imports the pilot pipeline
from pilot.aoi import parse_aoi
from pilot.features import extract
from pilot.model_service import predict
from pilot.report import build

FEATURE_COLS_PATH = Path("models/feature_cols.json")

app = FastAPI(title="Lithium Prospectivity MVP", version="0.1.0")

class ReportRequest(BaseModel):
    client: str = Field(..., min_length=1, description="Client name used in the report.")
    aoi_geojson: dict = Field(..., description="The area of interest as a GeoJSON object.")
    report_date: str | None = Field(None, description="Date (YYYY-MM-DD). Defaults to today.")

def _load_feature_cols() -> list[str]:
    if not FEATURE_COLS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"feature_cols file not found at {FEATURE_COLS_PATH}. "
                   "Check the FEATURE_COLS_PATH is in api.py."
        )
    with FEATURE_COLS_PATH.open() as f:
        return json.load(f)

@app.get("/")
def health() -> dict:
    return {"Status": "Ok", "service": "mvp-portal-api", "version": "0.1.0"}

@app.post("/reports")
def create_report(request: ReportRequest):
    if request.report_date:
        try:
            report_date = date.fromisoformat(request.report_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="report_date must be in YYYY-MM-DD format")
    else:
        report_date = date.today()

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False)
    try:
        json.dump(request.aoi_geojson, tmp)
        tmp.close()  # close so parse_aoi can open the path (required on Windows)
        aoi_path = tmp.name

        try:
            aoi = parse_aoi(aoi_path)
            features_df, coverage = extract(aoi)

            if coverage.adequate:
                feature_cols = _load_feature_cols()
                probabilities = predict(features_df[feature_cols])
            else:
                probabilities = []

            pdf_path = build(
                features_df=features_df,
                probabilities=probabilities,
                client=request.client,
                aoi_polygon=aoi.polygon,
                coverage=coverage,
                report_date=report_date,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Pipeline error: {exc}")
    finally:
        # Always remove the temp file, even if the pipeline raised.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline reported success but no PDF was found at {pdf_path}.",
        )
 
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{request.client}_prospectivity_report.pdf",
    )