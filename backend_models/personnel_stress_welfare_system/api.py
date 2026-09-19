"""
api.py
------
FastAPI backend implementing the 3-step flow:

  Step 1  Dual Data Ingestion
          POST /ingest/hr-csv          (Stream A: central committee CSV drop-zone)
          POST /ingest/wellness-form   (Stream B: soldier's daily mobile form)

  Step 2  Backend Processing
          Internal merge -> engineered features -> 4 models -> composite score
          -> SHAP/explainability reasons. Triggered automatically after each
          HR CSV ingest (POST /score/recompute), or on a schedule in production.

  Step 3  Role-Based Decision Dashboards
          GET /dashboard/commander/{unit_id}   -> aggregated, ANONYMIZED trends
          GET /dashboard/welfare/high-risk     -> individual IDs + reasons +
                                                    recommended interventions
                                                    (welfare_officer role only)

Run with:  uvicorn api:app --reload
Requires:  pip install fastapi uvicorn pydantic   (not installed in this
           sandbox, which has no network access -- code is written and
           reviewed for correctness but not executed here.)

SECURITY NOTE: the auth stub below (`get_current_user`) is illustrative only.
Production must use a real identity provider (OAuth2/OIDC/SAML against the
existing personnel directory), short-lived tokens, per-request audit logging,
and encryption at rest for anything joining wellness data to identity. See
README.md "Ethical & Privacy Safeguards" before deploying.
"""

from __future__ import annotations

import io
import json
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from feature_engineering import engineer_features
from models import TrainedSystem
from risk_scoring import explain_individual, recommend_intervention, score_population

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

app = FastAPI(title="Personnel Stress & Welfare Monitoring API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to the real dashboard origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory store (swap for a real encrypted DB -- Postgres w/ column-level
# encryption, or similar -- in production). Kept simple here for clarity.
# ---------------------------------------------------------------------------
STATE = {
    "hr_raw": pd.DataFrame(),
    "wellness_raw": pd.DataFrame(),
    "features": pd.DataFrame(),
    "scores": pd.DataFrame(),
    "system": None,  # TrainedSystem, loaded at startup
}


def load_trained_system() -> TrainedSystem:
    system = TrainedSystem(
        logreg=joblib.load(ARTIFACT_DIR / "model_logreg.joblib"),
        rforest=joblib.load(ARTIFACT_DIR / "model_rforest.joblib"),
        gboost=joblib.load(ARTIFACT_DIR / "model_gboost.joblib"),
        gboost_kind="loaded",
        isoforest=joblib.load(ARTIFACT_DIR / "model_isoforest.joblib"),
    )
    system.feature_medians = json.load(open(ARTIFACT_DIR / "feature_medians.json"))
    return system


@app.on_event("startup")
def _startup():
    STATE["system"] = load_trained_system()
    # In production, also load any previously-ingested HR/wellness data here.


# ---------------------------------------------------------------------------
# Minimal RBAC stub -- replace with real auth. Role is passed via header only
# for local testing/demo purposes.
# ---------------------------------------------------------------------------
Role = Literal["central_committee", "unit_commander", "welfare_officer", "soldier"]


class CurrentUser(BaseModel):
    user_id: str
    role: Role
    unit_id: Optional[str] = None  # required for unit_commander / soldier roles


def get_current_user(
    x_user_id: str = Header(...),
    x_user_role: Role = Header(...),
    x_unit_id: Optional[str] = Header(default=None),
) -> CurrentUser:
    # TODO(production): validate a signed JWT / session against the identity
    # provider instead of trusting client-supplied headers.
    return CurrentUser(user_id=x_user_id, role=x_user_role, unit_id=x_unit_id)


def require_role(user: CurrentUser, allowed: set[str]):
    if user.role not in allowed:
        raise HTTPException(status_code=403, detail=f"Role '{user.role}' is not authorized for this endpoint.")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class WellnessSubmission(BaseModel):
    soldier_id: str = Field(..., description="Pseudonymous soldier ID")
    date: date_type = Field(default_factory=date_type.today)
    sleep_hours: float = Field(..., ge=0, le=14)
    sleep_quality: int = Field(..., ge=1, le=5)
    fatigue_level: int = Field(..., ge=1, le=5)
    energy_level: int = Field(..., ge=1, le=5)
    mood_score: int = Field(..., ge=1, le=5)
    stress_perception: int = Field(..., ge=1, le=5)


class IngestResult(BaseModel):
    rows_ingested: int
    message: str


class RiskRecord(BaseModel):
    soldier_id: str
    risk_score: float
    risk_tier: str


class HighRiskRecord(RiskRecord):
    unit_id: str
    reasons: list[dict]
    recommendation: str


class CommanderTrend(BaseModel):
    unit_id: str
    n_soldiers: int
    avg_risk_score: float
    pct_elevated_or_high: float
    top_contributing_factors: list[str]


# ---------------------------------------------------------------------------
# Step 1: Dual Data Ingestion
# ---------------------------------------------------------------------------
@app.post("/ingest/hr-csv", response_model=IngestResult)
async def ingest_hr_csv(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """Stream A -- central committee's encrypted batch HR/Deployment CSV."""
    require_role(user, {"central_committee"})

    raw_bytes = await file.read()
    df = pd.read_csv(io.BytesIO(raw_bytes), parse_dates=["date"])

    required_cols = {
        "soldier_id", "unit_id", "date", "duty_hours", "weekly_overtime_hrs",
        "consecutive_duty_days", "night_shift", "rest_interval_hrs",
        "deployment_duration_days", "terrain_type", "threat_level",
        "leave_used_ytd", "leave_entitlement_annual", "days_since_last_leave",
        "rejected_leave_requests_ytd", "transfers_last_2yrs", "days_since_last_transfer",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV missing required columns: {sorted(missing)}")

    STATE["hr_raw"] = pd.concat([STATE["hr_raw"], df], ignore_index=True).drop_duplicates(
        subset=["soldier_id", "date"], keep="last"
    )
    return IngestResult(rows_ingested=len(df), message="HR/deployment batch ingested.")


@app.post("/ingest/wellness-form", response_model=IngestResult)
def ingest_wellness_form(
    submission: WellnessSubmission,
    user: CurrentUser = Depends(get_current_user),
):
    """Stream B -- soldier's optional daily mobile/web wellness check-in."""
    require_role(user, {"soldier"})
    if user.user_id != submission.soldier_id:
        raise HTTPException(status_code=403, detail="Cannot submit wellness data for another soldier_id.")

    row = pd.DataFrame([submission.model_dump()])
    row["form_submitted"] = 1
    STATE["wellness_raw"] = pd.concat([STATE["wellness_raw"], row], ignore_index=True).drop_duplicates(
        subset=["soldier_id", "date"], keep="last"
    )
    return IngestResult(rows_ingested=1, message="Wellness check-in recorded. Thank you.")


# ---------------------------------------------------------------------------
# Step 2: Backend Processing (merge -> features -> 4 models -> SHAP)
# ---------------------------------------------------------------------------
@app.post("/score/recompute", response_model=IngestResult)
def recompute_scores(user: CurrentUser = Depends(get_current_user)):
    require_role(user, {"central_committee", "welfare_officer"})

    if STATE["hr_raw"].empty:
        raise HTTPException(status_code=400, detail="No HR data ingested yet.")

    merged = STATE["hr_raw"].merge(
        STATE["wellness_raw"], on=["soldier_id", "date"], how="left"
    )
    for col in ["sleep_hours", "sleep_quality", "fatigue_level", "energy_level", "mood_score", "stress_perception"]:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged["form_submitted"] = merged.get("form_submitted", 0).fillna(0)

    feat = engineer_features(merged)
    STATE["features"] = feat

    scored = score_population(STATE["system"], feat)
    STATE["scores"] = scored

    return IngestResult(rows_ingested=len(scored), message="Risk scores recomputed for all ingested records.")


# ---------------------------------------------------------------------------
# Step 3: Role-Based Decision Dashboards
# ---------------------------------------------------------------------------
@app.get("/dashboard/commander/{unit_id}", response_model=CommanderTrend)
def commander_dashboard(unit_id: str, user: CurrentUser = Depends(get_current_user)):
    """Unit commander view -- AGGREGATED, ANONYMIZED. No individual IDs exposed."""
    require_role(user, {"unit_commander", "central_committee"})
    if user.role == "unit_commander" and user.unit_id != unit_id:
        raise HTTPException(status_code=403, detail="Commanders may only view their own unit.")

    scores = STATE["scores"]
    if scores.empty:
        raise HTTPException(status_code=400, detail="No scores computed yet. Call /score/recompute first.")

    unit_scores = scores[scores["unit_id"] == unit_id]
    latest = unit_scores[unit_scores["date"] == unit_scores["date"].max()]
    if latest.empty:
        raise HTTPException(status_code=404, detail=f"No data for unit {unit_id}.")

    pct_flagged = (latest["risk_tier"].isin(["elevated", "high"])).mean() * 100

    return CommanderTrend(
        unit_id=unit_id,
        n_soldiers=latest["soldier_id"].nunique(),
        avg_risk_score=round(float(latest["risk_score"].mean()), 1),
        pct_elevated_or_high=round(float(pct_flagged), 1),
        top_contributing_factors=[
            "Workload spike (7d vs 30d)", "Recovery deficit ratio", "Wellness trend deviation",
        ],  # aggregate SHAP summary in production; kept static here for brevity
    )


@app.get("/dashboard/welfare/high-risk", response_model=list[HighRiskRecord])
def welfare_high_risk(user: CurrentUser = Depends(get_current_user), limit: int = 20):
    """Welfare officer view -- individual high-risk IDs with reasons + recommendation."""
    require_role(user, {"welfare_officer"})

    scores = STATE["scores"]
    feat = STATE["features"]
    if scores.empty:
        raise HTTPException(status_code=400, detail="No scores computed yet. Call /score/recompute first.")

    latest_date = scores["date"].max()
    latest_scores = scores[scores["date"] == latest_date]
    latest_feat = feat[feat["date"] == latest_date]

    high_risk = latest_scores[latest_scores["risk_tier"].isin(["elevated", "high"])].sort_values(
        "risk_score", ascending=False
    ).head(limit)

    results = []
    for _, row in high_risk.iterrows():
        feat_row = latest_feat[latest_feat["soldier_id"] == row["soldier_id"]]
        reasons = explain_individual(STATE["system"], feat_row)
        recommendation = recommend_intervention(row["risk_tier"], reasons)
        results.append(HighRiskRecord(
            soldier_id=row["soldier_id"],
            unit_id=row["unit_id"],
            risk_score=row["risk_score"],
            risk_tier=row["risk_tier"],
            reasons=reasons,
            recommendation=recommendation,
        ))
    return results


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
