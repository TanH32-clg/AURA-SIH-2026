"""
risk_scoring.py
----------------
Combines the four trained models into one composite 0-100 risk score per
soldier-day, and generates human-readable "reasons" (SHAP where available,
permutation-importance fallback otherwise) for the welfare-officer dashboard.

Composite score design:
  - 70% weight: mean of the three supervised models' calibrated probabilities
    (LogReg, RandomForest, Gradient-Boosted/XGBoost)
  - 30% weight: Isolation Forest anomaly percentile
    (so a soldier can be flagged for an unusual *pattern* even if their
    absolute duty-load numbers look normal, and vice versa)

This blend is a reasonable starting point, not a validated clinical
instrument -- thresholds and weights should be tuned/re-validated against
real outcomes and reviewed by clinical/HR stakeholders before deployment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

from models import ALL_FEATURES, ANOMALY_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

FRIENDLY_NAMES = {
    "duty_hours": "Daily duty hours",
    "weekly_overtime_hrs": "Weekly overtime",
    "consecutive_duty_days": "Consecutive duty days without rest",
    "night_shift": "Night shift today",
    "rest_interval_hrs": "Rest interval between shifts",
    "deployment_duration_days": "Total deployment duration",
    "leave_used_ytd": "Leave days used (YTD)",
    "days_since_last_leave": "Days since last approved leave",
    "rejected_leave_requests_ytd": "Rejected leave requests (YTD)",
    "transfers_last_2yrs": "Unit transfers (last 2 yrs)",
    "days_since_last_transfer": "Days since last transfer",
    "leave_utilization_rate": "Leave utilization rate",
    "sleep_hours": "Self-reported sleep duration",
    "sleep_quality": "Self-reported sleep quality",
    "fatigue_level": "Self-reported fatigue",
    "energy_level": "Self-reported energy",
    "mood_score": "Self-reported mood",
    "stress_perception": "Self-reported stress",
    "duty_hours_roll7": "7-day avg duty hours",
    "duty_hours_roll30": "30-day avg duty hours",
    "workload_spike_ratio": "Workload spike (7d vs 30d)",
    "workload_vs_baseline_pct": "Workload vs personal baseline",
    "recovery_deficit_ratio": "Recovery deficit (high-intensity vs rest days)",
    "wellness_roll14": "14-day wellness trend",
    "wellness_roll21": "21-day wellness trend",
    "wellness_deviation_z": "Wellness deviation from personal baseline",
    "terrain_type": "Terrain / posting type",
    "threat_level": "Operational threat level",
}


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort().argsort()
    return order / max(len(values) - 1, 1) * 100


def score_population(system, feat: pd.DataFrame) -> pd.DataFrame:
    """Runs the trained system over `feat` and returns a scored dataframe."""
    X = feat[ALL_FEATURES]

    p_lr = system.logreg.predict_proba(X)[:, 1]
    p_rf = system.rforest.predict_proba(X)[:, 1]
    p_gb = system.gboost.predict_proba(X)[:, 1]
    supervised_mean = (p_lr + p_rf + p_gb) / 3.0

    anomaly_raw = -system.isoforest.named_steps["clf"].score_samples(
        system.isoforest.named_steps["prep"].transform(feat[ANOMALY_FEATURES])
    )
    anomaly_pctile = _percentile_rank(anomaly_raw) / 100.0

    composite = 0.7 * supervised_mean + 0.3 * anomaly_pctile
    composite_0_100 = np.clip(composite * 100, 0, 100)

    out = feat[["soldier_id", "unit_id", "date"]].copy()
    out["p_logreg"] = p_lr
    out["p_rforest"] = p_rf
    out["p_gboost"] = p_gb
    out["anomaly_percentile"] = anomaly_pctile * 100
    out["risk_score"] = composite_0_100.round(1)
    out["risk_tier"] = pd.cut(
        out["risk_score"], bins=[-1, 33, 66, 85, 100],
        labels=["low", "moderate", "elevated", "high"],
    )
    return out


def explain_individual(system, feat_row: pd.DataFrame, top_k: int = 4) -> list[dict]:
    """
    Returns the top_k human-readable reasons behind a single soldier-day's
    risk score, using SHAP on the gradient-boosting model when available,
    else a leave-one-out perturbation fallback so the dashboard always has
    *some* explanation to show.
    """
    X_row = feat_row[ALL_FEATURES]

    if HAS_SHAP:
        try:
            prep = system.gboost.named_steps["prep"]
            clf = system.gboost.named_steps["clf"]
            X_trans = prep.transform(X_row)
            feature_names = prep.get_feature_names_out()
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):  # some classifiers return [class0, class1]
                sv = sv[1]
            contributions = list(zip(feature_names, sv[0]))
            contributions.sort(key=lambda t: abs(t[1]), reverse=True)
            reasons = []
            for name, val in contributions[:top_k]:
                clean = name.split("__")[-1]
                friendly = FRIENDLY_NAMES.get(clean, clean)
                direction = "increases" if val > 0 else "decreases"
                reasons.append({
                    "feature": friendly,
                    "value": float(X_row[clean].iloc[0]) if clean in X_row.columns else None,
                    "impact": round(float(val), 4),
                    "direction": direction,
                })
            return reasons
        except Exception:
            pass  # fall through to permutation fallback below

    # --- Fallback: single-row leave-one-out perturbation against population median ---
    medians = system.feature_medians
    baseline = system.gboost.predict_proba(X_row)[:, 1][0]
    impacts = []
    for col in NUMERIC_FEATURES:
        if col not in medians:
            continue
        perturbed = X_row.copy()
        perturbed[col] = medians[col]
        p = system.gboost.predict_proba(perturbed)[:, 1][0]
        impacts.append((col, baseline - p))  # how much this feature's actual value adds to risk
    impacts.sort(key=lambda t: abs(t[1]), reverse=True)
    reasons = []
    for name, val in impacts[:top_k]:
        reasons.append({
            "feature": FRIENDLY_NAMES.get(name, name),
            "value": float(X_row[name].iloc[0]),
            "impact": round(float(val), 4),
            "direction": "increases" if val > 0 else "decreases",
        })
    return reasons


def recommend_intervention(risk_tier: str, reasons: list[dict]) -> str:
    """Simple rule-based recommendation text for the welfare officer view."""
    top_feature = reasons[0]["feature"].lower() if reasons else ""
    if risk_tier == "high":
        if "sleep" in top_feature or "fatigue" in top_feature:
            return "Priority confidential check-in recommended within 48h; consider rest rotation."
        if "leave" in top_feature:
            return "Priority confidential check-in recommended; review pending leave requests."
        return "Priority confidential check-in recommended within 48h."
    if risk_tier == "elevated":
        return "Schedule a routine welfare check-in within the week; monitor trend."
    if risk_tier == "moderate":
        return "No action required; continue passive monitoring."
    return "No action required."


if __name__ == "__main__":
    import joblib
    from models import TrainedSystem

    feat = pd.read_csv(ARTIFACTS_DIR / "features.csv", parse_dates=["date"])
    system = TrainedSystem(
        logreg=joblib.load(ARTIFACTS_DIR / "model_logreg.joblib"),
        rforest=joblib.load(ARTIFACTS_DIR / "model_rforest.joblib"),
        gboost=joblib.load(ARTIFACTS_DIR / "model_gboost.joblib"),
        gboost_kind="?",
        isoforest=joblib.load(ARTIFACTS_DIR / "model_isoforest.joblib"),
    )
    import json
    system.feature_medians = json.load(open(ARTIFACTS_DIR / "feature_medians.json"))

    latest_day = feat["date"].max()
    today = feat[feat["date"] == latest_day].reset_index(drop=True)
    scored = score_population(system, today)
    scored.to_csv(ARTIFACTS_DIR / "latest_risk_scores.csv", index=False)

    print(f"SHAP available: {HAS_SHAP}")
    print(scored.sort_values("risk_score", ascending=False).head(5).to_string())

    top_id = scored.sort_values("risk_score", ascending=False).iloc[0]["soldier_id"]
    row = today[today["soldier_id"] == top_id]
    reasons = explain_individual(system, row)
    tier = scored[scored["soldier_id"] == top_id]["risk_tier"].iloc[0]
    print(f"\nTop risk: {top_id} (tier={tier})")
    for r in reasons:
        print(f"  - {r['feature']}: value={r['value']}, impact={r['impact']} ({r['direction']} risk)")
    print("Recommendation:", recommend_intervention(tier, reasons))
