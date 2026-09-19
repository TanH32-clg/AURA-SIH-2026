"""
models.py
---------
Implements the four requested models:

  1. Logistic Regression   -> interpretable linear baseline risk classifier
  2. Random Forest         -> nonlinear risk classifier, handles interactions
  3. XGBoost                -> gradient-boosted risk classifier (best raw AUC,
                               typically the primary score); falls back to
                               sklearn's HistGradientBoostingClassifier if the
                               `xgboost` package isn't installed, so this file
                               runs in restricted environments too
  4. Isolation Forest       -> UNSUPERVISED anomaly detector over the wellness
                               trajectory features -- flags soldiers whose
                               pattern is anomalous even if no single feature
                               crosses a threshold (catches novel presentations
                               of risk the supervised label didn't anticipate)

Composite risk score = calibrated blend of the three supervised models'
probability outputs + an anomaly-boost from Isolation Forest.

NOTE ON THE TRAINING LABEL: real deployments will *not* have a ground-truth
"is_at_risk" label. This module synthesizes one for demonstration by
thresholding the hidden `_latent_strain` simulation variable (which is never
shown to the models as a feature). In production, replace `build_label()`
with real outcome data (e.g., substantiated welfare-officer interventions,
clinical referrals) collected under proper consent/ethics review -- see
README.md.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

NUMERIC_FEATURES = [
    "duty_hours", "weekly_overtime_hrs", "consecutive_duty_days", "night_shift",
    "rest_interval_hrs", "deployment_duration_days", "leave_used_ytd",
    "days_since_last_leave", "rejected_leave_requests_ytd", "transfers_last_2yrs",
    "days_since_last_transfer", "leave_utilization_rate",
    "sleep_hours", "sleep_quality", "fatigue_level", "energy_level",
    "mood_score", "stress_perception",
    "duty_hours_roll7", "duty_hours_roll30", "workload_spike_ratio",
    "workload_vs_baseline_pct", "recovery_deficit_ratio",
    "wellness_roll14", "wellness_roll21", "wellness_deviation_z",
]
CATEGORICAL_FEATURES = ["terrain_type", "threat_level"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The Isolation Forest only looks at the *wellness trajectory* signal --
# it's meant to catch sudden personal deviation, not restate the duty-load
# story the supervised models already cover.
ANOMALY_FEATURES = [
    "sleep_hours", "sleep_quality", "fatigue_level", "energy_level",
    "mood_score", "stress_perception", "wellness_deviation_z",
    "wellness_roll14", "workload_vs_baseline_pct", "recovery_deficit_ratio",
]


def build_label(df: pd.DataFrame, threshold: float = 0.55) -> pd.Series:
    """
    Synthesizes a demonstration ground-truth label from the hidden latent
    strain variable plus a bit of independent noise (so it's not perfectly
    separable from the observed features -- more realistic for evaluation).
    Replace with real outcome data in production.
    """
    noise = np.random.default_rng(7).normal(0, 0.08, size=len(df))
    score = df["_latent_strain"].to_numpy() + noise
    return (score > threshold).astype(int)


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


@dataclass
class TrainedSystem:
    logreg: Pipeline
    rforest: Pipeline
    gboost: Pipeline
    gboost_kind: str  # "xgboost" or "hist_gb" (fallback)
    isoforest: Pipeline
    feature_medians: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def train_all_models(feat: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> TrainedSystem:
    df = feat.copy()
    y = build_label(df)
    X = df[ALL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    metrics = {}

    # 1. Logistic Regression
    logreg = Pipeline([
        ("prep", _make_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    logreg.fit(X_train, y_train)
    p = logreg.predict_proba(X_test)[:, 1]
    metrics["logistic_regression"] = {
        "roc_auc": round(roc_auc_score(y_test, p), 4),
        "pr_auc": round(average_precision_score(y_test, p), 4),
        "f1": round(f1_score(y_test, p > 0.5), 4),
    }

    # 2. Random Forest
    rforest = Pipeline([
        ("prep", _make_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=15,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )),
    ])
    rforest.fit(X_train, y_train)
    p = rforest.predict_proba(X_test)[:, 1]
    metrics["random_forest"] = {
        "roc_auc": round(roc_auc_score(y_test, p), 4),
        "pr_auc": round(average_precision_score(y_test, p), 4),
        "f1": round(f1_score(y_test, p > 0.5), 4),
    }

    # 3. XGBoost (or HistGradientBoosting fallback)
    if HAS_XGBOOST:
        gb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=random_state,
        )
        gboost_kind = "xgboost"
    else:
        gb_clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300, random_state=random_state,
        )
        gboost_kind = "hist_gb (xgboost not installed in this environment)"

    gboost = Pipeline([("prep", _make_preprocessor()), ("clf", gb_clf)])
    gboost.fit(X_train, y_train)
    p = gboost.predict_proba(X_test)[:, 1]
    metrics["gradient_boosting"] = {
        "kind": gboost_kind,
        "roc_auc": round(roc_auc_score(y_test, p), 4),
        "pr_auc": round(average_precision_score(y_test, p), 4),
        "f1": round(f1_score(y_test, p > 0.5), 4),
    }

    # 4. Isolation Forest (unsupervised, wellness-trajectory anomaly detector)
    isoforest = Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), ANOMALY_FEATURES)])),
        ("clf", IsolationForest(
            n_estimators=300, contamination=0.08, random_state=random_state,
        )),
    ])
    isoforest.fit(df[ANOMALY_FEATURES])
    # sanity metric: correlation between anomaly score and the label (informational only --
    # isolation forest is unsupervised and isn't optimized against y)
    anomaly_score = -isoforest.named_steps["clf"].score_samples(
        isoforest.named_steps["prep"].transform(df[ANOMALY_FEATURES])
    )
    metrics["isolation_forest"] = {
        "note": "unsupervised; reported AUC is informational only",
        "roc_auc_vs_synthetic_label": round(roc_auc_score(y, anomaly_score), 4),
        "flagged_rate": round(float((anomaly_score > np.percentile(anomaly_score, 92)).mean()), 4),
    }

    feature_medians = X[NUMERIC_FEATURES].median().to_dict()

    return TrainedSystem(
        logreg=logreg, rforest=rforest, gboost=gboost, gboost_kind=gboost_kind,
        isoforest=isoforest, feature_medians=feature_medians, metrics=metrics,
    )


def save_system(system: TrainedSystem, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(system.logreg, out_dir / "model_logreg.joblib")
    joblib.dump(system.rforest, out_dir / "model_rforest.joblib")
    joblib.dump(system.gboost, out_dir / "model_gboost.joblib")
    joblib.dump(system.isoforest, out_dir / "model_isoforest.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"gboost_kind": system.gboost_kind, **system.metrics}, f, indent=2)
    with open(out_dir / "feature_medians.json", "w") as f:
        json.dump(system.feature_medians, f, indent=2)


if __name__ == "__main__":
    feat = pd.read_csv(ARTIFACTS_DIR / "features.csv", parse_dates=["date"])
    system = train_all_models(feat)
    save_system(system, ARTIFACTS_DIR)
    print(f"xgboost available: {HAS_XGBOOST}")
    print(json.dumps(system.metrics, indent=2))
