"""
feature_engineering.py
-----------------------
Turns raw daily (soldier x date) rows into the engineered features described
in the spec:

  Workload Spikes      -> 7d / 30d rolling duty-hour averages vs personal baseline
  Recovery Deficit      -> ratio of high-intensity duty days to rest days in a window
  Wellness Trajectory    -> z-score deviation of 14-21d wellness rolling mean from
                            the soldier's own established baseline (first 21 days)

Also imputes/ffills the *voluntary* wellness fields (they're optional, so
missingness is expected) using a soldier-level forward-fill capped at 3 days,
then a soldier-level median for anything still missing early in the series.
This avoids fabricating wellness signals across long non-response gaps.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

WELLNESS_COLS = [
    "sleep_hours", "sleep_quality", "fatigue_level",
    "energy_level", "mood_score", "stress_perception",
]

ORG_NUMERIC_COLS = [
    "duty_hours", "weekly_overtime_hrs", "consecutive_duty_days",
    "night_shift", "rest_interval_hrs", "deployment_duration_days",
    "leave_used_ytd", "days_since_last_leave", "rejected_leave_requests_ytd",
    "transfers_last_2yrs", "days_since_last_transfer",
]


def _impute_wellness(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    for col in WELLNESS_COLS:
        g[col] = g[col].ffill(limit=3)
        med = g[col].median()
        g[col] = g[col].fillna(med)
    return g


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["soldier_id", "date"]).copy()

    out_parts = []
    for sid, g in df.groupby("soldier_id"):
        g = _impute_wellness(g)

        # Workload spikes: 7d / 30d rolling duty hours vs personal baseline
        g["duty_hours_roll7"] = g["duty_hours"].rolling(7, min_periods=3).mean()
        g["duty_hours_roll30"] = g["duty_hours"].rolling(30, min_periods=10).mean()
        personal_baseline = g["duty_hours"].iloc[:21].mean()
        g["duty_baseline"] = personal_baseline
        g["workload_spike_ratio"] = g["duty_hours_roll7"] / g["duty_hours_roll30"].replace(0, np.nan)
        g["workload_vs_baseline_pct"] = (g["duty_hours_roll7"] - personal_baseline) / max(personal_baseline, 1e-6)

        # Recovery deficit: high-intensity days vs rest days over trailing 14d window
        high_intensity = (g["duty_hours"] > 10).astype(int)
        rest_day = g["is_rest_day"]
        g["high_intensity_days_14d"] = high_intensity.rolling(14, min_periods=5).sum()
        g["rest_days_14d"] = rest_day.rolling(14, min_periods=5).sum()
        g["recovery_deficit_ratio"] = g["high_intensity_days_14d"] / (g["rest_days_14d"] + 1)

        # Wellness trajectory: rolling 14d/21d mean of a composite wellness index
        # composite: higher = better wellbeing (so we can detect *downward* deviation)
        g["wellness_index"] = (
            g["sleep_quality"].astype(float)
            + g["energy_level"].astype(float)
            + g["mood_score"].astype(float)
            + (6 - g["fatigue_level"].astype(float))
            + (6 - g["stress_perception"].astype(float))
        ) / 5.0

        wellness_baseline = g["wellness_index"].iloc[:21].mean()
        wellness_std = g["wellness_index"].iloc[:21].std(ddof=0) or 1.0
        g["wellness_baseline"] = wellness_baseline
        g["wellness_roll14"] = g["wellness_index"].rolling(14, min_periods=5).mean()
        g["wellness_roll21"] = g["wellness_index"].rolling(21, min_periods=7).mean()
        g["wellness_deviation_z"] = (g["wellness_roll14"] - wellness_baseline) / wellness_std

        # Leave utilization rate
        g["leave_utilization_rate"] = g["leave_used_ytd"] / g["leave_entitlement_annual"].replace(0, np.nan)

        out_parts.append(g)

    feat = pd.concat(out_parts, ignore_index=True)

    # Fill early-window NaNs (rolling warm-up) with column medians
    roll_cols = [
        "duty_hours_roll7", "duty_hours_roll30", "workload_spike_ratio",
        "workload_vs_baseline_pct", "high_intensity_days_14d", "rest_days_14d",
        "recovery_deficit_ratio", "wellness_roll14", "wellness_roll21",
        "wellness_deviation_z",
    ]
    for col in roll_cols:
        feat[col] = feat[col].fillna(feat[col].median())

    return feat


if __name__ == "__main__":
    raw = pd.read_csv(ARTIFACTS_DIR / "raw_dataset.csv", parse_dates=["date"])
    feat = engineer_features(raw)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "features.csv"
    feat.to_csv(out_path, index=False)
    print(f"Engineered features -> {out_path}  shape={feat.shape}")
    print(feat[[
        "soldier_id", "date", "workload_spike_ratio", "recovery_deficit_ratio",
        "wellness_deviation_z", "leave_utilization_rate"
    ]].tail(5).to_string())
