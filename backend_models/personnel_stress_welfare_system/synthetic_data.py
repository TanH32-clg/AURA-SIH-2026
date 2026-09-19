"""
synthetic_data.py
------------------
Generates a realistic (synthetic) longitudinal dataset that mimics:

  Stream A (Organizational / HR CSV upload):
      duty hours, overtime, consecutive duty days, night shift frequency,
      rest intervals, deployment duration, terrain/threat exposure,
      leave utilization, days since last leave, rejected leave requests,
      transfer frequency, relocation intervals.

  Stream B (Voluntary daily wellness form):
      sleep duration/quality, fatigue, energy, mood, perceived stress.
      Submission is OPTIONAL, so we simulate realistic missingness.

No real personnel data is used or required -- this is purely synthetic,
for pipeline development, testing and demonstration. Replace with the
real ingestion paths (CSV drop-zone / form POSTs) in production.

IMPORTANT (see README.md "Ethical & Privacy Safeguards"):
  - `soldier_id` here is a synthetic pseudonymous key. In production this
    should be a rotated/pseudonymous ID, never a name/service number,
    accessible in reversible form only to authorized welfare officers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RNG = np.random.default_rng(42)

TERRAINS = ["normal", "high_altitude", "hard_area"]
THREAT_LEVELS = ["low", "medium", "high"]


def _make_soldier_profiles(n_soldiers: int, n_units: int) -> pd.DataFrame:
    """Static/slow-moving attributes per soldier (Stream A, mostly)."""
    soldier_ids = [f"SLD-{i:05d}" for i in range(1, n_soldiers + 1)]
    unit_ids = RNG.integers(1, n_units + 1, size=n_soldiers)

    terrain = RNG.choice(TERRAINS, size=n_soldiers, p=[0.55, 0.25, 0.20])
    threat = RNG.choice(THREAT_LEVELS, size=n_soldiers, p=[0.4, 0.4, 0.2])

    # A latent "chronic strain" propensity per soldier drives correlated
    # patterns across duty load, sleep, mood, etc. This is what our models
    # are meant to (partially) recover from the observable features.
    latent_strain = RNG.beta(a=2.0, b=5.0, size=n_soldiers)  # skewed low, long tail

    transfers_last_2yrs = RNG.poisson(lam=1.0 + latent_strain * 2, size=n_soldiers)
    days_since_last_transfer = RNG.integers(15, 900, size=n_soldiers)
    deployment_duration_days = RNG.integers(10, 400, size=n_soldiers)
    leave_entitlement_annual = np.full(n_soldiers, 30)

    profiles = pd.DataFrame({
        "soldier_id": soldier_ids,
        "unit_id": [f"UNIT-{u:03d}" for u in unit_ids],
        "terrain_type": terrain,
        "threat_level": threat,
        "latent_strain": latent_strain,  # NOT exposed to models; used only to simulate labels
        "transfers_last_2yrs": transfers_last_2yrs,
        "days_since_last_transfer": days_since_last_transfer,
        "deployment_duration_days_start": deployment_duration_days,
        "leave_entitlement_annual": leave_entitlement_annual,
    })
    return profiles


def generate_dataset(
    n_soldiers: int = 180,
    n_units: int = 9,
    n_days: int = 90,
    form_response_rate: float = 0.62,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a long-format DataFrame: one row per (soldier_id, date).
    Wellness (Stream B) fields are NaN on days the soldier did not submit
    the voluntary form -- this missingness is intentional and must be
    handled downstream (see feature_engineering.py).
    """
    global RNG
    RNG = np.random.default_rng(seed)

    profiles = _make_soldier_profiles(n_soldiers, n_units)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

    rows = []
    for _, p in profiles.iterrows():
        strain = p["latent_strain"]
        consecutive_duty = 0
        days_since_leave = int(RNG.integers(5, 60))
        leave_used_ytd = int(RNG.integers(0, 8))
        rejected_leave_ytd = int(RNG.poisson(strain * 3))
        deployment_day0 = p["deployment_duration_days_start"]

        # Rolling wellness baseline per soldier, established in the first ~14 days
        base_sleep = RNG.normal(6.8 - strain * 1.6, 0.4)
        base_mood = RNG.normal(3.6 - strain * 1.2, 0.3)

        for day_idx, d in enumerate(dates):
            # --- Stream A: organizational / duty data (daily) ---
            overload_shock = 1.0 if RNG.random() < (0.08 + strain * 0.25) else 0.0
            duty_hours = np.clip(
                RNG.normal(8.5 + strain * 4.5 + overload_shock * 4, 1.5), 4, 18
            )
            night_shift = int(RNG.random() < (0.15 + strain * 0.35))
            is_rest_day = RNG.random() < max(0.05, 0.22 - strain * 0.15)

            if is_rest_day:
                consecutive_duty = 0
                rest_interval_hrs = float(RNG.normal(30, 6))
            else:
                consecutive_duty += 1
                rest_interval_hrs = float(np.clip(RNG.normal(11 - strain * 5, 2), 3, 24))

            weekly_overtime_hrs = np.clip(RNG.normal(6 + strain * 10, 3), 0, 40)

            if RNG.random() < 0.01:
                days_since_leave = 0
                leave_used_ytd += int(RNG.integers(1, 5))
            else:
                days_since_leave += 1

            if RNG.random() < (0.002 + strain * 0.01):
                rejected_leave_ytd += 1

            deployment_duration_days = int(deployment_day0 + day_idx)

            # --- Stream B: voluntary wellness form (may be missing) ---
            submitted = RNG.random() < form_response_rate
            if submitted:
                fatigue_pressure = strain * 2 + max(0, duty_hours - 10) * 0.15 + consecutive_duty * 0.03
                sleep_hours = float(np.clip(base_sleep - fatigue_pressure * 0.5 + RNG.normal(0, 0.6), 2.5, 9.5))
                sleep_quality = int(np.clip(round(5 - fatigue_pressure * 1.1 + RNG.normal(0, 0.7)), 1, 5))
                fatigue_level = int(np.clip(round(2 + fatigue_pressure * 1.3 + RNG.normal(0, 0.6)), 1, 5))
                energy_level = int(np.clip(round(6 - fatigue_level + RNG.normal(0, 0.5)), 1, 5))
                mood_score = int(np.clip(round(base_mood - fatigue_pressure * 0.6 + RNG.normal(0, 0.6)), 1, 5))
                stress_perception = int(np.clip(round(2 + fatigue_pressure * 1.2 + RNG.normal(0, 0.6)), 1, 5))
            else:
                sleep_hours = sleep_quality = fatigue_level = energy_level = mood_score = stress_perception = np.nan

            rows.append({
                "soldier_id": p["soldier_id"],
                "unit_id": p["unit_id"],
                "date": d,
                "terrain_type": p["terrain_type"],
                "threat_level": p["threat_level"],
                "duty_hours": round(duty_hours, 2),
                "weekly_overtime_hrs": round(weekly_overtime_hrs, 2),
                "consecutive_duty_days": consecutive_duty,
                "night_shift": night_shift,
                "rest_interval_hrs": round(rest_interval_hrs, 2),
                "is_rest_day": int(is_rest_day),
                "deployment_duration_days": deployment_duration_days,
                "leave_used_ytd": leave_used_ytd,
                "leave_entitlement_annual": int(p["leave_entitlement_annual"]),
                "days_since_last_leave": days_since_leave,
                "rejected_leave_requests_ytd": rejected_leave_ytd,
                "transfers_last_2yrs": int(p["transfers_last_2yrs"]),
                "days_since_last_transfer": int(p["days_since_last_transfer"]) + day_idx,
                "form_submitted": int(submitted),
                "sleep_hours": sleep_hours,
                "sleep_quality": sleep_quality,
                "fatigue_level": fatigue_level,
                "energy_level": energy_level,
                "mood_score": mood_score,
                "stress_perception": stress_perception,
                "_latent_strain": strain,  # hidden ground-truth driver, kept only for label synthesis
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "raw_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows for {df['soldier_id'].nunique()} soldiers "
          f"over {df['date'].nunique()} days -> {out_path}")
    print(df.head(3).to_string())
