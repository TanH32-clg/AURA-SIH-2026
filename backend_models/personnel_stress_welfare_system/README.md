# AI-Based Predictive Personnel Stress & Welfare Monitoring System

A working reference implementation matching the architecture you specified:
dual data ingestion (org CSV + voluntary wellness form) → FastAPI backend
merging the streams and running four ML models with SHAP explainability →
role-based live dashboards (unit commander vs. welfare officer).

## What's included

| File | Purpose |
|---|---|
| `synthetic_data.py` | Generates a realistic synthetic dataset (180 soldiers × 90 days) matching every field you listed, so the pipeline is runnable/demoable without real personnel data. |
| `feature_engineering.py` | Builds the derived temporal features: workload spikes (7d/30d rolling), recovery deficit ratio, wellness trajectory deviation (14–21d z-score vs. personal baseline), leave utilization. Also handles the voluntary form's expected missingness (bounded forward-fill, then median). |
| `models.py` | Trains all **4 requested models**: Logistic Regression, Random Forest, XGBoost (auto-falls back to `HistGradientBoostingClassifier` if `xgboost` isn't installed), and Isolation Forest (unsupervised, trained only on wellness-trajectory features to catch anomalous patterns). |
| `risk_scoring.py` | Blends the 3 supervised probabilities (70%) with the Isolation Forest anomaly percentile (30%) into a 0–100 composite score + tier. Generates per-individual explanations via SHAP `TreeExplainer` (auto-falls back to a leave-one-out perturbation method if `shap` isn't installed). |
| `api.py` | FastAPI backend: `/ingest/hr-csv` (Stream A), `/ingest/wellness-form` (Stream B), `/score/recompute` (Step 2 processing), `/dashboard/commander/{unit_id}` and `/dashboard/welfare/high-risk` (Step 3, role-gated). |
| `dashboards/commander.html` | Bootstrap + Chart.js — aggregated, **anonymized** unit trends only. |
| `dashboards/welfare_officer.html` | Bootstrap — individual high-risk watchlist with AI reasons and recommended interventions, gated to the welfare-officer role. |

## Why 4 models, and what each is *for*

- **Logistic Regression** — fast, transparent baseline. Its coefficients are directly auditable, which matters when the output can trigger a welfare intervention: you can show *exactly* how each feature is weighted, not just a black-box score.
- **Random Forest** — captures nonlinear interactions (e.g., duty hours only matter combined with low rest interval) without much tuning; robust to outliers in the org data.
- **XGBoost** — typically the strongest raw discriminator on tabular data like this; used as the primary model for SHAP explanations shown to welfare officers.
- **Isolation Forest** — the only *unsupervised* model. It doesn't need a risk label at all — it flags soldiers whose wellness trajectory pattern is statistically unusual for them, which catches presentations the supervised models (trained on a necessarily incomplete definition of "risk") might miss. This is important because your supervised label will always be an imperfect proxy.

All three supervised models are trained on the identical feature set and split so you can compare them directly (`artifacts/metrics.json` after running).

## Running it

```bash
pip install -r requirements.txt

python synthetic_data.py        # -> artifacts/raw_dataset.csv
python feature_engineering.py   # -> artifacts/features.csv
python models.py                # -> artifacts/model_*.joblib, metrics.json
python risk_scoring.py          # -> artifacts/latest_risk_scores.csv (sanity check)

uvicorn api:app --reload        # backend on http://localhost:8000
```

Then open `dashboards/commander.html` and `dashboards/welfare_officer.html`
directly in a browser (they currently use inline demo data — swap the
`records`/`tierData` JS objects for `fetch()` calls to the FastAPI endpoints
once you're pointing at a live backend).

I ran the full pipeline in this sandbox (no `xgboost`/`shap`/`fastapi` — no
network access here) using the automatic fallbacks, end to end successfully:
LogReg/RF/GB all score ROC-AUC ≈ 0.97 on the held-out synthetic split, and
Isolation Forest's anomaly ranking correlates at ROC-AUC ≈ 0.82 with the
(synthetic) ground truth despite never seeing it during training. Install the
real `xgboost`/`shap` packages in your target environment for the intended
gradient-boosted model and true SHAP values.

## Replacing the synthetic label

Real deployments won't have a ground-truth "at risk" column. `models.py`'s
`build_label()` currently thresholds a hidden simulation variable for demo
purposes only. In production, replace it with real, ethically-sourced
outcome data — e.g., substantiated welfare-officer interventions, voluntary
clinical referrals, or validated survey instruments (PHQ-9/GAD-7-style,
administered under proper consent) — collected under clinical/ethics board
oversight, not just "did something bad happen to this person."

## Ethical & privacy safeguards to build in before deployment

This system infers psychological state from behavioral proxies for people
who can't easily opt out of the organization itself, so a few things are
worth deciding deliberately rather than by default:

- **Consent and purpose limitation for Stream B.** Keep the wellness form
  genuinely voluntary and make clear, in writing, that it will only be used
  for welfare check-ins — never performance reviews, deployment eligibility,
  or disciplinary action. If soldiers suspect otherwise, response rates and
  honesty both drop, and the fear itself becomes a stressor.
- **Hard separation between the commander and welfare-officer views** (which
  this build enforces at the API layer) — commanders should only ever see
  aggregated, anonymized trends for workload balancing, never individual
  scores or wellness answers.
- **Human-in-the-loop, not automated action.** The model should trigger an
  *offer* of a confidential check-in, never an automatic flag to a chain of
  command, restriction of duties, or weapons/security clearance — that
  decision should stay with trained welfare/medical personnel.
- **Audit logging** on every access to individually-identifiable risk data
  (who viewed which soldier's record, when), and a defined data-retention /
  deletion policy.
- **Bias and calibration review** across units, ranks, terrain postings, and
  tenure before trusting the score operationally — synthetic training data
  won't reveal real-world skew.
- **A clear escalation path that a soldier can trigger themselves** (e.g., a
  "request check-in" button) so the system augments rather than replaces
  people's ability to ask for help directly.

None of this is a blocker to building or piloting the system — it's the
same governance layer any serious occupational-health analytics platform
needs — but it's easier to design in from the start than retrofit.
