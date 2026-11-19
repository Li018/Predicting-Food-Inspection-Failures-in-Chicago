# Topic 1 — execution checklist

Starting remote commit: `80f92b7c8a2f43cb2478ba15d0574a287fac670e` (`main`).

Only mark a step complete after its stated outputs have been generated and checked.
Each numbered step is committed and pushed independently. Existing cleaning,
point-in-time features, model selection, calibration and evaluation are reused.

- [x] **Step 1:** Inspect the latest remote code; create this checklist; keep data snapshots local and untracked.
- [x] **Step 2:** Full-data profile, annual drift, rule-change comparison, temporal split sizes and license linkage (`reports/metrics/data_summary.json`).
- [x] **Step 3:** Full-data logistic regression, validation selection, 2024 calibration and 2025 evaluation.
- [ ] **Step 4:** Full-data LightGBM and held-out calibration/evaluation.
- [ ] **Step 5:** Existing XGBoost/CatBoost implementations and a unified model comparison with confidence intervals.
- [ ] **Step 6:** Publication-quality annual drift, ROC/PR and Top-K figures.
- [ ] **Step 7:** Calibration, global SHAP and key-feature dependence figures.
- [ ] **Step 8:** High-score false positives, missed failures, true positives and local SHAP explanations.
- [ ] **Step 9:** Evidence-based research and operational analysis (`reports/topic1_report.md`).
- [ ] **Step 10:** Final README, reproduction command, full pytest, metric/link consistency and tracked-file checks.

## Fixed experiment

- Target: a future routine/Canvass inspection fails, using information available before that inspection.
- Train: 2019–2022; validation: 2023; calibration: 2024; final test: 2025.
- Raw records, long text, model objects and full Parquet datasets stay local.
- Remote deliverables: lightweight code, aggregate CSV/JSON metrics, PNG figures and Markdown.
- No CI work or notebooks during these ten steps.

## Checkpoint notes

### Step 1

The remote contains implemented data preparation, conservative entity resolution,
strictly historical features, prevalence/logistic/LightGBM model families and held-out calibration.
The two data snapshots inherited from the remote have been retained locally and
removed from Git tracking; no history rewriting is performed. Numerical research
claims remain pending until the full-data runs complete.

### Step 2

Verified 304,348 inspection rows and 201,981 relevant license transactions from
1,185,058 downloaded license records. The modeling cohort contains 51,626 targets:
train 27,396; validation 7,277; calibration 8,590; test 8,363. Test Fail rate is
22.5517%; test point-in-time license availability is 79.5528%. The 2018 first/second
half Fail rates are 24.0205% / 19.8465%; this is descriptive, not causal.

### Step 3

Full-data prevalence and logistic regression completed with 500 establishment-cluster
bootstrap replicates. Logistic selection on 2023 AP chose C=0.1 without class
weighting (validation AP 0.378830). Its 2025 AUC/AP are 0.632637/0.314566;
calibration improves Brier 0.169531 to 0.168513 and LogLoss 0.517131 to 0.515382.
Top-10% precision is 0.362007 (lift 1.605231). No convergence warnings; model and
row-level predictions remain local. Added a thin per-family runner around existing modules.
