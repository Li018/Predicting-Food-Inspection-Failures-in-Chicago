# Topic 1 — execution checklist

Starting remote commit: `80f92b7c8a2f43cb2478ba15d0574a287fac670e` (`main`).

Only mark a step complete after its stated outputs have been generated and checked.
Each numbered step is committed and pushed independently. Existing cleaning,
point-in-time features, model selection, calibration and evaluation are reused.

- [x] **Step 1:** Inspect the latest remote code; create this checklist; keep data snapshots local and untracked.
- [x] **Step 2:** Full-data profile, annual drift, rule-change comparison, temporal split sizes and license linkage (`reports/metrics/data_summary.json`).
- [x] **Step 3:** Full-data logistic regression, validation selection, 2024 calibration and 2025 evaluation.
- [x] **Step 4:** Full-data LightGBM and held-out calibration/evaluation.
- [x] **Step 5:** Existing XGBoost/CatBoost implementations and a unified model comparison with confidence intervals.
- [x] **Step 6:** Publication-quality annual drift, ROC/PR and Top-K figures.
- [x] **Step 7:** Calibration, global SHAP and key-feature dependence figures.
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

### Step 4

LightGBM completed all six prespecified candidates (no warnings). Validation AP
is 0.378063; 2025 AUC/AP are 0.635730/0.321491. The selected class-balanced model
needs calibration: Brier improves 0.220831 to 0.168129 and LogLoss 0.631613 to
0.514865. Top-10% precision is 0.378734 (lift 1.679400). Ranking is unchanged by
calibration. Logistic regression remains the validation-selected model; 2025
descriptive performance does not replace the prespecified selection rule.

### Step 5

The comparison includes all three existing families, each raw and calibrated, with
95% entity-cluster intervals. XGBoost/CatBoost were not implemented in the starting
repository and were not added. The validation-selected primary model is calibrated
logistic regression. Paired 2025 LightGBM-minus-LR AP is +0.006925, with 95% CI
[-0.003442, 0.016447]; all five paired metric intervals include zero. The small
test advantage does not support declaring LightGBM superior.

### Step 6

Four 300-dpi figures now cover annual drift, ROC/PR, pooled Top-K, and daily
capacity/yield; underlying aggregate tables are saved. At nominal daily Top-10%,
rounding allocates 948 slots (11.34% of the observed cohort): primary LR finds
341 failures versus 214.22 expected under same-day random allocation (+59.18%).
This is conditional retrospective yield; no causal or days-earlier claim is made.
All charts were inspected for labels, clipping and metric consistency.

### Step 7

Added held-out calibration curves and primary-model global/dependence SHAP figures.
LightGBM mean probability falls from 44.9878% to 22.6518% after 2024 calibration
(2025 actual 22.5517%); LR falls from 24.7029% to 22.9894%. Exact LR-logit
SHAP uses all 27,396 training rows as reference, groups 100 transformed columns
back to 39 features, and samples 1,500 test rows for display. Affine calibration
scaling reconstructs all 8,363 observed test probabilities within 4.45e-16.
History duration, scored-day counts and failed-day counts dominate attribution;
correlated predictors can offset one another and are not causal effects.
