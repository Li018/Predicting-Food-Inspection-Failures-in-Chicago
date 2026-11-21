# Topic 1 — execution checklist

Starting remote commit: `80f92b7c8a2f43cb2478ba15d0574a287fac670e` (`main`).

Only mark a step complete after its stated outputs have been generated and checked.
Each numbered step is committed and pushed independently. Existing cleaning,
point-in-time features, model selection, calibration and evaluation are reused.

- [x] **Step 1:** Inspect the latest remote code; create this checklist; keep data snapshots local and untracked.
- [x] **Step 2:** Full-data profile, annual drift, rule-change comparison, temporal split sizes and license linkage (`reports/metrics/data_summary.json`).
- [x] **Step 3:** Full-data logistic regression, validation selection, 2024 calibration and 2025 evaluation.
- [x] **Step 4:** Full-data LightGBM and held-out calibration/evaluation.
- [x] **Step 5:** All model families implemented in the starting repository, with a unified comparison and confidence intervals (XGBoost/CatBoost absent).
- [x] **Step 6:** Publication-quality annual drift, ROC/PR and Top-K figures.
- [x] **Step 7:** Calibration, global SHAP and key-feature dependence figures.
- [x] **Step 8:** High-score false positives, missed failures, true positives and local SHAP explanations.
- [x] **Step 9:** Evidence-based research and operational analysis (`reports/topic1_report.md`).
- [x] **Step 10:** Final README, reproduction command, full pytest, metric/link consistency and tracked-file checks.

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

### Step 8

Completed nine anonymous FP/FN/TP cases, three locally exact SHAP waterfalls,
full error-group profiles, subgroup metrics and frozen-model lag sensitivity.
At pooled Top-10%, LR yields TP/FP/FN/TN = 303/534/1,583/5,943; the p>=0.5
confusion is documented separately. Of missed failures, 68.86% last passed and
only 0.57% are cold starts; 34.83% of false positives pass with conditions.
Thirty-day lag changes LR AP from 0.314566 to 0.313942 and retains 96.18% of
Top-10% allocation weight, while the largest individual probability change is
12.55 percentage points. Correlated-feature contribution cancellation is measured
without causal or feature-removal claims. No identities or row-level predictions
are exported; all four new figures were visually inspected.

### Step 9

Completed the Chinese Topic 1 research report with all 39 feature definitions,
11 figures and traceable aggregate evidence. Independent data and model reviews
checked the split/linkage figures, selection and uncertainty, calibration, global
and local explanations, lag sensitivity and daily-capacity recommendations.
The report distinguishes event-time safeguards from unobservable publication
history, conditional ranking yield from prospective policy impact, and SHAP
associations from outcome causes. Final pytest and README/asset QA remain Step 10.

### Step 10

Finalized the illustrated README, reproduction entry point, report QA status and
a single authoritative progress log. Full pytest: 55 passed, one third-party
pytest-plugin deprecation warning. Delivery QA verifies six model rows and
78 confidence intervals, source/split/figure/SHAP/error/lag consistency, all
39 dictionary features, Markdown links, and 11 300-dpi PNGs. No raw data, Parquet
snapshots or fitted models are tracked at the current HEAD/pending delivery.
The four reused core modules remain byte-identical to the starting remote commit.
The 13-stage reproduction plan passed its actual dry run, including frozen source
hashes; full analysis stages were already executed individually during Steps 2–8.
The new wrapper was not represented as a second end-to-end training run.

## Incremental literature and presentation extension · 2026-11-21

### Commit 1 — Feature-group ablation and seasonality / Complaint check

Reviewed the existing core package, reproduction entry point, README and full report
before adding work. Existing annual drift, individual SHAP and error slices remain intact.
Added a shared four-group partition of all 39 fields, three nested LR ablations
(13 → 27 → 37 fields), and a separately loaded, never-refitted 39-field primary reference.
Each ablation selects among the original four LR candidates on 2023 AP, fits only
on 2019–2022, calibrates only on 2024, and reports 2025 AP / ROC-AUC / Brier.
Test calibrated AP: 0.310701 → 0.308815 → 0.311790; frozen primary AP 0.314566.
The non-monotonic AP result is retained rather than presented as an automatic gain.

Added exact-type Canvass versus Complaint annual, year-month and month-of-year
aggregates using recognized outcomes, same-label entity-day collapsing and conflict
exclusion. In 2019–2025, pooled monthly Fail-rate ranges are 6.79 and 7.94 percentage
points respectively; both peak in August and reach their minimum in December.
Equal-year summaries and the July 2018 break are separate descriptive diagnostics,
not causal effects or an exact replication of old-code critical-violation findings.

Validation before this commit: **63 passed, 1 warning** (all original 55 plus 8 new
regressions); Ruff and `git diff --check` passed. All 33 protected original files,
including frozen snapshots, fitted models, predictions, SHAP, core source, existing
tests and original figures, retain their initial SHA-256. No original model fit,
source manifest, reproduction command or Speaker Notes was changed.

### Commit 2 — Grouped SHAP, standardized odds ratios and subgroup audit

Loaded the original LR, analytic SHAP and predictions without fitting. Signed
per-observation group sums cover all 39 fields and reconstruct original probabilities
within 3.33e-16. Both the original fixed 1,500-case explanation sample and the full
8,363-case test set are reported. Numeric ORs distinguish raw and calibrated
coefficients per training SD; encoded coordinates have separately stated units.

Added historical Risk / Facility slices with AP, Brier, calibration bias, 10 fixed
equal-width-bin ECE and selection under the same global annual 837-slot allocation.
Risk 3 has 76 observations and is explicitly small-sample. The audit describes
operational disparities, not a guarantee that excluding sanitarian inputs removes
label or proxy bias. The new ECE convention is distinct from the old equal-frequency
calibration-plot diagnostic.

The official calendar-slot calculation is adapted to six fixed 2025 two-month
windows: mean Days Earlier 5.9526; 38.123% of Fail outcomes are delayed. Negative
advances remain in the mean and exact-score ties use uniform expected date slots;
per-day expected capacity error is zero. The output explicitly records both the
2019 time-invariance limitation and the use of actual-date features that might not
be available on a simulated earlier date. This is a retrospective reference.

Validation before this commit: **77 passed, 1 warning** (original 55 retained);
Ruff/new-file whitespace checks passed, protected 33-file baseline unchanged.
Added 14 tests for signed cancellation, OR units, common subgroup capacity,
calibration bins, negative advances, ties/window pooling and prediction alignment.

### Commit 3 — CDPH 2015 / Kannan 2019 report comparison

Incrementally extended README and the research report with measured ablation,
seasonality/comparison trends, signed four-group SHAP, standardized ORs, historical
Risk/Facility slices and fixed-window Days Earlier. Added a concise three-way
comparison of eras, targets, inspector inputs, evaluation methods and interpretation
limits. Prior 55-test statements now explicitly identify the frozen research baseline.
The original study results and all original figures remain intact.

Validation before this commit: **90 passed, 1 warning**; Ruff and whitespace checks
passed. Thirteen new public-artifact checks reconcile source hashes, primary-model
reference metrics, nested deltas, seasonal denominators, grouped cancellation, ORs,
subgroup calibration bins/capacity and positive-weighted reordering. Documentation
tables are checked against their CSV/JSON evidence. These checks work without the
private local row-level artifacts. All 33 protected baseline hashes remain unchanged.
