# Frozen research protocol — Topic 1

Protocol established before examining any model validation or test scores.
Configuration: [`config/topic1.json`](../config/topic1.json).

## 1. Decision and target

At the start of a candidate routine-inspection day, estimate the probability that the
next routine inspection will return **Fail**. Fail = 1; Pass and Pass w/ Conditions = 0.
Only explicitly recognized routine inspection types enter the target cohort;
Canvass Re-Inspection, complaints, licensing visits, and non-inspection outcomes do
not become routine targets. Earlier visits of any type may contribute to history.

Retrospective evaluation uses the actual routine inspection date as the scoring date.
It therefore evaluates ranking among inspected establishments, conditional on the
historical inspection policy. It does not establish performance for every uninspected
business or prove that future scheduling changes will preserve the same outcome.
The production scoring interface accepts an explicit date and supplied candidate list.

## 2. Information boundary

- Default history cutoff: scoring date minus one calendar day. Same-day records never
  inform a target, even if an inspection ID sorts before another ID.
- Event dates do not prove publication dates. The portal publishes retrospective
  snapshots without a complete historical availability log. Repeat with 7- and 30-day
  lags to quantify sensitivity; none of these settings proves historical availability.
- A current visit's violations, result, risk, facility type, and revised business
  attributes are forbidden as features. Only past observations provide mutable values.
- The license/address identifies the supplied candidate. Names, raw license numbers,
  target IDs, and addresses are not model inputs.
- Valid positive license number plus normalized address forms a conservative entity.
  Missing licenses use a separate name/address key; no fuzzy chain-store merging.
- Same-entity, same-day conflicting routine labels are excluded and counted; exact
  duplicate target events are collapsed deterministically.
- Past inspection results in the test year may update later test-day histories. This
  is a fixed-model rolling operational evaluation, not an attempt to forecast the
  whole year from January 1 with frozen covariates.

## 3. License semantics

Use `license_number`, not internal transaction `license_id`, to join inspections.
Only records issued by the observation cutoff and started by the scoring date can
contribute. The earliest eligible term start supplies observed license tenure;
renewal terms are not opening dates. The 2002 source boundary causes left truncation.
Address agreement prevents transferring tenure across unrelated sites.

Current final `license_status` is not copied backwards. The main model uses the
recorded term interval (in term / term expired / end unknown), explicitly distinct
from verified legal active status. Historical addresses can also have been corrected
retrospectively; this unavoidable source limitation is documented.

## 4. Splitting, selection, calibration

Training 2019–2022; validation 2023; calibration 2024; final test 2025.
2010–2018 supplies history only. This keeps model fitting after the documented
July 2018 inspection procedure change while retaining clearly labeled historical
outcomes as predictors. The COVID period and annual distribution changes are reported.

Fit all preprocessing on the training fold only. Compare a prevalence baseline,
regularized logistic regression, and a prespecified finite LightGBM search including
unweighted and class-balanced candidates. Use validation average precision as the
primary selection metric. Keep the selected estimator trained on 2019–2022; do not
refit or change parameters after looking at calibration or test outcomes. Fit sigmoid
calibration on 2024 only. Test reports raw and calibrated probabilities for transparency.

No random train/test split, oversampling before splitting, target encoding on all
data, or feature selection from test outcomes is allowed. Dataset filtering, joins,
and imputation must not use future observations to repair earlier records.

## 5. Evaluation and interpretation

Report average precision (AP; stepwise PR-area convention), ROC-AUC, Brier score,
log loss, calibration, Precision/Recall/Lift at top 5%, 10%, and 20%. Distinguish pooled
annual ranking from per-day capacity-constrained ranking; only the latter resembles
an operational daily queue. Bootstrap by establishment entity for uncertainty.
Report sample size, failure prevalence, cold-start coverage, and subgroup limitations.

SHAP describes model associations in the model's raw output space; it does not
identify causes of food safety failures or prove policy effects. Inspect global and
local attributions, feature dependence, confident false positives/negatives, and
observed license-history availability. Business recommendations remain conditional
on measured ranking performance and prospective validation.

## 6. Reproducibility and acceptance

Use fixed queries through 2025-12-31, explicit source manifests with SHA-256 hashes,
deterministic CPU models, recorded package versions, and fixed random seeds. Downloads
are cached and large raw files ignored by Git. Any committed compressed snapshot must
stay well below GitHub's per-file limit. A future live download can change if the city
corrects source rows; distinguish same-snapshot reproducibility from live refresh.

Before final delivery: targeted unit and leakage tests pass, the complete pipeline
executes, independent rerun reproduces deterministic numeric outputs, all figures are
visually checked, and verified milestones are committed and pushed to this repository.

## Official references

1. [Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5)
2. [Business Licenses](https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr)
3. [Post-July-2018 official filtered view](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections-7-1-2018-Present/qizy-d2wf)
