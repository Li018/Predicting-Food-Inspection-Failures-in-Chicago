# Predicting Food Inspection Failures in Chicago

## Topic 1 — implementation in progress

This repository develops a reproducible prediction pipeline for the failure risk of a
business's next **routine (Canvass) food inspection**, using information available
before that inspection. The complete research protocol is frozen in
[`reports/research_protocol.md`](reports/research_protocol.md) before model fitting.

The implementation is being saved in verified milestones. Results will be populated
only after downloading the official data, running the experiments, and completing
independent leakage and reproducibility checks.

### Data sources

- [Chicago Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5)
- [Chicago Business Licenses](https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr)

### Fixed temporal protocol

| Role | Inspection dates | Permitted use |
|---|---|---|
| Historical context | 2010–2018 | Earlier history features only |
| Training | 2019–2022 | Fit preprocessing and estimators |
| Validation | 2023 | Choose hyperparameters and model family |
| Calibration | 2024 | Fit fixed sigmoid probability calibration |
| Final test | 2025 | One locked evaluation; never choose models |

All current-inspection outcomes, violations, facility updates, and risk updates are
excluded from model inputs. License tenure is an **observed license-age proxy**,
not a verified business opening date. The official dataset is a retrospective
snapshot; the research explicitly distinguishes event-time validity from unobserved
historical publication times.
