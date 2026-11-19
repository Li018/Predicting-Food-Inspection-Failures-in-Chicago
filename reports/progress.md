# Verified implementation milestones

This log records completed work only. Experimental results are published after the
corresponding pipeline has executed; an implementation checkpoint is not a final result.

| Milestone | Evidence | State |
|---|---|---|
| Frozen research protocol and package skeleton | Commit `ece91db`; 2019–2022 / 2023 / 2024 / 2025 split | Saved |
| Restore remote state and reproducible Python environment | Python 3.12; exact dependency lock; imports verified | Saved in this checkpoint |
| Official data download and source manifest | In preparation | Pending |
| Point-in-time features and leakage regressions | In preparation | Pending |
| Probability/ranking metrics and cluster-bootstrap implementation | Six unit tests; constant-score and tie-order regressions | Saved in this checkpoint |
| Baselines, model selection, probability calibration | In preparation | Pending |
| Full experiments, SHAP, publication figures and report | Awaiting verified data and features | Pending |
| Independent review and complete deterministic rerun | Awaiting complete pipeline | Pending |

Every completed, verified milestone is committed and pushed to `main`. Large raw
downloads, temporary training artifacts, and the local environment remain ignored.

## Recovery note

The authoritative remote state at the beginning of this continuation was `ece91db`.
Earlier conversation mentioned additional local data and code, but those artifacts
were not present in the remote repository or the current checkout. They are being
reconstructed from the frozen protocol and official sources; no previous experiment
numbers are copied into the new results without running the corresponding analysis.
