# Local validation — 2026-11-16

Validated on Windows with Python 3.12.14 and CPU PyTorch. Direct dependency
versions are recorded in `requirements-tested.txt`.

| Check | Result |
| --- | --- |
| Unit, generator, integration, and regression tests | 19 passed |
| Extracted ZIP: synthetic stages 01–05 | Passed |
| Extracted ZIP: optional basic figure stage 06 | Passed |
| Wheel build with declared package metadata | Passed |
| Python syntax and local Markdown links | Passed |
| Git staging of four synthetic workbooks | Passed |
| Git exclusions for local data, checkpoints, environments, and outputs | Passed |
| Archive integrity and per-file SHA-256 manifest | Passed |

The integration test generates independent synthetic inputs in a temporary
directory and checks preprocessing, model checkpoints, Day 3 epoch-selection
metadata, Day 4 forecasts, filtered/raw metrics, 30-dimensional policy input,
31-dimensional reward state, and finite action scores. Regression checks cover
three- and ten-minute output labels and the fixed-record environment's behavior.

The extracted example run used `configs/synthetic_smoke.yaml --figures` with the
bundled synthetic records. This confirms software execution, not experimental
forecasting performance or physiological effects. Human experimental records
and pretrained experimental weights are not part of the release. Full-grid
experimental training and the optional report/ablation suite were not rerun.

The GitHub Actions workflow is included for future repository runs. Local
validation does not assert that a remote GitHub Actions run has occurred.
