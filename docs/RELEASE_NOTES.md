# Public code preparation — 2026-11-16

This package extracts the LSTM/DQN software from the existing
`sweat_prediction_rl_project`. The original working project is unchanged.

## Included

- LSTM models, cross-validation, epoch selection, ensemble evaluation, and
  rolling forecast utilities.
- DQN, fixed-record surrogate environment, state construction, optional
  protocol-surrogate and feature/sensor ablation analyses.
- Python figure/report utilities and the original forecasting configurations.
- Deterministically generated synthetic Excel inputs, a small demonstration
  configuration, unit/integration tests, and a GitHub Actions test workflow.

## Packaging changes

1. All core stage defaults now use `configs/denoised_5min.yaml`. The original
   `configs/default.yaml` retains its ten-minute task and is selected explicitly.
2. `scripts/run_all.py` runs the five core stages; `--figures` adds basic plots.
   Local workbook-generation dependencies and Node.js are no longer required.
3. Package dependencies are declared in `pyproject.toml`; SciPy is explicit for
   optional statistical reports. The unused scikit-learn dependency is omitted.
4. DQN CSV horizon labels and observed-outcome flags follow the configured
   horizon, including the retained three- and ten-minute configurations.
5. Non-CPU device settings raise a clear error rather than being silently ignored.
6. Empty, non-finite, or unsorted input sheets raise a clear input error.
7. Participant demographic configuration and unrelated cumulative physiology
   analyses are excluded. No real measurements, derived experimental outputs,
   pretrained checkpoints, editor environments, or local dependency folders are
   included.
8. New bilingual documentation explains installation, input format, validation
   boundaries, synthetic data, and GitHub upload steps.

The training objective, candidate search, preprocessing equations, DQN
optimization, and surrogate reward/transition rules are preserved. These
packaging changes do not establish new scientific performance. Original
experimental results are not regenerated or distributed in this code release.

## License and citation

The owner has not selected an open-source license for this prepared package.
No license, author list, publication DOI, or citation metadata has been invented.
Choose and add the appropriate `LICENSE` before granting an open-source reuse
license, and add verified publication details when available.
