# e-SAFER: LSTM Forecasting and DQN Recommendation Demo

[中文说明](README.zh-CN.md)

Research code for the e-SAFER forecasting and exercise-intensity recommendation workflow: minute-level sensor preprocessing, per-channel LSTM ensembles, forecast-derived state construction, and a DQN policy demonstration in a fixed-record surrogate environment.

The main configuration uses **5 minutes of history to forecast the next 5 minutes** for `Na`, `K`, `Ca`, `TEMP`, and `FR`. The public package contains source code and clearly labelled synthetic examples. Human experimental data, experiment-derived tables, and pretrained experimental checkpoints are excluded.

> The DQN component demonstrates software behaviour in a surrogate environment. It does not establish physiological action effects, independent policy generalization, or validated human closed-loop control. See [the implementation limits](docs/RL_LIMITATIONS.md).

## Quick start with synthetic data

Run commands from the repository root. Use Python 3.10 or later; the current implementation runs on CPU.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install -e ".[dev]"
python scripts/run_all.py --config configs/synthetic_smoke.yaml
python -m pytest
```

The included `data/synthetic/Day1.xlsx`–`Day4.xlsx` are generated examples, accompanied by `SYNTHETIC_DATA.json`. To regenerate them, run `python scripts/generate_synthetic_data.py --force`; existing files are protected unless `--force` is supplied. The smoke configuration uses a reduced search grid, 3 training epochs, two ensemble seeds, and 8 DQN episodes. Its output in `outputs_synthetic/` checks that the workflow runs; **its metrics are not experimental or manuscript results**. Add `--figures` to the runner to generate the basic plots.

## Run with your own experimental data

Place `Day1.xlsx`, `Day2.xlsx`, `Day3.xlsx`, and `Day4.xlsx` in `data/raw/`. Each signal occupies a separate worksheet with time in seconds in the first column and its numeric value in the second column. See the [data format and worksheet mapping](docs/DATA_FORMAT.md).

```bash
python scripts/run_all.py
# Equivalent explicit configuration:
python scripts/run_all.py --config configs/denoised_5min.yaml
```

The runner executes stages 01–05: preprocessing, LSTM training, Day 4 forecasting, state construction, and DQN training/reporting. `--figures` adds stage 06. The core workflow requires Python only. Use `--start-at N` to resume at an existing stage with the matching configuration and checkpoints.

The full configuration searches 192 LSTM candidates per target across three day-wise folds, before persistence-blend selection and final ensemble training. This is substantially longer than the synthetic smoke run. Each run writes to the configuration's output directory and can replace files from an earlier run with that configuration.

Individual stages can also be run separately:

```bash
python scripts/01_preprocess.py --config configs/denoised_5min.yaml
python scripts/02_train_lstm.py --config configs/denoised_5min.yaml
python scripts/03_evaluate_day4.py --config configs/denoised_5min.yaml
python scripts/04_build_rl_state.py --config configs/denoised_5min.yaml
python scripts/05_train_dqn_demo.py --config configs/denoised_5min.yaml
python scripts/06_make_figures.py --config configs/denoised_5min.yaml
```

## Models and evaluation

### LSTM forecasting

- Signals are aligned to 1-minute averages. The main configuration applies causal Holt filtering to the five sensor channels and retains the unfiltered minute averages for separate evaluation.
- A single-layer LSTM is trained for each target, with a direct multi-step trajectory head and an auxiliary three-value summary head. Candidate settings include single-channel/multichannel context, relative/absolute-relative representations, optional historical HR, local-trend skip, and persistence blending.
- Candidate selection uses leave-one-day-out cross-validation within Days 1–3. Epoch selection fits Days 1–2 and validates on Day 3; the selected epoch count is then used to refit on Days 1–3.
- The main ensemble uses seeds `101`, `202`, `303`, `404`, and `505`. Day 4 is held out from LSTM model selection and fitting.
- Fixed-origin metrics use origins 10, 20, 30, 40, and 50 min. The main configuration also generates forecasts at every minute from 10 through 55. Forecasts extending beyond the available record are not fully evaluable.

`R²`, RMSE, and MAE are regression metrics. `metrics.json` evaluates predictions against the filtered signal; `metrics_against_raw.json` evaluates the same predictions against unfiltered minute averages. These are distinct evaluation targets. Ensemble SD describes variation among fitted members; it is not a calibrated confidence interval.

### DQN policy demonstration

The main Q-network has architecture **30 → 64 → 32 → 3**, with ReLU hidden activations. Each of the five sensor channels contributes six observations:

1. Current value.
2. Predicted endpoint change.
3. Predicted mean change.
4. Predicted trajectory slope.
5. Ensemble endpoint SD.
6. Slope of the observed history through the current minute.

HR is excluded from the Q-network's 30 inputs and retained in the 31-dimensional internal reward state. Actions are `0 = reduce`, `1 = maintain`, and `2 = increase` intensity. Training uses replay memory, a target network, and epsilon-greedy exploration.

In the standard surrogate, action effects are configurable offsets to the current standardized HR/TEMP/FR values. The next observation advances to the next unchanged recorded state. The forecast-risk term uses predicted changes and SD, which these actions do not alter. Thus, rewards in this implementation do not demonstrate that actions improve future electrolyte physiology.

The DQN is trained and reported on the same Day 4 state sequence, including sequence-wide normalization and HR quantiles. Day 4 is an independent test day for the **LSTM only**, not an independent DQN evaluation. Softmax-transformed Q-values are display scores, not calibrated action probabilities or confidence estimates.

## Configuration files

| Configuration | Purpose | Output directory |
| --- | --- | --- |
| `synthetic_smoke.yaml` | Small synthetic end-to-end check | `outputs_synthetic/` |
| `denoised_5min.yaml` | Main filtered 5-minute workflow | `outputs_denoised_5min/` |
| `denoised_3min.yaml` | Filtered 3-minute comparison | `outputs_denoised_3min/` |
| `denoised_10min.yaml` | Filtered 10-minute comparison | `outputs_denoised_10min/` |
| `short_horizon_3min.yaml` | Unfiltered 3-minute comparison | `outputs_3min/` |
| `default.yaml` | Legacy unfiltered 10-minute workflow | `outputs/` |

The legacy file name `default.yaml` does not indicate the runner's default. The default run uses `denoised_5min.yaml`. Keep forecast horizons, filtering targets, and output directories distinct when comparing results.

## Optional analyses

For the main five-minute experimental workflow, the retained analysis scripts include:

```bash
python scripts/07_make_prediction_decision_figures.py --config configs/denoised_5min.yaml
python scripts/08_make_project_schematic.py --config configs/denoised_5min.yaml
python scripts/08_make_robustness_figures.py --config configs/denoised_5min.yaml
python scripts/10_rl_feature_ablation.py --config configs/denoised_5min.yaml
python scripts/09_export_all_origin_data.py --config configs/denoised_5min.yaml
```

The robustness and ablation scripts retrain DQNs and can take additional time. **Before running stage 07, complete all three configurations `denoised_3min.yaml`, `denoised_5min.yaml`, and `denoised_10min.yaml`**: its horizon-comparison panel requires all three output directories. Run the Origin-data export after the analyses whose tables you want to export. Internal figure numbers are script identifiers and need not match a manuscript's final figure numbering.

`11_protocol_informed_ablation.py` and `12_make_sensor_ablation_summary.py` retain an optional protocol-informed surrogate comparison. It uses additional modelling assumptions and is separate from the standard fixed-record demo. It is not part of the default pipeline or evidence of a validated intervention.

## Repository layout

```text
configs/                    Main, comparison, and smoke configurations
data/raw/                   Your local experimental workbooks (excluded from Git)
data/synthetic/             Public synthetic workbooks and provenance marker
docs/                       Data format, limitations, and GitHub upload guide
scripts/                    Pipeline stages, runner, and optional analyses
src/sweat_rl/
  models/                   LSTM architecture
  training/                 LSTM selection, fitting, and forecasting
  rl/                       State builder, DQN, environments, and ablations
  visualization/            Plotting helpers
tests/                      Numerical and workflow checks
```

Generated output directories contain `processed/`, `models/`, `training_history/`, `evaluation/`, and `rl/`. Key files include `models/cv_summary.json`, `evaluation/metrics.json`, `evaluation/day4_rolling_forecasts.csv`, `rl/rl_state.csv`, `rl/dqn_rewards.csv`, and `rl/decision_timeline.csv`. Generated outputs and checkpoints are excluded from Git by default.

LSTM checkpoints contain scaler metadata and are loaded with `torch.load(..., weights_only=False)`. Use checkpoints generated locally by this pipeline or otherwise trusted files.

## Data and reuse

No human experimental data or pretrained experimental models are distributed here. Synthetic examples serve software testing only. `FR` is the original code's channel identifier; the software performs no new flow-rate calibration or unit conversion.

No reuse license is assigned in this package. The repository owner should choose an appropriate license before granting reuse rights. Author names, a paper DOI, and repository URLs have not been invented. See the [Chinese GitHub upload guide](docs/GITHUB_UPLOAD.zh-CN.md) for publishing steps.
