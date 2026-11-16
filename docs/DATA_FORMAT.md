# Data format

## Distributed examples and local data

`data/synthetic/` contains generated example workbooks and `SYNTHETIC_DATA.json`. These signals are synthetic and are intended only to exercise the software pipeline. They do not represent human observations or reproduce experimental performance.

Human experimental workbooks are not included. To run your own data with the main configuration, place four files in `data/raw/`:

```text
Day1.xlsx
Day2.xlsx
Day3.xlsx
Day4.xlsx
```

Choose the input directory through `paths.raw_dir` in the configuration. The synthetic configuration reads `data/synthetic/`, while experimental configurations read `data/raw/`. Relative paths are resolved against the repository root inferred from the configuration file's parent directory, so keep custom configurations directly in `configs/`.

## Worksheets

The supplied experimental configurations use this mapping:

| Workbook | Na | K | Ca | TEMP | FR | HR |
| --- | --- | --- | --- | --- | --- | --- |
| `Day1.xlsx` | `Na` | `K` | `Ca` | `TEMP-10` | `FR` | `HR-10` |
| `Day2.xlsx` | `Na` | `K` | `Ca` | `TEMP` | `FR` | `HR` |
| `Day3.xlsx` | `Na` | `K` | `Ca` | `TEMP` | `FR` | `HR` |
| `Day4.xlsx` | `Na10` | `K10` | `Ca10` | `TEMP10` | `FR` | `HR` |

Each worksheet must have a header row and at least two columns. The loader uses column positions rather than header names:

| Time_s | Value |
| --- | --- |
| 310 | 40.1 |
| 320 | 40.3 |
| 330 | 40.2 |

- The first column contains elapsed time in **seconds**, with ascending timestamps.
- The second column contains numeric signal values in consistent source units.
- Additional columns are ignored. Rows with nonnumeric time or value entries are removed.
- Supply adequate coverage for the complete analysis interval. Do not rely on interpolation to fill large gaps.
- The code does not convert sensor voltage to concentration, convert units, or calibrate the `FR` channel. Provide the appropriate calibrated values for your analysis and document their units separately.

Change `data.sheet_map` when your sheet names differ. Keep the internal channel identifiers `Na`, `K`, `Ca`, `TEMP`, `FR`, and `HR` unless adapting the code that consumes them.

## Minute aggregation and filtering

The main configuration processes minutes 6–55. Minute `m` is the average of available samples in the interval `((m − 1) × 60, m × 60]` seconds. For example, minute 6 uses samples after 300 s and through 360 s.

If a minute has no samples, the implementation interpolates at 10-second positions using `numpy.interp` and averages those positions. This fallback can use surrounding observations and extends endpoint values outside the measured range. Provide complete minute coverage when causal timing matters; the fallback is not a missing-data model.

Holt filtering is applied after aggregation. The standard settings are `alpha = 0.1`, with `beta = 0.1` for Na/K/TEMP/FR and `beta = 0.05` for Ca. Filtering proceeds forward in time. Original minute averages are retained as `Na_raw`, `K_raw`, `Ca_raw`, `TEMP_raw`, and `FR_raw`. HR is not filtered by the main configuration.

Each generated `processed/DayN.csv` includes `Minute`, the five model channels, `HR`, and retained raw columns when filtering is enabled.

## Day split and forecast windows

The workflow expects all four day files. Days 1–3 support LSTM selection and fitting; Day 4 supplies LSTM test forecasts. At origin `t`, the five-minute input consists of minutes `t − 4` through `t`, and the five-minute output predicts minutes `t + 1` through `t + 5`.

For the main configuration, training uses complete five-minute outcomes, fixed-origin evaluation uses origins 10/20/30/40/50, and rolling prediction/state construction covers origins 10–55. A prediction from origin 55 extends to minute 60, beyond the record ending at minute 55. Such forecasts can be generated, but their outcomes are unobserved.

The DQN uses the Day 4 derived state sequence for both fitting and reporting. This is separate from the independent Day 4 LSTM evaluation and does not constitute an independent policy test.

## Result interpretation

| File | Meaning |
| --- | --- |
| `evaluation/metrics.json` | Fixed-origin forecast metrics against the processed target; filtered in the main configuration |
| `evaluation/metrics_against_raw.json` | The same predictions evaluated against unfiltered minute averages, when retained |
| `evaluation/day4_trajectory.csv` | Fixed-origin predictions, ensemble SD, available actual values, and errors |
| `evaluation/day4_rolling_forecasts.csv` | Available rolling forecast/outcome pairs by target, origin, and lead time |
| `evaluation/rolling_ensemble_predictions.json` | Full generated rolling trajectories, including those beyond the observation window |
| `evaluation/day4_rolling_consensus.csv` | Aggregation of overlapping forecasts for display and associated metrics |
| `rl/rl_state.csv` | Origin minute, HR reward context, and forecast-derived sensor features |
| `rl/decision_timeline.csv` | Greedy DQN action, Q-values, display scores, and surrogate diagnostics |

Keep metrics against filtered and unfiltered targets separate. Ensemble SD and softmax Q display scores are not calibrated uncertainty statements. Synthetic metrics are software-check outputs only.

## 中文要点

四天数据各为一个 `.xlsx` 工作簿，每个通道单独一个工作表。第一列为秒数，第二列为数值，第一行为表头。主配置处理第 6–55 分钟，按分钟区间计算均值，再进行因果滤波。工作表名可在 `data.sheet_map` 修改。实验数据放在 `data/raw/`，合成检查读取 `data/synthetic/`；两者分别输出，不能将合成结果写成论文实验结果。
