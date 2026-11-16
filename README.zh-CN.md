# e-SAFER：LSTM 预测与 DQN 运动强度建议演示

[English](README.md)

本项目整理 e-SAFER 的分钟级数据预处理、LSTM 多模型预测、预测特征构建和 DQN 运动强度建议流程。主配置采用**过去 5 分钟数据，预测未来 5 分钟**，预测通道为 `Na`、`K`、`Ca`、`TEMP` 和 `FR`。

公开包只包含代码和有明确标记的合成示例，不含人体原始实验数据、实验数据衍生表格和实验预训练权重。

> DQN 目前是固定记录代理环境中的软件演示。其结果不代表已验证的人体动作响应、独立策略泛化或人体闭环控制。具体边界见 [RL_LIMITATIONS.md](docs/RL_LIMITATIONS.md)。

## 用合成数据快速运行

在仓库根目录运行。需要 Python 3.10 或更高版本，当前实现使用 CPU。

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux 激活环境：

```bash
source .venv/bin/activate
```

安装并运行：

```bash
python -m pip install -e ".[dev]"
python scripts/run_all.py --config configs/synthetic_smoke.yaml
python -m pytest
```

`data/synthetic/` 已包含合成的 `Day1.xlsx`–`Day4.xlsx` 和来源标记 `SYNTHETIC_DATA.json`。如需重新生成，运行 `python scripts/generate_synthetic_data.py --force`；默认会保护已有文件。合成检查配置采用缩小的搜索空间、3 个训练轮次、2 个集成随机种子和 8 个 DQN episode，结果写入 `outputs_synthetic/`。

**该配置只检查程序能否运行，不能用其指标作为实验或论文结果。** 运行入口加 `--figures` 可生成基础图。

## 用自己的实验数据运行

将四个工作簿放入 `data/raw/`：

```text
data/raw/
  Day1.xlsx
  Day2.xlsx
  Day3.xlsx
  Day4.xlsx
```

每个通道对应一个工作表，第一列为秒数，第二列为数值。工作表名称、列要求和处理方式见 [数据格式说明](docs/DATA_FORMAT.md)。

```bash
python scripts/run_all.py
# 等价于显式指定主配置：
python scripts/run_all.py --config configs/denoised_5min.yaml
```

默认依次运行 01–05：预处理、LSTM 训练、Day 4 预测、RL 状态构建、DQN 训练及建议输出。`--figures` 增加 06 基础绘图。主流程只依赖 Python。已有前序结果时，可用 `--start-at N` 从指定步骤续跑，并保留与现有权重匹配的配置。

完整配置对每个预测目标搜索 192 组 LSTM 候选设置，每组执行 3 个按天划分的验证折，之后还要选择 persistence 融合权重并训练最终集成模型。因此，完整训练明显慢于合成数据检查。重复运行同一配置会更新其输出目录中的结果。

也可以逐步运行：

```bash
python scripts/01_preprocess.py --config configs/denoised_5min.yaml
python scripts/02_train_lstm.py --config configs/denoised_5min.yaml
python scripts/03_evaluate_day4.py --config configs/denoised_5min.yaml
python scripts/04_build_rl_state.py --config configs/denoised_5min.yaml
python scripts/05_train_dqn_demo.py --config configs/denoised_5min.yaml
python scripts/06_make_figures.py --config configs/denoised_5min.yaml
```

## LSTM 的训练与验证

- 原始通道先计算 1 分钟均值。主配置对五个传感通道使用因果 Holt 滤波，同时保留未滤波的分钟均值。
- 每个预测目标使用单层 LSTM，分别连接多步轨迹输出头和 3 维辅助摘要输出头。候选设置包括单通道/多通道、相对值/绝对值与相对值拼接、历史 HR、局部趋势跳连和 persistence 融合。
- 候选模型仅在 Days 1–3 内进行按天留一交叉验证。训练轮数通过 Days 1–2 训练、Day 3 验证确定，再使用选定轮数在 Days 1–3 上重新拟合。
- 主配置每个目标训练 5 个随机种子模型：`101`、`202`、`303`、`404`、`505`。Day 4 不参与 LSTM 模型选择和拟合。
- 固定起点指标使用第 10、20、30、40、50 分钟的预测；主配置还生成第 10–55 分钟逐分钟滚动预测。超出记录末尾的未来轨迹无法完整验证。

`metrics.json` 为预测相对于滤波后信号的回归指标；`metrics_against_raw.json` 为同一预测相对于未滤波分钟均值的指标。两者评价目标不同，不能混用。R² 是回归指标，不应写成百分比准确率。集成 SD 表示模型之间的离散程度，不是经过校准的置信区间。

## DQN 的输入与边界

主 Q 网络为 **30 → 64 → 32 → 3**，隐藏层采用 ReLU。五个传感通道各贡献 6 维：当前值、预测终点变化、预测平均变化、预测轨迹斜率、集成终点 SD、截至当前时刻的观测历史斜率。

HR 不进入 Q 网络的 30 维输入，而是保留在 31 维内部状态中参与奖励计算。三个动作依次为 `0 = 降低强度`、`1 = 保持强度`、`2 = 提高强度`。DQN 使用经验回放、目标网络和 epsilon-greedy 探索。

标准代理环境只对当前标准化 HR/TEMP/FR 施加预设偏移，下一次观测仍前进至下一条未修改的记录。奖励中的预测风险项由预测变化和 SD 构成，不随这些动作变化，因此该实现不能证明运动建议改善了未来电解质生理状态。

DQN 的标准化、HR 分位数、训练和结果报告使用同一条 Day 4 状态序列。**Day 4 的独立测试含义仅适用于 LSTM，不适用于 DQN。** Q 值经 softmax 得到的 policy score 是展示分数，不是经过校准的动作概率或置信度。

## 配置文件

| 文件 | 用途 | 输出目录 |
| --- | --- | --- |
| `synthetic_smoke.yaml` | 小规模合成数据全流程检查 | `outputs_synthetic/` |
| `denoised_5min.yaml` | 主版本：滤波后 5 分钟预测 | `outputs_denoised_5min/` |
| `denoised_3min.yaml` | 滤波后 3 分钟比较 | `outputs_denoised_3min/` |
| `denoised_10min.yaml` | 滤波后 10 分钟比较 | `outputs_denoised_10min/` |
| `short_horizon_3min.yaml` | 未滤波 3 分钟比较 | `outputs_3min/` |
| `default.yaml` | 保留的旧版未滤波 10 分钟任务 | `outputs/` |

`default.yaml` 是旧文件名，当前默认入口使用 `denoised_5min.yaml`。不同预测时长、滤波目标和输出目录应分别管理。

## 可选分析

主版本保留以下绘图、稳健性和消融脚本：

```bash
python scripts/07_make_prediction_decision_figures.py --config configs/denoised_5min.yaml
python scripts/08_make_project_schematic.py --config configs/denoised_5min.yaml
python scripts/08_make_robustness_figures.py --config configs/denoised_5min.yaml
python scripts/10_rl_feature_ablation.py --config configs/denoised_5min.yaml
python scripts/09_export_all_origin_data.py --config configs/denoised_5min.yaml
```

稳健性和消融脚本会重新训练 DQN。**运行 07 前，必须先完成 `denoised_3min.yaml`、`denoised_5min.yaml`、`denoised_10min.yaml` 三套配置**；其中的跨时长比较需要读取三套结果。Origin 导出应在需要导出的分析完成后运行。脚本内的 Fig 编号只用于区分输出，不保证与最终论文编号一致。

`11_protocol_informed_ablation.py` 和 `12_make_sensor_ablation_summary.py` 为可选的协议代理环境比较，采用额外假设，与标准固定记录环境分开。它们不属于默认流程，也不能作为已验证人体干预的证据。

## 目录与输出

```text
configs/                    主配置、比较配置和合成检查配置
data/raw/                   本地实验工作簿，默认不提交 Git
data/synthetic/             可公开的合成工作簿与来源标记
docs/                       数据格式、实现边界、GitHub 上传说明
scripts/                    流程入口、分步脚本和可选分析
src/sweat_rl/models/        LSTM 结构
src/sweat_rl/training/      模型选择、训练与预测
src/sweat_rl/rl/            状态构建、DQN、代理环境与消融
src/sweat_rl/visualization/ 绘图工具
tests/                      数值与流程检查
```

输出目录包含 `processed/`、`models/`、`training_history/`、`evaluation/`、`rl/`。重要结果有 `models/cv_summary.json`、`evaluation/metrics.json`、`evaluation/day4_rolling_forecasts.csv`、`rl/rl_state.csv`、`rl/dqn_rewards.csv` 和 `rl/decision_timeline.csv`。生成结果和模型权重默认不进入 Git。

LSTM 检查点包含标准化参数，加载时使用 `torch.load(..., weights_only=False)`。请使用本流程在本地生成或来源可信的检查点。

`FR` 沿用原代码的通道名；本代码不新增流量校准或单位换算。公开包未指定开源许可证，复用权限应由仓库所有者选定许可证后明确。作者、论文 DOI 和仓库地址未擅自填写。上传步骤见 [GitHub 上传说明](docs/GITHUB_UPLOAD.zh-CN.md)。
