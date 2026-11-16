# GitHub 上传说明

## 准备发布内容

本代码包已按“代码 + 合成示例”的范围整理。仓库根目录应包含 `README.md`、`README.zh-CN.md`、`pyproject.toml`、`configs/`、`src/`、`scripts/`、`tests/`、`docs/` 和 `data/synthetic/`。

实验原始工作簿、实验数据衍生结果、训练权重和本地虚拟环境不属于本次公开内容。`.gitignore` 用于避免后续将这些本地文件意外加入 Git。`data/synthetic/` 下的四个工作簿是可公开的合成示例，应连同来源标记一起保留。

本包未替你决定开源许可证。发布前，根据代码权属和机构要求选择合适许可证；仓库可见并不自动等于授予他人开源复用权限。README 中尚未提供作者、论文 DOI 或正式引用条目，待信息确定后再补充。

## 先验证运行

在仓库根目录创建并激活 Python 环境后运行：

```bash
python -m pip install -e ".[dev]"
python scripts/run_all.py --config configs/synthetic_smoke.yaml
python -m pytest
```

合成检查的结果目录 `outputs_synthetic/` 不需要上传。其指标不代表实验性能。默认真实数据流程为 `python scripts/run_all.py`，需要你自行在 `data/raw/` 提供实验文件。

## 通过 Git 上传

先在 GitHub 新建空仓库，例如命名为 `e-SAFER-LSTM-DQN`。为便于首次推送，可先不在网页中自动生成 README、许可证或 `.gitignore`。

在本地解压后的 `e-SAFER-LSTM-DQN` 目录打开终端：

```bash
git init
git branch -M main
git add .
git status --short
```

检查暂存列表：应包含源码和合成示例，不应包含 `.venv/`、`outputs*/`、`data/raw/Day*.xlsx` 或实验权重。确认后提交：

```bash
git commit -m "Add e-SAFER LSTM and DQN research code"
```

将下列地址中的 `YOUR_USERNAME` 替换为你的账号，并使用实际仓库名称：

```bash
git remote add origin https://github.com/YOUR_USERNAME/e-SAFER-LSTM-DQN.git
git push -u origin main
```

这些命令由你在自己的 GitHub 登录环境中执行。本次整理只生成本地发布包，没有替你创建 GitHub 仓库、提交代码或执行推送。

如果 GitHub 仓库已经有提交，先使用适合该仓库的正常克隆与合并流程，不要为覆盖既有内容而强制推送。

## 通过网页上传

也可以在 GitHub 仓库选择 **Add file → Upload files**，上传解压后的仓库内容并保留目录结构。上传的是仓库文件本身，而不是把整个压缩包作为唯一文件上传。网页操作同样应排除本地实验数据、结果目录和虚拟环境。

## 后续补充

正式论文信息确定后，可以补充作者与联系信息、论文标题和 DOI、引用格式、数据获取说明，以及经确认的实验复现条件。保留 README 中对固定记录代理环境、DQN 训练/报告范围、滤波指标和合成示例的说明。
