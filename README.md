# LabBox

> 本地优先、零侵入的 AI 实验记录与差异对比工具。
> Local-first, zero-instrumentation recorder and differ for AI training experiments.

```bash
labbox run --name baseline -- python train.py --epochs 20
labbox run --name no-amp  -- python train.py --epochs 20 --amp false
labbox compare baseline no-amp
```

> **Status:** `v0.1.0` · Windows 11 · MVP acceptance 10/10 (`python scripts/verify_mvp.py`)

LabBox 不替代 MLflow / W&B / DVC。它解决的问题只有一个：

> 昨天能跑通的实验，今天为什么跑不起来；两次实验之间到底改了什么；指标变化能不能被现有证据支持。

设计约束（完整规格见 [DEVELOPMENT.md](DEVELOPMENT.md)）：

- **Evidence before explanation**：规则引擎先确定事实，LLM 只能引用事实 ID 做解释，不能推断因果。
- **Never break training**：记录失败不得影响训练；训练进程的退出码原样返回。
- **Local-first**：所有数据写在项目内的 `.labbox/runs/`，默认不发出任何网络请求。

---

## 安装

**LabBox 不能装进你的训练环境。** 它需要从外部读取那个环境（见 §14.3 与
[ADR 0001](docs/adr/0001-separate-install-and-cross-interpreter-probe.md)）：

```bash
uv tool install labbox        # 或 pipx install labbox
```

Windows 上还有一个已实测的坑：**LabBox 本体必须安装在纯 ASCII 路径下**。
editable 安装写出的 `.pth` 文件由 `site.py` 以系统本地编码（中文 Windows 上是 cp936）解码，
如果安装路径含中文，`labbox.exe` 会以 `ModuleNotFoundError: No module named 'labbox'` 失败。
被记录的用户项目可以放在任意路径（含中文），不受影响。详见 §16.3。

开发环境：

```bash
uv venv && uv pip install -e ".[dev]"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1     # ruff + pyright + pytest + MVP 验收
```

## 命令

| 命令 | 作用 |
|---|---|
| `labbox init` | 创建 `.labbox/config.toml`，写入 `.gitignore`，输出本机能力报告 |
| `labbox --version` | 打印版本号，不需要已初始化的项目 |
| `labbox run --name NAME -- <命令>` | 运行并记录一次实验；退出码与原始命令一致 |
| `labbox list` / `labbox show RUN` | 查看与浏览已记录的 run |
| `labbox compare A B [--json] [--save]` | 输出结构化差异事实（F-xxx）与诊断 |
| `labbox report RUN` | 生成可离线打开的单文件 HTML 报告 |
| `labbox explain RUN [--against B] [--dry-run]` | 可选的 LLM 解释；`--dry-run` 只展示将要外发的数据类别 |
| `labbox doctor` | 检查 Git / GPU / psutil / 配置 / LLM 端点 |
| `labbox clean` | 查看与删除 run 目录（`--list`、`--dry-run`、`--keep-last`、`--older-than`） |

## 一个 run 目录里有什么

```text
.labbox/runs/20260922-153012-baseline/
├── run.json                 状态、时间、退出码（raw + 归一化）、索引链接
├── environment.json         OS / Python / torch 等版本 / CUDA / 驱动 / GPU / CPU / RAM / Git / 脱敏后的环境变量
├── command.txt              原始命令
├── metrics.json             最终值与最优值（缺失时明确写 metrics_missing，绝不猜）
├── dataset-manifest.json    数据 digest 与策略
├── config-values.json       超参数取值 + 每个值的来源
├── artifacts.json           checkpoint 路径/大小/hash（默认不复制、超大不 hash）
├── diagnostics.json         诊断 + 复现性 checklist
├── report.html              离线报告
├── reproduce.ps1 / .sh      只做校验和打印，绝不改动环境
├── logs/                    stdout.log（原始字节）、stdout.plain.log（去 ANSI）、stderr.log、events.jsonl
└── samples/system.jsonl     GPU 利用率/显存/温度/功耗 + CPU/RAM 时序（另有 system.csv）
```

JSON 文档的机器可读 schema 由代码生成，放在 [docs/schemas](docs/schemas/)。

## LabBox 不会做的事

这些"不做"是产品的一部分：

- 不声称 `fast_manifest` 的 digest 能证明数据内容不变（它只覆盖相对路径 + 大小 + 首尾 64 字节；
  mtime 被有意排除，理由见 [ADR 0002](docs/adr/0002-dataset-digest-excludes-mtime.md)）。
- 不从任意日志文本里"猜"指标。没有结构化指标就报 `METRICS_MISSING`。
- 不说"准确率下降是 X 造成的"。多个变量同时变化时，只说"无法单独归因"
  （[ADR 0003](docs/adr/0003-evidence-before-explanation.md)）。
- 不默认外发任何数据。LLM 层默认关闭，开启后也只发送结构化事实；`--dry-run` 可先看外发内容。
- 不复制大 checkpoint，不安装驱动，不改你的 venv。

## 环境要求

- Python ≥ 3.11（读取配置用标准库 `tomllib`）
- **当前发布与验证范围：Windows 11。** Linux 分支保留在代码中，但未验证、不在发布范围内
- 无 NVIDIA GPU 时全部核心功能可用（GPU 字段记为 unknown）
- 可选：`nvidia-smi` / `nvidia-ml-py` 用于 GPU 采集；`git` 用于代码状态

## 已知限制

1. **控制台中断（Ctrl+C）已在真实终端验证通过**（`scripts/spike_process.py` 16/16）：
   中断后 run 记为 `interrupted`、9 类文件全部落盘、退出码归一化为 130（原始值
   `0xC000013A`）。无控制台的环境（CI 容器）会自动跳过这两条用例并把它们报为 `SKIP`。
   过程中发现的两个平台陷阱（Ctrl+Break 会直接杀死进程、`CREATE_NEW_PROCESS_GROUP`
   会禁用该组的 Ctrl+C）记录在 §16.3-7，并已在代码里处理。
2. **脚本内设置的随机种子从外部不可见。** 零侵入下 `torch.manual_seed(7)` 无法被观测，
   LabBox 会把它标为 unknown 而不是猜测；用 `--seed 7` 这类命令行参数则能被记录。
3. `compare` 的配置取值只支持 YAML / JSON / TOML；Hydra 组合求值、INI、代码内默认值不在范围内。
4. **只在 Windows 上验证过。** POSIX 分支（`SIGINT` 进程组、`reproduce.sh`）保留在代码中，
   但当前版本不承诺 Linux/macOS 可用性，CI 也只跑 `windows-latest`。

## 测试

```bash
pytest -q                      # 全部（无 GPU 机器上 gpu 标记用 -m "not gpu" 跳过）
pytest -q -m "not gpu"
python scripts/verify_mvp.py   # 逐条核对 DEVELOPMENT.md §21 的 MVP 完成定义
```

## 许可

MIT，见 [LICENSE](LICENSE)。
