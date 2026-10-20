# LabBox 开发文档

> Local-first, zero-instrumentation AI experiment debugger.<br>
> 本文档定义 LabBox MVP 的产品边界、技术架构、数据协议、开发顺序和验收标准。

- 文档版本：0.2.1
- 状态：定稿（MVP 范围与协议锁定；发布验证范围 = Windows）
- 更新日期：2026-10-20
- v0.2.1 变更：发布范围收敛为 Windows（Linux 分支保留但不验收）；CI 改为 `windows-latest`；补 `labbox --version`；移除空的根 `templates/` 与过时 sdist artifacts 配置
- v0.2 变更：数据集 digest 定义剔除 mtime；新增安装与解释器模型（14.3）；禁止采集器初始化 CUDA context；新增配置取值提取（9.6）与无指标场景约束（2.4）；Windows 中断与退出码语义具体化（6.3）；`reproducibility_score` 改为 checklist + band（12.1）；补 `labbox clean` 与存储回收（4.7）；启动性能指标区分冷热启动（17）；22 节中阻塞 M0 的问题已定稿并移出
- 项目名：LabBox（暂定）
- 核心命令：`labbox run -- python train.py`

---

## 1. 项目定位

LabBox 是一个面向个人开发者、学生实验室和小型研究团队的本地优先实验记录与诊断工具。

它不试图替代 MLflow、W&B 或 DVC，而是解决以下高频问题：

1. 昨天能跑通的实验，今天为什么跑不起来？
2. 两次实验之间到底发生了什么变化？
3. 指标变化能否由当前记录支持？
4. 哪个实验值得复现，哪个实验缺少复现条件？
5. 如何在不修改或少修改训练脚本的情况下完成记录？

### 1.1 核心价值

> 不是“再做一个实验看板”，而是“找出实验差异，并解释为什么结果不可信或不可复现”。

### 1.2 核心原则

- **Local-first**：默认不依赖云服务，所有数据保存在本地。
- **Zero-instrumentation first**：基础功能只需包裹训练命令，不要求用户先接入 SDK。
- **Evidence before explanation**：规则引擎先确定事实，LLM 只能基于事实解释，不能凭空推断因果。
- **Never break training**：记录失败不能导致训练失败；子进程退出码必须被完整保留。
- **Windows-first**：当前发布与验收只覆盖 Windows + NVIDIA GPU。POSIX 分支保留在代码中，但不在发布验证范围内，也不作为兼容性承诺。
- **Privacy by default**：环境变量、日志和路径必须经过脱敏；默认不上传任何数据。
- **Reproducible or explicit**：无法证明的复现条件要明确标记为未知，而不是伪装成完整复现。

---

## 2. MVP 范围

### 2.1 必须实现

- `labbox init`
- `labbox run -- <command>`
- 启动、监控并原样转发训练进程输出
- 记录子进程退出码、开始时间、结束时间和总耗时
- 记录 Git、Python、PyTorch、CUDA、驱动、GPU、CPU、RAM 和主要 pip 包
- 记录训练命令、环境变量、随机种子、配置文件和数据版本
- 从 YAML/JSON/TOML 配置文件中提取超参数值并与命令行覆盖合并（见 9.6）
- 采集 stdout/stderr
- 采集基础 GPU 指标：利用率、显存占用、温度
- 收集 checkpoint 和其他 artifact 的路径、大小与 hash
- 生成 `run.json`、`environment.json`、`command.txt`、`metrics.json`、`dataset-manifest.json`、`config-values.json`
- 生成 `reproduce.ps1` 和 `reproduce.sh`
- `labbox compare <run-a> <run-b>`
- `labbox report <run-id>` 生成静态 HTML 报告
- 可选的 LLM 解释层，通过 OpenAI-compatible API 调用本地或远程模型
- 对失败的训练运行也生成尽可能完整的记录
### 2.2 暂不实现

- 多人协作、账号系统和权限系统
- 云同步、远程存储和模型注册中心
- 自动复制大型 checkpoint
- 完整实验调度、任务队列和超参数搜索
- Kubernetes/Slurm 集成
- 浏览器实时训练看板
- 自动证明 bitwise reproducibility
- 对任意第三方日志进行高精度指标抽取
- 训练框架级别的深度 hook

### 2.3 MVP 成功标准

用户只需执行：

```bash
labbox run --name baseline -- python train.py --config configs/base.yaml
```

即可得到完整实验目录；运行第二条命令后：

```bash
labbox compare baseline yolo-bs128
```

能准确列出代码、环境、数据、参数、随机性和硬件差异，并明确区分：

- 已知事实
- 推断
- 缺失证据
- 无法判断的因果关系

### 2.4 两条硬约束（优先级高于任何功能）

1. **无指标也必须可用**：零侵入场景下训练脚本往往不输出结构化指标。此时 `compare` 仍须完整输出代码/环境/硬件/命令/数据差异，并在顶部显式标注 `METRICS_MISSING`，禁止把"无指标"渲染成"指标相同"。
2. **行为与裸命令一致**：`labbox run -- X` 的退出码、stdout 内容、stderr 内容与直接执行 `X` 一致；记录失败只能追加警告，不能改变训练结果。

---

## 3. 目标用户和使用场景

### 3.1 目标用户

- 拥有本地 NVIDIA GPU 的个人开发者
- 需要频繁做对照实验的本科生和研究生
- 管理少量训练任务的小型研究团队
- 希望保留本地数据主权、又需要实验解释能力的用户

### 3.2 主要场景

#### 场景 A：复现失败诊断

用户运行相同命令，但结果差异较大。LabBox 对比两次实验，发现：

- Git commit 不同
- Torch 版本变化
- CUDA 驱动变化
- 数据集 digest 不同
- `torch.use_deterministic_algorithms` 未设置

#### 场景 B：控制变量检查

用户比较 Run #17 和 Run #18，LabBox 指出唯一显著变化是 batch size 与 AMP 开关，并提示这两个变量同时变化，无法单独归因。

#### 场景 C：训练异常解释

某次运行吞吐骤降。LabBox 关联 GPU 利用率、温度、显存和主机内存曲线，给出可能原因和原始证据。

#### 场景 D：实验归档

用户完成实验后，LabBox 生成可浏览的报告和可移植的复现说明，并标注哪些条件已记录、哪些条件未知。

---

## 4. CLI 设计

### 4.1 命令总览

```text
labbox init
labbox run [OPTIONS] -- COMMAND [ARGS...]
labbox list
labbox show RUN
labbox compare RUN_A RUN_B
labbox report RUN
labbox explain RUN [--against RUN_B]
labbox doctor
labbox clean
```

### 4.2 `labbox init`

初始化项目。

行为：

- 创建 `.labbox/config.toml`
- 将 `.labbox/runs/` 加入 `.gitignore`，或提示用户确认
- 检测 Git、Python、NVIDIA 工具和 shell 环境
- 输出能力报告，不因可选能力缺失而失败

### 4.3 `labbox run`

示例：

```bash
labbox run --name baseline -- python train.py --epochs 20
labbox run --name no-amp --tag ablation -- python train.py --amp false
labbox run --repeat baseline -- python train.py
```

主要参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--name` | 可读运行名 | 自动生成 |
| `--tag` | 可重复标签 | 无 |
| `--project` | 项目名 | Git 仓库名 |
| `--dataset` | 数据集路径或版本标识 | 自动探测 |
| `--metric-file` | 训练脚本输出的 JSON/JSONL 指标路径 | 无 |
| `--sample-interval` | 系统/GPU 采样间隔（秒） | `2.0` |
| `--no-gpu` | 禁用 GPU 采集 | `false` |
| `--no-network` | 禁止任何远端解释请求 | `false` |
| `--redact-env` | 额外脱敏的变量名正则 | 无 |
| `--copy-artifacts` | 将 artifact 复制到运行目录 | `false` |

退出行为：

- 训练成功：LabBox 返回 `0`
- 训练失败：LabBox 必须返回训练进程原始退出码
- LabBox 自身记录失败：默认显示警告，但不覆盖训练退出码
- 用户 Ctrl+C：向子进程转发中断，完成收尾后返回相应退出码

### 4.4 `labbox compare`

输出应包含：

- 代码差异
- 环境差异
- 硬件差异
- 命令差异
- 参数和配置差异
- 数据版本差异
- 随机性设置差异
- 指标差异
- 复现风险差异
- 可引用的 fact ID

示例输出：

```text
Changed facts
  [F-001] git.commit         a1b2c3d -> d4e5f6a
  [F-002] dataset.digest     sha256:91ab... -> sha256:7c22...
  [F-003] train.batch_size   64 -> 128
  [F-004] train.amp          true -> false

Diagnostics
  [HIGH] 数据与 batch size 同时变化，无法单独归因。
  [HIGH] dataset.digest 变化，结果不具备直接可比性。
  [INFO] CUDA、PyTorch 和 GPU 型号未变化。

Metrics
  accuracy                   84.21% -> 80.43%   -3.78 pp
```

### 4.5 `labbox explain`

LLM 输入只能包含经过清洗的结构化事实，不允许直接把任意日志或完整环境变量发送给模型。

输出必须：

- 引用事实 ID
- 区分事实、推断和建议
- 明确列出证据不足项
- 不允许声称未提供的因果关系已经成立

### 4.6 `labbox doctor`

检查：

- Python 和 LabBox 版本
- Git 是否可用
- CUDA / NVIDIA driver 是否可读取
- GPU 采样能力
- 当前目录是否已初始化
- 配置文件是否合法
- LLM endpoint 配置是否有效，但默认不发送请求

### 4.7 `labbox clean`

管理本地存储与敏感数据。run 目录含日志和环境快照，属于**必须可清理**的数据。

```text
labbox clean --list                 # 按大小排序列出所有 run 及总占用
labbox clean <run>                  # 删除单个 run（默认二次确认）
labbox clean --keep-last 20         # 仅保留最近 20 个 run
labbox clean --older-than 90d       # 按时间清理
labbox clean --all --yes            # 清空 .labbox/runs
```

规则：

- 所有删除默认需要确认，`--yes` 才能跳过；`--dry-run` 只打印将被删除的路径。
- 只允许删除 `.labbox/runs/` 下的目录，路径必须先规范化再校验前缀，禁止穿越。
- `.labbox/config.toml` 中 `store.max_total_bytes` 超限时，`labbox run` 结束时提示（不自动删除）。
- 删除失败（文件被占用）必须报告具体文件，不静默跳过。

---

## 5. 总体架构

```text
CLI
 └── Application Services
      ├── RunService
      ├── CompareService
      ├── ReportService
      └── ExplainService
           │
           ├── Domain
           │    ├── RunRecord
           │    ├── DiffFact
           │    ├── Diagnostic
           │    └── MetricDelta
           │
           ├── Recorder
           │    ├── ProcessRunner
           │    ├── EnvironmentCollector
           │    ├── HardwareSampler
           │    ├── DatasetManifestBuilder
           │    ├── MetricsIngestor
           │    └── ArtifactCollector
           │
           ├── Analysis
           │    ├── DiffEngine
           │    ├── RuleEngine
           │    └── ReproducibilityScorer
           │
           └── Infrastructure
                ├── FilesystemStore
                ├── GitClient
                ├── NvidiaClient
                ├── ShellGenerator
                └── LLMProvider
```

### 5.1 分层约束

- `domain` 不依赖 CLI、文件系统或第三方 API。
- `analysis` 只消费规范化 domain 对象。
- `infrastructure` 负责外部命令、文件和网络。
- `cli` 只负责参数解析、调用 service 和展示结果。
- LLM 不能直接访问文件系统、Git 或环境；它只能接收 `ExplainContext`。

---

## 6. 运行生命周期

```text
1. 解析 CLI 与配置
2. 创建 run_id 和临时目录
3. 写入 run.started 事件
4. 采集运行前环境快照
5. 构建命令、Git、数据集和配置清单
6. 启动训练子进程
7. 持续转发 stdout/stderr
8. 按固定间隔采集 CPU/RAM/GPU 指标
9. 训练结束或被中断
10. 采集最终环境与硬件状态
11. 收集 metrics 和 artifacts
12. 运行 refactor/diff/reproducibility rules
13. 原子写入所有 JSON
14. 生成 reproduce 脚本
15. 生成 HTML 报告
16. 返回训练进程退出码
```

### 6.1 失败策略

每项采集必须允许独立失败：

| 组件 | 失败行为 |
|---|---|
| GPU 采集 | 记录 `unavailable`，继续训练 |
| Git 采集 | 记录 `not_a_repository`，继续 |
| 数据集 hash | 记录错误和部分 manifest，继续 |
| metrics 文件 | 标记缺失，不创建虚假指标 |
| artifact 收集 | 记录路径和错误，继续 |
| 报告生成 | 保留所有原始记录，提示报告失败 |
| LLM 解释 | 不影响 run/compare/report 的原始结果 |

### 6.2 原子写入

`run.json` 等关键文件采用：

```text
写临时文件 -> flush -> fsync（可用时） -> os.replace()
```

避免进程崩溃后留下半个 JSON 文件。

### 6.3 中断处理

Windows 与 POSIX 语义不同，必须分别实现并分别测试：

- **Windows**：LabBox 与子进程共享同一控制台时，Ctrl+C 由系统同时投递给父进程和子进程，因此 LabBox **不主动向子进程重复发送** `CTRL_C_EVENT`（否则训练脚本可能收到两次中断）。LabBox 捕获 `KeyboardInterrupt` 后只做两件事：停止采样、等待子进程退出并落盘。二次 Ctrl+C 才允许升级为强制终止，并记录 `forced_kill: true`。
- **POSIX**：向子进程组发送 `SIGINT`，`--kill-grace`（默认 10s）后升级为 `SIGTERM`，再超时 `SIGKILL`。
- **退出码保真**：Windows 下被 Ctrl+C 终止的子进程常见退出码是 `0xC000013A`（-1073741510）而不是 130。协议要求同时记录：
  - `exit_code_raw`：操作系统返回的原始值，永不加工；
  - `exit_code`：归一化后的 POSIX 风格值（`0xC000013A -> 130`）；
  - `exit_meaning`：`exited | signaled | killed | unknown`。
- 最终状态记录为 `interrupted`；已完成日志和采样必须落盘，即使训练脚本被强杀。

---

## 7. 本地目录协议

```text
<project>/
├── .labbox/
│   ├── config.toml
│   └── runs/
│       └── 20260922-153012-baseline/
│           ├── run.json
│           ├── environment.json
│           ├── command.txt
│           ├── metrics.json
│           ├── dataset-manifest.json
│           ├── config-values.json
│           ├── explain.json          （仅在使用 LLM 解释后生成）
│           ├── artifacts.json
│           ├── diagnostics.json
│           ├── report.html
│           ├── logs/
│           │   ├── stdout.log
│           │   ├── stdout.plain.log
│           │   ├── stderr.log
│           │   └── events.jsonl
│           ├── samples/
│           │   ├── system.jsonl
│           │   └── system.csv
│           ├── checkpoints/
│           │   └── manifest.json
│           ├── reproduce.ps1
│           └── reproduce.sh
```

### 7.1 Checkpoint 策略

默认只记录：

- 原始路径
- 文件大小
- 修改时间
- hash
- 可选远端地址
- 是否已归档

默认不复制 checkpoint。`--copy-artifacts` 可启用复制，但必须设置大小限制并要求确认。
复制保留 artifact 的项目相对路径（去掉开头的 `checkpoints/`），落在
`<run>/checkpoints/<原相对路径>`：只取 basename 会让 `a/best.pt` 和 `c/best.pt` 互相覆盖，
归档目录里最后只剩一个。复制是分块流式进行的，不会把 checkpoint 整个读进内存。

---

## 8. 数据协议

所有 JSON 必须带 `schema_version`。MVP 使用 `1.0`。

### 8.1 `run.json`

```json
{
  "schema_version": "1.0",
  "run_id": "20260922-153012-baseline",
  "name": "baseline",
  "project": "cifar100-resnet",
  "status": "succeeded",
  "tags": ["baseline"],
  "started_at": "2026-09-22T15:30:12+08:00",
  "ended_at": "2026-09-22T17:13:33+08:00",
  "duration_seconds": 6201.2,
  "exit_code": 0,
  "exit_code_raw": 0,
  "exit_meaning": "exited",
  "interrupted": false,
  "command": {
    "raw": "python train.py --epochs 20",
    "argv": ["python", "train.py", "--epochs", "20"],
    "cwd": "C:/code/project"
  },
  "ended_environment": {
    "captured_at": "2026-09-22T17:13:33+08:00",
    "git_commit": "a1b2c3d",
    "git_dirty": false,
    "gpu_count": 1,
    "duration_seconds": 6201.2
  },
  "links": {
    "environment": "environment.json",
    "metrics": "metrics.json",
    "dataset_manifest": "dataset-manifest.json",
    "artifacts": "artifacts.json",
    "diagnostics": "diagnostics.json",
    "ended_environment": "run.json#ended_environment",
    "report": "report.html"
  }
}
```

`links.ended_environment` 指向同一个 `run.json` 里的 `ended_environment` 字段：该字段必须
真的存在，否则这条 link 只是装饰而不是证据。

### 8.2 `environment.json`

```json
{
  "schema_version": "1.0",
  "captured_at": "2026-09-22T15:30:12+08:00",
  "os": {
    "name": "Windows",
    "version": "11",
    "arch": "AMD64"
  },
  "python": {
    "version": "3.11.9",
    "executable": "C:/Python311/python.exe"
  },
  "frameworks": {
    "torch": "2.6.0",
    "torchvision": "0.21.0"
  },
  "cuda": {
    "runtime": "12.4",
    "driver": "552.22",
    "cudnn": "9.x"
  },
  "git": {
    "commit": "a1b2c3d",
    "branch": "main",
    "dirty": true,
    "diff_hash": "sha256:..."
  },
  "host": {
    "cpu_model": "AMD Ryzen 9",
    "physical_cores": 12,
    "logical_cores": 24,
    "ram_bytes": 68719476736
  },
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 3090",
      "vram_bytes": 25769803776,
      "driver": "552.22"
    }
  ],
  "env": {
    "PYTHONHASHSEED": "0",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
  },
  "package_source": "requirements.txt"
}
```

禁止记录未经脱敏的完整环境变量。变量名包含 `TOKEN`、`KEY`、`SECRET`、`PASSWORD`、`PASS`、`CREDENTIAL` 时默认屏蔽。

### 8.3 `metrics.json`

```json
{
  "schema_version": "1.0",
  "source": "metrics.jsonl",
  "primary_metric": "accuracy",
  "direction": "maximize",
  "final": {
    "accuracy": 0.8421,
    "loss": 0.4312
  },
  "best": {
    "accuracy": {
      "value": 0.851,
      "step": 18,
      "timestamp": "2026-09-22T16:40:10+08:00"
    }
  },
  "series_ref": "metrics/history.jsonl",
  "warnings": []
}
```

指标来源优先级：

1. `--metric-file` 指定的结构化文件
2. `.labbox/config.toml` 中的解析规则
3. 用户注册的正则规则
4. 无指标，不猜测

不要求 MVP 从任意日志文本自动提取指标。

### 8.4 `dataset-manifest.json`

```json
{
  "schema_version": "1.0",
  "datasets": [
    {
      "id": "cifar100-train",
      "source": "data/cifar100/train",
      "kind": "filesystem",
      "strategy": "fast_manifest",
      "digest": "sha256:91ab...",
      "file_count": 50000,
      "total_bytes": 123456789,
      "generated_at": "2026-09-22T15:29:50+08:00",
      "exclusions": ["*.tmp", "__pycache__/**"],
      "warnings": []
    }
  ]
}
```

支持的策略：

- `fast_manifest`：路径 + 大小 + 内容采样，默认
- `full_content`：完整内容 hash，用户显式开启
- `external_version`：Hugging Face revision、DVC hash、对象存储 version ID 等
- `declared`：用户只声明版本，LabBox 不做验证

**digest 计算定义（必须严格按此实现）**：

```text
entries   = 目录内所有非排除文件，按 POSIX 相对路径字典序排序
per_file  = f"{rel_path}|{size_bytes}|{head64_sha256}|{tail64_sha256}"
digest    = sha256("\n".join(per_file))
```

- **修改时间不进 digest**。mtime 会在拷贝、解压、重新 checkout 后全量变化，把它纳入 digest 会让"数据版本变了"这条最关键的诊断长期误报，从而摧毁产品核心卖点。
- mtime 作为独立 metadata 记录，出现"路径与大小相同但 mtime 明显变新"时单独产生 `dataset_files_touched` 诊断，措辞为"文件可能被就地修改过"，而不是"数据版本变化"。
- `head64/tail64` 为文件首尾各 64 字节的 sha256；文件小于 128 字节时只算一次完整 hash。
- 不能把 `fast_manifest` 描述为内容不可变证明；`zero_byte_files`、`sampled_only` 必须在 `warnings` 中体现。

### 8.5 `artifacts.json`

```json
{
  "schema_version": "1.0",
  "artifacts": [
    {
      "kind": "checkpoint",
      "path": "checkpoints/best.pt",
      "size_bytes": 987654321,
      "sha256": "abc123...",
      "archived": false,
      "storage_uri": null
    }
  ],
  "errors": []
}
```

### 8.6 `diagnostics.json`

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-09-22T17:13:34+08:00",
  "reproducibility": {
    "band": "medium",
    "score": 72,
    "checklist": [
      {"id": "code_locatable", "state": "partial", "reason": "git.dirty=true"},
      {"id": "randomness_controlled", "state": "missing", "reason": "未记录 seed"}
    ]
  },
  "diagnostics": [
    {
      "id": "D-001",
      "severity": "warning",
      "code": "git_dirty",
      "message": "训练开始时 Git 工作区存在未提交修改。",
      "evidence": ["git.dirty=true", "git.diff_hash=sha256:..."],
      "suggestion": "提交或导出补丁后再运行。"
    }
  ]
}
```

`reproducibility_score` 只是风险提示，不是数学意义上的复现概率。

---

## 9. 采集器设计

### 9.1 ProcessRunner

职责：

- 使用 `subprocess.Popen` 启动子进程
- 不经过 shell，避免命令拼接和注入
- 同步转发 stdout/stderr
- 保留原始退出码
- 支持中断转发
- 不因日志落盘失败而中断训练

Windows 注意点：

- 需要同时处理管道和终端输出
- 避免死锁，stdout/stderr 由独立 reader 处理
- 正确传递 Unicode 和 ANSI 输出

**Windows 编码与命令行约束（实测要求）**：

- 子进程一律使用 `argv` list + `shell=False` 启动，参数中含引号、`&&`、`|`、空格、中文时不得被解释为 shell 语法。
- 控制台默认代码页可能是 GBK/cp936 而非 UTF-8。日志落盘以**原始字节**写入（`O_BINARY`），解码只在展示层做，使用 `errors="replace"`，禁止因为解码异常丢行。
- ANSI 转义序列原样保留在 `logs/stdout.log`，另存一份剥离 ANSI 的 `logs/stdout.plain.log` 供报告使用。
- 子进程环境显式设置 `PYTHONUNBUFFERED=1`，并在文档中说明：不加此行时行缓冲会破坏"输出与裸命令一致"。

### 9.2 EnvironmentCollector

**采集器运行在 LabBox 自己的解释器里，与被训练脚本使用的解释器不是同一个（见 14.4）。**

数据来源：

- Python：`platform`、`sys.version`（LabBox 自身）+ **对训练解释器起只读子进程**获取版本信息
- torch/CUDA：通过训练解释器的只读子进程查询，见下方禁止清单
- GPU：`nvidia-ml-py`，失败时回退 `nvidia-smi --query-gpu`
- CPU/RAM：`psutil`
- pip：子进程内 `importlib.metadata`，按显式或主要依赖列表收集
- Git：`git rev-parse`、`git status --porcelain`、`git diff` hash

**禁止清单（会导致训练被干扰或结果不真实）**：

- `torch.cuda.is_available()`、`torch.cuda.device_count()`、`torch.cuda.get_device_name()`：这些调用会**初始化 CUDA context**，可能占用数百 MB 显存并影响随后训练的显存上限与吞吐。GPU 信息一律走 NVML。
- 允许：`torch.__version__`、`torch.version.cuda`、`torch.version.git_version`（纯字符串，不建立 context）。
- 禁止导入训练项目内的任何模块；禁止 `pip install`、`pip freeze` 之外的包管理命令。
- 探测子进程必须带超时（默认 10s）和输出大小上限；超时记 `status=timeout`。

约束：

- 不执行训练项目代码
- 不自动安装依赖
- 不运行 `pip freeze` 之外可能修改环境的命令
- 超时必须可配置，单次采集默认不超过 10 秒

### 9.3 HardwareSampler

采样字段：

- 时间戳
- GPU utilization
- GPU memory used/total
- GPU temperature
- 可选功耗
- CPU utilization
- process/host RAM used

采样失败必须记录错误但继续运行。默认每 2 秒采样一次，长训练可自动降低为 5 秒，避免大量小文件。

### 9.4 MetricsIngestor

MVP 支持：

- `.json`
- `.jsonl`
- 用户配置的 regex
- 内置解析器两种：通用 `key: value` 行式日志、HuggingFace Trainer 的 `metrics.json`
- 显式调用 `labbox.log_metric()`（后期）

MVP 不做：

- 自动扫描所有日志并猜测指标
- 将打印文本中的最后一个浮点数当作准确率
- 在指标缺失时推测数值

指标完全缺失时（零侵入场景的常见结果）必须产生 `METRICS_MISSING` 诊断，`metrics.json` 写 `{"final": null, "warnings": ["metrics_missing"]}`，而不是省略文件或写空对象。

### 9.6 ConfigValueExtractor

`compare` 需要输出 `train.batch_size 64 -> 128` 这类事实，因此配置取值提取是 MVP 功能，而不是可选项。

MVP 支持范围（明确边界）：

- 格式：YAML、JSON、TOML
- 解析后拍平为点路径：`{"train": {"batch_size": 64}}` -> `train.batch_size = 64`
- 来源优先级：命令行 `--key.path` 覆盖 > 配置文件 > 代码默认值（未知）
- 配置文件路径来自 `--config`、`labbox run --config-file` 或 `.labbox/config.toml` 的 `configs.patterns`

明确不做：

- INI、HOCON、protobuf text、Hydra 组合求值、OmegaConf 插值解析
- 尝试推断代码里的 argparse 默认值

冲突与不确定性处理：同一 key 同时出现在 argv 和配置文件时，两个来源都记录在 `sources` 里，生效值取 argv；无法确定生效值时 `confidence = "unknown"` 并生成 `hyperparam_source_ambiguous` 诊断。禁止静默选择一侧。

### 9.5 ArtifactCollector

发现来源：

- 配置中的 glob
- 命令行 `--artifact`
- 常见目录中的 checkpoint 后缀
- metrics 文件

大文件默认只做可选 hash。超过阈值时先记录 metadata，hash 由用户显式确认或后台执行。

---

## 10. 对比引擎

### 10.1 输入

两个规范化 `RunRecord`，不直接比较原始文件。

### 10.2 差异类型

| 类型 | 示例 |
|---|---|
| `changed` | Git commit 不同 |
| `added` | 新依赖、新 artifact |
| `removed` | 缺少原数据集声明 |
| `numeric_delta` | batch size 64 -> 128 |
| `metric_delta` | accuracy -3.78 pp |
| `unknown` | 一侧未采集到值 |
| `incomparable` | 数据 digest 不同，直接比较指标没有意义 |

### 10.3 Fact 结构

```json
{
  "id": "F-003",
  "path": "train.batch_size",
  "kind": "numeric_delta",
  "left": 64,
  "right": 128,
  "delta": 64,
  "sources": ["config.yaml:12", "command.argv"],
  "confidence": "high"
}
```

### 10.4 规则引擎

规则必须可测试、可解释。首批规则：

- Git commit 变化
- Git dirty 状态变化
- 数据集 digest/版本变化
- Python、PyTorch、CUDA 变化
- GPU 型号变化
- batch size、学习率、AMP、精度变化
- seed 缺失或变化
- CUDA deterministic 设置缺失或变化
- 同一比较中多个关键变量同时变化
- 指标方向不一致
- 一侧指标缺失
- 环境采集失败
- checkpoint 缺失或 hash 不匹配

### 10.5 因果表达约束

允许：

> 数据版本和 batch size 同时变化，因此当前证据无法单独归因。

不允许：

> 准确率下降 3.8% 是数据集版本变化造成的。

除非存在受控实验或更多证据，否则只能给出候选解释和验证建议。

---

## 11. AI 解释层

### 11.1 Provider 抽象

```python
class LLMProvider(Protocol):
    async def explain(self, context: ExplainContext) -> ExplainResult:
        ...
```

首批实现：

- OpenAI-compatible endpoint
- 本地推理服务 endpoint
- 禁用模式的 `NullProvider`

不要将 `Qwen3.8` 或其他具体模型名写死在核心代码中。模型名称只属于用户配置。

### 11.2 输入协议

`ExplainContext` 包含：

- 任务类型：单次诊断或双次对比
- 结构化 facts
- 结构化 diagnostics
- 精选 metrics
- 缺失字段 list
- 数据来源引用
- 隐私级别

不包含：

- 完整环境变量
- 未脱敏日志
- 任意本地文件内容
- Git diff 全文（除非用户显式允许）
- API key 和认证头

### 11.3 输出协议

```json
{
  "summary": "string",
  "facts": [
    {
      "fact_id": "F-003",
      "interpretation": "string"
    }
  ],
  "hypotheses": [
    {
      "text": "string",
      "confidence": "low|medium|high",
      "evidence_fact_ids": ["F-003", "F-004"],
      "validation": "string"
    }
  ],
  "unknowns": ["string"],
  "next_steps": ["string"]
}
```

响应必须通过 JSON Schema 校验。校验失败时原样展示确定性分析，不展示半成品模型输出。

### 11.4 Prompt Injection 防护

日志、Git commit message、文件路径和配置内容都可能包含恶意指令。AI 层必须：

- 将输入标记为不可信数据
- 不执行输入中的指令
- 不根据输入请求读取额外文件
- 不允许输出工具调用
- 只允许引用已知 fact ID

---

## 12. 复现性设计

### 12.1 复现性评分维度

单一数字（如 `72`）没有可解释性——权重是人为设定的，72 与 68 无法区分，容易被误读为数学意义上的复现概率。因此主输出是 **checklist**，数字仅作为次要摘要字段保留。

| 维度 | id | 判定 |
|---|---|---|
| 代码可定位 | `code_locatable` | commit 存在且 `dirty=false` 为 `ok`；dirty 但有 diff_hash 为 `partial`；无 commit 为 `missing` |
| 依赖可重建 | `deps_reproducible` | 有 lock 文件或完整版本清单为 `ok`，仅部分为 `partial` |
| 数据可验证 | `data_verifiable` | `full_content`/`external_version` 为 `ok`，`fast_manifest` 为 `partial`，`declared`/无为 `missing` |
| 随机性可控制 | `randomness_controlled` | seed 已设 + deterministic 已设为 `ok`，只设 seed 为 `partial` |
| 硬件可识别 | `hardware_identified` | GPU 型号 + CUDA + 驱动齐全为 `ok` |
| 命令可重放 | `command_replayable` | argv 与 cwd 完整为 `ok` |
| artifact 可定位 | `artifacts_locatable` | 有路径 + 大小为 `partial`，有 hash 为 `ok` |

输出：

```json
{
  "reproducibility": {
    "band": "medium",
    "checklist": [
      {"id": "data_verifiable", "state": "partial", "reason": "strategy=fast_manifest, 未做完整内容 hash"}
    ],
    "score": 72
  }
}
```

`band` 由 `ok/partial/missing` 计数决定（任一维度 `missing` 且属于前四行重要维度即为 `low`）。`score` 由同一份集中定义、版本化的规则表算出，**报告中的主展示是 checklist 与 band，`score` 以小字号附注呈现**。

### 12.2 复现脚本边界

`reproduce.ps1` 和 `reproduce.sh` 只负责：

- 校验当前目录和工具
- 提示创建虚拟环境
- 提示安装指定依赖
- 校验 Git commit
- 校验数据集 digest
- 输出原始训练命令
- 可选自动执行

脚本不得：

- 自动覆盖用户环境
- 自动安装 GPU 驱动
- 声称能够恢复完全相同结果
- 将缺失信息静默忽略

---

## 13. HTML 报告

报告是只读快照，不依赖后端服务。

MVP 包含：

- Run summary
- Environment
- Command
- Metrics
- Hardware timeline
- Dataset manifest status
- Artifacts
- Reproducibility diagnostics
- Changed facts（对比模式）
- AI explanation（若启用且用户请求）

技术要求：

- 单文件 HTML，CSS/JS 内联优先
- 不加载远程字体、脚本和跟踪像素
- 用户内容必须 HTML escape
- 无 LLM 也可完整阅读
- 页面生成失败不得影响 run 数据

---

## 14. 技术栈

### 14.1 语言和运行时

- Python `>=3.11`
- 类型检查：`mypy` 或 `pyright`
- 代码质量：`ruff`
- 测试：`pytest`
- 构建：`hatchling`
- 开发环境管理：`uv`

### 14.2 主要库

- `typer`：CLI
- `pydantic`：schema 和校验
- `rich`：终端展示
- `psutil`：CPU/RAM
- `nvidia-ml-py`：GPU 采集
- `jinja2`：HTML 报告
- `httpx`：LLM API
- `tomli-w`：仅写 TOML 时需要；读取使用 Python 3.11 `tomllib`

核心记录路径的第三方依赖应尽量少，避免通过 LabBox 改变训练环境。

### 14.3 安装与解释器模型（已定稿）

LabBox **不安装进用户的训练环境**。理由：LabBox 需要 pydantic/typer/rich/httpx 等依赖，若装进训练 venv，则"不改变训练环境"这条原则自动失效，而且依赖冲突会让用户第一次试用就炸掉自己的实验环境。

定稿模型：

```text
用户训练 venv (torch 2.6, ...)   ←── 只被读取，永不被写入
        ▲
        │ 只读子进程探测（argv 指向的解释器，超时 10s）
        │
LabBox 独立环境（uv tool install / pipx / 独立 venv）
        │
        └── subprocess.Popen 启动训练命令（不经过 shell）
```

由此产生的三条实现要求：

1. **训练解释器推断**：命令 argv[0] 若解析为 `python`/`python3`/`py`/`python.exe`，用 `shutil.which` 得到实际可执行文件（`py -3.11` 形式需二次解析）；否则记 `interpreter.kind = "external"`，跳过包版本采集，只采 OS/CPU/RAM/Git。`labbox run --python <path>` 可显式覆盖。
2. **跨解释器采集只能走子进程**：`subprocess.run([interp, "-c", PROBE_SRC])`，`PROBE_SRC` 是核心代码里的一段常量字符串，输出 JSON。这意味着采集必然带一次进程启动开销（Windows 冷启动约 100–300ms），因此环境采集预算放宽到 `< 3 s`。
3. **禁止 `import` 训练环境包**：LabBox 进程内不得 `import torch`，即使可用。NVML 与 `nvidia-smi` 是驱动级信息，与解释器无关，在 LabBox 自身环境查询。

开发期安装方式：`uv tool install --editable .`（或 `uv run labbox`），README 中同时给出 pipx 路径。

### 14.4 推荐仓库结构

```text
labbox/
├── pyproject.toml
├── README.md
├── DEVELOPMENT.md
├── src/
│   └── labbox/
│       ├── __init__.py
│       ├── cli/
│       ├── config/
│       ├── domain/
│       ├── recorder/
│       ├── analysis/
│       ├── report/
│       ├── llm/
│       └── infrastructure/
├── templates/
│   └── report/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── schemas/
    └── adr/
```

---

## 15. 配置设计

`.labbox/config.toml` 示例：

```toml
schema_version = "1.0"

[run]
sample_interval_seconds = 2.0
store_logs = true

[git]
enabled = true
include_diff_hash = true

[gpu]
enabled = true
metrics = ["utilization", "memory", "temperature", "power"]

[datasets]
strategy = "fast_manifest"
paths = ["data"]
exclude = ["**/__pycache__/**", "**/*.tmp"]

[artifacts]
patterns = ["checkpoints/*.pt", "outputs/**/best.*"]
hash_max_bytes = 2147483648
copy_by_default = false

[metrics]
files = ["metrics.jsonl"]
primary = "accuracy"
direction = "maximize"

[privacy]
redact_env_patterns = ["(?i).*token.*", "(?i).*key.*", "(?i).*secret.*"]
store_raw_logs = true

[llm]
enabled = false
provider = "openai_compatible"
base_url = "http://127.0.0.1:8000/v1"
model = "local-model"
send_logs = false
```

配置优先级：

```text
CLI 参数 > 项目配置 > 用户配置 > 默认值
```

所有路径在展示和保存时需明确是绝对路径还是项目相对路径。

---

## 16. 安全与隐私

### 16.1 威胁模型

需要防范：

- API key、token 和密码写入环境快照
- 日志携带敏感数据
- Git remote URL 包含凭据
- LLM endpoint 收到未授权数据
- HTML 报告中的 XSS
- 配置路径穿越
- artifact glob 意外读取项目外大目录
- Prompt injection
- 非 ASCII（中文）与含空格路径在 shell 脚本、HTML、日志中的转义错误
- run 目录残留敏感数据且无法清理

### 16.3 实测到的 Windows 平台限制

以下行为来自 `scripts/spike_process.py` 在 Windows 11 + cp936 控制台上的实测，不是推测：

1. **子进程写入管道时使用本地编码（cp936/GBK），不是 UTF-8。** 因此日志必须按原始字节落盘、
   展示层再用 `errors="replace"` 解码；任何"按 UTF-8 解码失败就报错"的实现都会炸掉真实训练。
2. **`taskkill /F` 终止的子进程退出码是 `1`**，与 `exit(1)` 完全无法区分。因此
   "谁杀的进程"这一信息只能由 LabBox 自己记录（`killed_by_labbox`），不能从退出码推断。
3. **无法用 `os._exit(0xC000013A)` 伪造 Ctrl+C 退出码**（实测：`Popen` 报回的是 `1`），
   该值只能由真实控制台事件产生。**已在真实终端实测通过**：LabBox 作为父进程收到
   `KeyboardInterrupt`，未向子进程二次发信号，子进程状态为
   `raw=3221225786`（即 `0xC000013A` 的无符号形式）→ 归一化 `exit_code=130`、
   `interrupted=true`，与 6.3 的设计一致。复现：在 cmd/PowerShell 窗口里执行
   `python scripts/spike_process.py`，其中两条 console-gated 用例为
   ①ProcessRunner 层的 Ctrl+C 捕获（**已实测通过**），②完整 `labbox run` 被中断后仍然完成
   finalize：`status=interrupted` 且 metrics/artifacts/diagnostics/config-values/report/
   reproduce 全部落盘（**本次新增，尚待一次真实终端验证**）。
   无控制台的环境（CI 容器）会把这两条报为 `SKIP`。
4. **editable 安装的 `.pth` 文件用本地编码解码**：当安装路径含非 ASCII 字符
   （例如 `C:\code\github小项目`）时，`site.py` 用 cp936 读取 UTF-8 写入的路径导致
   `import labbox` 失败、`labbox.exe` 无法启动。约束：**LabBox 本体必须安装在纯 ASCII
   路径下**（pipx / `uv tool install`），而被记录的用户项目可以位于任意路径。
   开发环境用 `scripts/labbox_dev.py` 绕过。
5. `.gitignore` 必须包含 `.labbox/runs/`：否则 `git status` 会把每个 run 都算成 untracked，
6. **录制器中途死亡会留下"无法解释的残缺目录"**（真实终端实测发现）：强杀 LabBox 后
   run 目录只剩 6 个文件，`run.json` 停在 `status=running`，而 metrics / artifacts /
   report / reproduce 全部不存在，读者无法判断这是"被中断"还是"没实现"。
   修复方式：`_prepare_files` 在写 `run.json` 之前先写一版
   `diagnostics.json`（`code=run_in_progress`），并把该文件登记进 `run.json#links`；
   `labbox doctor` 额外报告 `status` 仍为 running 的 run。
   因此"初版诊断文件"是崩溃安全设计的一部分，不是冗余写入。
7. **Windows 中断事件的两个实测陷阱**（都踩过，都已在代码里处理）：
   - `CTRL_BREAK_EVENT` 不会被 CPython 转成 `KeyboardInterrupt`，默认行为是让进程以
     `0xC000013A` 直接终止 —— 实测中 LabBox 因此死在训练中途、finalize 完全没跑。
     对策：`labbox.cli.main` 在 Windows 上把 `SIGBREAK` 绑到 `default_int_handler`，
     使 Ctrl+Break 与 Ctrl+C 等价（`tests/unit/test_cli_console_handlers.py`）。
   - `CREATE_NEW_PROCESS_GROUP` 会**禁用该进程组的 Ctrl+C**（它只保证 Ctrl+Break 可投递），
     所以用它 + `CTRL_C_EVENT` 模拟按键时"没人收到中断"，run 会正常跑完并记为 succeeded。
     对策：验证脚本改为给整个控制台发 Ctrl+C（`os.kill(0, CTRL_C_EVENT)`），并让 spike
     自身忽略这一次中断 —— 这是与真实按键最接近的等价物。

### 16.2 强制要求

- 默认只写本地
- 所有远端请求必须显式配置
- 数据发送前展示数据类别，并支持 `--dry-run`
- 内置敏感变量脱敏
- HTML escape 所有用户数据
- 允许用户删除单个 run
- config 不允许默认跟随符号链接到项目外
- 测试必须包含 secret redaction 和 HTML escaping

---

## 17. 性能要求

- `labbox` 自身启动开销：冷启动（含解释器与依赖导入）`< 1.5 s`，热启动 `< 500 ms`；用 `labbox --version` 实测
- 环境采集：常规环境 `< 3 s`（含一次跨解释器子进程探测的固定开销）
- 系统采样：默认间隔不高于训练 CPU 占用的 `1%`
- 日志转发：不得明显降低训练吞吐
- JSON 写入：不能在训练循环期间频繁 rewrite 大文件
- 采样数据：JSONL 追加写；关键 summary 在结束时写
- 数据集 `fast_manifest`：100k 小文件应做到可取消、可显示进度
- 大 artifact hash：默认熔断并延迟处理

性能测试需要对比裸命令与 `labbox run` 的墙钟时间和平均 CPU 占用。

### 17.1 实测基线（Windows 11 / Python 3.11.9 / 无 GPU / `scripts/bench_overhead.py`）

| 指标 | 实测 | 预算 |
|---|---:|---:|
| LabBox 启动（解释器 + 依赖导入） | 0.48 s | < 1.5 s |
| 单次运行的固定记录成本（探测 + git + 报告） | 1.14 s | < 3 s |
| 其中 `--dataset` 数据 hash（50 文件 / 45 KB） | +0.06 s | 计入固定成本 |
| 采样开销（5.15 s 训练，2 s 间隔） | 0.010 s ≈ **0.2 %** | ≤ 5 % 墙钟 |
| 每个 run 落盘大小 | ≈ 99 KiB | — |

结论：记录成本是**每次运行的固定开销**，不是随 step 累积的税；采样循环本身对墙钟的影响在
千分位量级。因此对真实（分钟级）训练，相对开销会趋近于 0。上表由
`python scripts/bench_overhead.py` 可复现，超出预算时该脚本以非 0 退出。

---

## 18. 测试策略

### 18.1 单元测试

- schema 序列化与向后兼容
- 环境变量脱敏
- 命令 argv 构造
- Git 状态解析
- metrics JSON/JSONL 解析
- dataset manifest 稳定排序
- diff fact 生成
- 规则触发和 severity
- reproducibility score
- HTML escaping
- prompt context sanitization

### 18.2 集成测试

使用假的训练脚本模拟：

- 成功训练
- 非零退出码
- 长时间输出
- stdout/stderr 并发输出
- Unicode 和 ANSI
- 被 Ctrl+C 中断
- 缺失 GPU
- 缺失 Git
- 损坏 metrics
- 超大 artifact 策略

### 18.3 端到端验收

```bash
labbox init
labbox run --name run-a -- python examples/train_fake.py --lr 0.1
labbox run --name run-b -- python examples/train_fake.py --lr 0.01
labbox compare run-a run-b
labbox report run-b
```

验收要求：

- 两次 run 均完整落盘
- 退出码与原命令一致
- 对比结果准确引用 fact ID
- 报告可离线打开
- 无 GPU 环境仍能运行
- 输出中不存在测试 secret

---

## 19. 可观测性与日志

LabBox 自身日志与训练日志必须分开。

- 用户 stdout/stderr：`logs/stdout.log`、`logs/stderr.log`
- LabBox 事件：`logs/events.jsonl`
- 终端错误：使用结构化错误码
- debug 模式通过 `LABBOX_DEBUG=1` 或 `--verbose`

统一错误结构：

```json
{
  "code": "GPU_COLLECTOR_UNAVAILABLE",
  "message": "NVIDIA driver was not detected.",
  "component": "hardware_sampler",
  "recoverable": true,
  "details": {}
}
```

禁止在异常信息中泄露 API key。

---

## 20. 开发里程碑

### M0：协议和骨架

产出：

- 仓库结构
- `pyproject.toml`
- domain schema
- `labbox init`
- 测试基础设施

验收：

- schema round-trip 测试通过
- 无 GPU、无 Git 环境可启动

### M1：可靠运行记录

产出：

- `labbox run`
- 进程转发
- 环境采集
- GPU/CPU/RAM 采样
- 日志和基础 run manifest
- 原子写入与中断处理

验收：

- 成功、失败和中断场景均生成记录
- 返回原始退出码

### M2：数据、指标和 artifact

产出：

- dataset manifest
- metrics JSON/JSONL
- artifact manifest
- `reproduce.ps1` / `reproduce.sh`

验收：

- 大文件策略有测试
- 指标缺失不被伪造
- 路径和 shell 参数正确转义

### M3：对比和诊断

产出：

- `labbox list/show`
- `labbox compare`
- DiffEngine
- RuleEngine
- reproducibility score
- `diagnostics.json`

验收：

- 首批规则覆盖完整
- 同时变化变量会被标记为不可单独归因
- 缺失数据与真实变化被区分

### M4：报告和 AI 解释

产出：

- `labbox report`
- 离线 HTML
- LLM provider 抽象
- `labbox explain`
- JSON Schema 输出校验和 prompt injection 防护

验收：

- 无 LLM 时报告完整
- AI 输出引用 fact ID
- 隐私测试通过

### M5：可用性打磨

产出：

- 安装包和发布流程
- README、示例和录屏
- Windows CI
- 性能基准
- 示例实验集

验收：

- 新用户在 5 分钟内跑通第一个实验
- 3090 上完成 CNN、YOLO 和小型 Transformer 的实战验证

---

## 21. MVP 完成定义

满足以下条件才视为 MVP 完成：

1. 用户不修改训练脚本也能完成基础记录。
2. `labbox run` 不改变训练结果和退出码。
3. 成功、失败、中断三类运行均完整保留。
4. 环境变量默认脱敏。
5. 能准确对比两次运行的结构化差异。
6. 明确区分事实、推断和证据缺失。
7. GPU 不存在时仍可完整使用核心功能。
8. HTML 报告可离线打开。
9. LLM 关闭时所有核心功能仍然可用。
10. Windows 集成测试通过。

---

## 22. 已定稿决策与剩余待确认问题

### 22.1 已定稿（原 22 节中阻塞 M0 的项）

| 原问题 | 决策 |
|---|---|
| 项目正式名称与包名 | `LabBox`，Python 包名/import 名 `labbox`，命令 `labbox` |
| `.labbox/runs` 是否进 `.gitignore` | 默认写入 `.gitignore`，`labbox init --no-gitignore` 可关闭，改为提示 |
| 安装模型 | LabBox 独立环境 + 跨解释器只读子进程探测（14.3） |
| 数据集默认发现策略 | 不猜路径。默认只处理 `[datasets] paths` 显式声明与 `--dataset`；两者皆无时记 `dataset: not_declared` 并给出诊断，避免误报 |
| checkpoint hash 默认阈值 | `hash_max_bytes = 2 GiB`，超过则只记 metadata + `artifact_hash_deferred`，由 `--hash-artifacts` 显式放开 |
| LLM 协议 | 只实现 OpenAI-compatible `chat/completions` 一个协议，本地与远端都走它 |
| Python 版本 | `>=3.11`（用 `tomllib`），不支持 3.10 |
| 许可证 | MIT |

### 22.2 剩余待确认（不阻塞里程碑）

1. 报告是否嵌入硬件时序图，或只生成独立 CSV 供外部工具绘制。默认方案：生成 `samples/system.csv` + 报告内 SVG 折线（无 JS）。
2. 首个真实验收项目（CIFAR 小 CNN / YOLO / 小型 LoRA）。M0–M2 全部用 `examples/` 假训练脚本，M3 前再定。
3. metrics 是否提供 Python SDK。默认坚持零侵入 + 内置解析器，SDK 推到 MVP 之后。

---

## 23. 第一版实现顺序

严格按以下顺序开发，避免过早进入 UI：

```text
1.  Pydantic domain schema
2.  FilesystemStore + atomic write
2b. ProcessRunner Windows 预研（spike）：shell 注入、管道死锁、退出码、编码四项先验证再写实现
3.  ProcessRunner + fake training scripts
4.  EnvironmentCollector（跨解释器探测 + 脱敏）
5.  HardwareSampler（FakeBackend 优先，真卡测试打 gpu 标记）
6.  run.json / logs / samples 生命周期组装  ← M1 终点
7.  ConfigValueExtractor（YAML/JSON/TOML 拍平）
8.  MetricsIngestor
9.  ArtifactCollector
10. DatasetManifestBuilder
11. reproduce 脚本生成器                    ← M2 终点
12. DiffEngine
13. RuleEngine
14. diagnostics.json + reproducibility checklist
15. compare / list / show CLI               ← M3 终点
16. HTML report
17. LLM explain                             ← M4 终点
```

第 2b 步是本顺序中唯一插进去的预研项，原因：ProcessRunner 同时受”Never break training”和”退出码保真”两条硬约束，且 Windows 语义只能实测，不能靠推理写完。

第一步的完成标准不是”能记录很多东西”，而是：

> 用一个假训练脚本跑出成功、失败、中断三种可靠记录，并且行为与裸命令一致。

### 23.1 里程碑依赖关系

- M1 不依赖 GPU、不依赖 Git、不依赖任何第三方训练框架。
- M3 的 DiffEngine 只消费 M1/M2 产出的规范化对象，不读原始文件。
- M4 的 LLM 层是唯一可完全删除的部分；删除后 M1–M3 行为不变。

---

## 24. 开发约束

- 不允许为了记录一个可选字段而破坏训练命令。
- 所有核心 schema 变化必须升版本并提供迁移策略。
- 所有外部命令必须设置超时。
- 所有子进程参数使用 argv list，禁止 shell 字符串拼接。
- 所有新增诊断规则必须有单元测试和真实反例。
- 所有模型输出必须经过 schema 校验。
- 所有远端能力必须可关闭。
- 任何”复现成功”结论都必须能追溯到具体证据。
- 采集器代码中禁止出现 `torch.cuda.is_available()` / `device_count()` / `import torch`（LabBox 自身环境内），Code Review 时按硬约束处理。
- 数据集 digest 的输入字段只能是 `相对路径 + 大小 + 内容采样`，任何把 mtime 纳入 digest 的改动视为 bug。
- 指标缺失只能记录为缺失；任何”从日志里找最后一个浮点数”式的推断不得进入核心路径。
- 涉及路径、命令、HTML 的输出必须经过转义或 escape，且有对应测试。

---

## 25. 实现状态（MVP 完成定义逐条核验）

核验方式必须是可执行的，不接受口头声明。`python scripts/verify_mvp.py` 会在临时项目里真跑
CLI 并逐条判定下列 10 条，任何一条不成立就返回非 0。

| # | 完成标准 | 状态 | 证据 |
|---|---|---|---|
| 1 | 不修改训练脚本也能完成基础记录 | 已验证 | `verify_mvp.py` 第 1 项；`tests/integration/test_m1_acceptance.py` |
| 2 | `labbox run` 不改变训练结果与退出码 | 已验证 | `test_stdout_and_stderr_are_byte_identical_to_the_bare_command`、`test_exit_codes_are_preserved_exactly` |
| 3 | 成功 / 失败 / 中断三类运行均完整保留 | 已验证 | 成功/失败见 M1 用例；录制器被强杀时留下 `status=running` + 初版 `diagnostics.json`（`code=run_in_progress`），并在 `labbox doctor` 中列为 stale，绝不伪成功；真实 Ctrl+C 在终端实测：finalize 全量落盘、`status=interrupted`、退出码 130（§25.2.1） |
| 4 | 环境变量默认脱敏 | 已验证 | `tests/unit/test_redact.py`、`test_environment_secrets_are_never_written` |
| 5 | 能准确对比结构化差异 | 已验证 | `tests/unit/test_diff_rules.py`、`tests/integration/test_m3_acceptance.py` |
| 6 | 区分事实 / 推断 / 证据缺失 | 已验证 | `unknown` 与 `incomparable` 是独立 diff 类型；`evidence_gaps` 诊断；测试断言输出中不出现因果表述 |
| 7 | 无 GPU 时核心功能可用 | 已验证 | 本轮全部测试都在无 NVIDIA 工具的机器上通过；`tests/integration/test_sampler.py` 用 `FakeBackend` 覆盖 |
| 8 | HTML 报告可离线打开 | 已验证 | `tests/unit/test_report_html.py` 断言不出现 `http(s)://`、`<script>`、`@import`、`url(` |
| 9 | LLM 关闭时核心功能可用 | 已验证 | `tests/unit/test_llm_guards.py` + `test_explain_refuses_without_a_configured_model` |
| 10 | Windows 集成测试通过 | 已验证 | 本轮 420 个测试在 Windows 11 上通过；`.github/workflows/ci.yml` 只跑 `windows-latest` 矩阵 |

### 25.1 里程碑

M0 协议骨架、M1 可靠运行记录、M2 数据/指标/artifact/reproduce、M3 对比与诊断、M4 报告与解释层、
M5 收尾（README、schema 导出、性能基线、验收脚本、CI）均已落地。

### 25.2 明确未完成 / 需要人参与的事项

1. **3090 上的实战验证（§20 M5）。** 需要一台有 NVIDIA 驱动的机器：
   `pytest -m gpu` 在无 GPU 机器上被跳过，`NvmlBackend` 的真卡路径尚未执行过。
   建议顺手用它记录一次真实的 CIFAR / YOLO 训练，一次动作同时验证真卡采集与实战可用性。
2. **Linux 不在发布范围（已定稿）。** POSIX 分支保留在代码中，但 CI 只跑 Windows，
   也不承诺 Linux/macOS 可用性；不再作为发布待办。
3. **仓库已初始化（2026-10-20）。** `main` 分支首次提交已完成；远端仓库、分支保护与发布流程由所有者配置。

### 25.2.1 已关闭的中断验证（2026-09-22）

真实终端实测 `scripts/spike_process.py` **16/16 PASS**，其中新增的两条：

```text
PASS  interrupted CLI run finalizes            status=interrupted exit=130 raw=3221225786 missing=[]
PASS  interrupted CLI run returns the Ctrl+C code  labbox exit=130 recorded=130 (raw 3221225786)
```

含义：Ctrl+C 中断训练后，LabBox 仍完成 finalize（metrics / artifacts / dataset-manifest /
config-values / diagnostics / report.html / reproduce.ps1 / reproduce.sh / samples 全部落盘），
run 记为 `interrupted`，并以子进程的归一化退出码 130 退出。过程中发现并修掉的两个真实缺陷见 §16.3-6、§16.3-7。

### 25.3 质量闸门

```text
ruff check src tests scripts examples     # 全部通过
ruff format --check src                   # 全部通过
pyright（src + examples）                  # 0 errors；tests 见 pyproject 中的说明
pytest -q                                 # 全部通过
python scripts/verify_mvp.py              # 10 PASS / 0 SKIP
python scripts/spike_process.py            # 需真实终端：16/16 PASS(无控制台时 2 条 SKIP)
python scripts/bench_overhead.py          # within budget(见 17.1)
powershell -File scripts/check.ps1        # 以上全部串起来
```
