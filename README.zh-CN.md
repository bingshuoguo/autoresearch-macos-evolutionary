# autoresearch-macos-genetic

[English README](README.md)

![teaser](progress.png)

这是一个把 `autoresearch` 的核心想法改造成 macOS 版本遗传搜索控制器的分支。

它的目标不是让 agent 每次只改一次 `train.py`、跑一次、再决定 keep/reset，而是让 agent 在一个小型但真实的 LLM 训练环境里运行显式的 genetic loop：

- 渲染一批 candidate experiments
- 每个 candidate 固定训练 5 分钟
- 按 `val_bpb`、有效性、资源占用、复杂度排序
- 保存 artifacts 和 lineage
- 从当前 generation breeding 下一代

你醒来之后，看到的不再是一串离散的单次试验，而是一条可恢复、可回放、可比较的 population history。

## 核心组成

这个仓库现在主要由 5 个部分组成：

- **`prepare.py`**
  固定数据准备、tokenizer、dataloader 和评估逻辑。搜索过程中不修改。
- **`train.py`**
  单个 candidate 的执行 skeleton。支持传统的 `uv run train.py`，也支持 `--experiment <path>`。
- **`evolve.py`**
  主控制器。负责 seed/resume population、执行 generation、排序、写入状态和 artifacts。
- **`evolution/`**
  放 schema、registry、ranking、artifacts、runner 等遗传搜索基础设施。
- **`program.md`**
  给 agent 的工作契约。当前已经切换到 genetic workflow。

## 运行机制

### 1. 固定训练预算

每个 individual 的训练时间固定为 **5 分钟**。这样不同模型大小、超参数和结构变体之间仍然可以在同一预算下比较。

主要指标是：

- **`val_bpb`**：越低越好

### 2. 显式状态持久化

controller 会维护这些关键文件：

- `population/current_generation.json`
  当前 generation 的排名结果，也是下次运行的恢复点
- `population/archive/`
  历史 generation 快照
- `results/runs.tsv`
  每个 individual 一行
- `artifacts/<individual_id>/experiment.json`
  渲染后的实验配置
- `artifacts/<individual_id>/registry_manifest.json`
  当次运行使用的 slot registry 标识
- `artifacts/<individual_id>/run.log`
  原始训练日志
- `artifacts/<individual_id>/summary.json`
  解析后的摘要

### 3. 多代遗传搜索

当前 controller 已经支持：

- 初始 seed population
- 从 `current_generation.json` 恢复
- elitism
- tournament selection
- numeric crossover
- 小幅 mutation
- generation 执行、排序与持久化

也就是说，它已经不是“伪 evolutionary shell”，而是一个能真正跨代运行的最小可用 genetic controller。

## 快速开始

**运行要求：**

- Apple Silicon Mac
- macOS / MPS
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/)

```bash
# 1. 安装依赖
uv sync

# 2. 准备数据和 tokenizer（一次性）
uv run prepare.py

# 3. 传统单次 smoke test（可选）
uv run train.py

# 4. 运行一个小规模 genetic generation
uv run evolve.py --population-size 2 --generation-limit 1
```

如果目录里已经有 `population/current_generation.json`，再次运行 `uv run evolve.py ...` 会自动从当前 generation 继续 breeding 下一代。

## 常用工作流

### 启动/继续一轮遗传搜索

```bash
uv run evolve.py --generation-limit 1
```

### 调试某个历史 individual

```bash
uv run train.py --experiment artifacts/<individual_id>/experiment.json
```

### 查看当前 population 状态

直接看：

- `population/current_generation.json`
- `results/runs.tsv`

### 查看失败个体

重点看：

- `artifacts/<individual_id>/run.log`
- `artifacts/<individual_id>/summary.json`

## 给 agent 的建议提示词

你可以把 agent 拉到仓库根目录，然后让它：

```text
Read program.md, inspect the current population state, and continue the genetic search run.
```

## 设计原则

- **固定预算优先**
  每个 individual 固定 5 分钟，比“跑到收敛”为主的系统更容易自动化比较。
- **训练 skeleton 稳定**
  `train.py` 负责单个 candidate 的执行，不再承担整个研究控制器。
- **controller 状态显式**
  generation、TSV、artifact 全都落盘，方便恢复和审计。
- **先顺序执行，后并行优化**
  v1 先保证正确性和可复现性，再考虑 future parallel backend。
- **保留 legacy debug 路径**
  `train.py` 仍然可单独跑，便于定位某个个体的问题。

## 平台说明

这个 fork 当前是 **macOS / Apple Silicon / MPS** 定位。

虽然 README 历史上提到过更广的平台支持，但当前代码实际上仍然强依赖 macOS + MPS 运行条件。把它理解为“面向 Mac 的 genetic-search fork”是更准确的。

## 项目结构

```text
prepare.py      — 数据准备、tokenizer、评估
train.py        — 单个 candidate 的训练 skeleton
evolve.py       — genetic controller 入口
evolution/      — schema、registry、ranking、artifacts、runner
program.md      — agent 工作契约
pyproject.toml  — 依赖定义
```

## 上游关系

这个仓库来自下面这条链路：

- `karpathy/autoresearch`
- `miolini/autoresearch-macos`
- `bingshuoguo/autoresearch-macos-genetic`

## License

MIT
