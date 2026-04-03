# autoresearch-macos-genetic

[简体中文文档](README.zh-CN.md)

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it run a genetic search loop overnight. The controller renders candidate experiments, trains each one for 5 minutes, ranks valid and invalid runs, archives the artifacts, and breeds the next generation. You wake up in the morning to a lineage of experiments and, hopefully, a better model family. The training code here is a simplified single-device implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is still that you are mostly programming the research process through `program.md`, but the default controller is now the explicit `evolve.py` population runner instead of an ad hoc keep-or-reset loop. A bit more context on the original project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Open source project worth to look at

Open source collabaration platform for agentic swarms in organizations and communityies. 

[SentientWave Automata](https://github.com/sentientwave/automata)

## How it works

The repo is still deliberately small, but the core pieces now have clearer boundaries:

- **`prepare.py`** — fixed constants, one-time data prep, tokenizer training, dataloader, and evaluation. Not modified during search.
- **`train.py`** — shared execution skeleton for a single candidate run. It supports legacy `uv run train.py` execution and `--experiment <path>` execution for the population runner.
- **`evolve.py`** — the primary genetic controller. It seeds or resumes the population, executes one or more generations, ranks individuals, and persists artifacts.
- **`evolution/`** — schema normalization, slot registry identity, ranking, artifact writers, and runner logic.
- **`program.md`** — baseline agent contract for the genetic workflow. This is the main file the human iterates on.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** Apple Silicon Mac (M1/M2/M3/M4 with Metal/MPS support), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Legacy single-run smoke test (~5 min)
uv run train.py

# 5. Run one small genetic generation (~population_size * 5 min)
uv run evolve.py --population-size 2 --generation-limit 1
```

The legacy `train.py` command is still useful for smoke tests and debugging. The primary controller is now `evolve.py`. Re-running `uv run evolve.py ...` after a completed generation resumes from `population/current_generation.json` and breeds the next generation automatically.

**Platforms support**. This fork officially supports **macOS (Apple Silicon / MPS)** and CPU environments, while preserving the original NVIDIA GPU support. It removes the hardcoded dependency on FlashAttention-3, falling back to PyTorch's native Scaled Dot Product Attention (SDPA) with manual sliding window causal masking when needed. It also features MPS-specific optimizations (disabling unsupported `torch.compile` paths, lowering memory batch sizes for Metal bounds, and precisely casting optimizer states) allowing you to run autonomous research agents directly on your Mac!

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Read program.md, inspect the current population state, and continue the genetic search run.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities
train.py        — shared training skeleton / single-run debug path
evolve.py       — genetic controller entrypoint
evolution/      — schema, registry, ranking, artifacts, runner
program.md      — agent instructions for the genetic loop
pyproject.toml  — dependencies
```

## Design choices

- **Explicit controller state.** `population/current_generation.json`, `results/runs.tsv`, and per-individual artifacts make lineage and replayability explicit.
- **Stable training skeleton.** `train.py` remains the execution harness for one candidate, while `evolve.py` owns population search and persistence.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Sequential executor first.** One individual runs at a time on one device. This keeps v1 simple and leaves room for a future parallel backend.
- **Self-contained.** No external services or orchestration stack. Just PyTorch, a few small packages, and explicit local artifacts.

## Platform support

This fork currently targets **macOS on Apple Silicon / MPS**. Both `prepare.py` and `train.py` enforce that runtime contract. If you want a broader multi-platform version, treat this repository as the macOS-specific genetic-search fork rather than a generic launcher.

If you're going to be using autoresearch on Apple Macbooks in particular, I'd recommend one of the forks below. On top of this, if you'd like half-decent results at such a small scale, I'd recommend this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) which is cleaner than what exists out there otherwise. It should be a drop in replacement because I have encoded it in exactly the same format. Any of your favorite coding agents should be able to do the swap :)

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)

## License

MIT
