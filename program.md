# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up or resume an experiment, work with the user to:

1. **Agree on the branch context**: if this is a fresh run, create a dedicated branch such as `autoresearch/<tag>`. If this is a continuation, stay on the existing research branch.
2. **Read the in-scope files**: the repo is still small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — shared execution skeleton for one candidate run.
   - `evolve.py` — primary genetic controller.
   - `evolution/` — schema, registry, ranking, artifacts, and runner logic.
3. **Verify data exists**: check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
4. **Verify the worktree is clean before launch**: `uv run evolve.py` refuses to run on a dirty worktree. If you intentionally change controller or model code, commit those changes before launching the next generation.
5. **Confirm resume state**:
   - if `population/current_generation.json` does not exist, the first controller run seeds generation 0 automatically
   - if it exists, the controller resumes from it and breeds the next generation automatically
6. **Confirm and go**: once setup looks good, start or continue the genetic loop.

## Experimentation

The primary controller is now:

```bash
uv run evolve.py --generation-limit 1
```

Each individual run inside the population still trains for a **fixed 5-minute time budget**. A generation therefore takes roughly:

```text
population_size * 5 minutes + startup/eval overhead
```

**Primary workflow**

- `uv run evolve.py` is the main genetic search loop
- `uv run train.py` is legacy compatibility mode for single-run debugging
- `uv run train.py --experiment artifacts/<individual_id>/experiment.json` is the precise rerun path for one saved individual

**What you CAN do**

- Modify `train.py` when you are changing the shared execution skeleton
- Modify `evolve.py` or `evolution/*.py` when you are changing the search process, ranking, persistence, or generation logic
- Inspect and compare population artifacts across generations

**What you CANNOT do**

- Modify `prepare.py`. It is read-only and contains the fixed evaluation harness.
- Install new packages or add dependencies. You can only use what is already in `pyproject.toml`.
- Hand-edit `results/runs.tsv`, `population/current_generation.json`, or archived artifacts to fake controller state.

**Goal**

Get the lowest `val_bpb` subject to stability and reasonable complexity. Since the controller is population-based, you no longer keep or discard candidates manually. The automated loop owns:

- selection
- crossover
- mutation
- execution
- ranking
- archive persistence

**Simplicity criterion**

All else being equal, simpler is better. Small gains that add ugly complexity are not worth much. Small gains that come from deletion or simplification are strong wins. Use the artifact history and ranking metadata to judge whether more complex branches deserve to survive.

## Output and artifacts

The controller owns these files:

- `population/current_generation.json` — the ranked population state the next run resumes from
- `population/archive/` — archived generation snapshots
- `results/runs.tsv` — one row per executed individual
- `artifacts/<individual_id>/experiment.json` — rendered experiment config
- `artifacts/<individual_id>/registry_manifest.json` — registry identity used for the run
- `artifacts/<individual_id>/run.log` — raw execution log
- `artifacts/<individual_id>/summary.json` — parsed execution summary

When debugging one individual, inspect its artifact directory first. Use the saved `experiment.json` to rerun the candidate exactly.

## The controller loop

LOOP FOREVER:

1. Inspect the current git state and confirm the worktree is clean.
2. Inspect `population/current_generation.json` and `results/runs.tsv` to understand the latest ranked population.
3. If you are changing controller or model code, commit those changes before launching the next generation.
4. Run one more generation with `uv run evolve.py --generation-limit 1`.
5. Inspect the new winner and the invalid runs.
6. For failures, inspect `artifacts/<individual_id>/run.log` and `summary.json`.
7. If needed, rerun a specific saved candidate with `uv run train.py --experiment artifacts/<individual_id>/experiment.json`.
8. Continue indefinitely until the human interrupts you.

**Important differences from the old greedy loop**

- Do not manually append TSV rows; the controller does that.
- Do not `git reset` after a bad candidate; invalid and weak individuals are filtered by ranking, not by branch rewrites.
- Do not decide keep/discard on a single commit at a time; reason about families of individuals across generations.

**Timeouts and crashes**

- Treat any controller timeout, crash, or invalid output as a failed individual, not as a reason to corrupt the population state manually.
- Use the saved artifacts to debug repeated failures.
- If the controller logic itself is broken, fix it, commit the fix, and then resume.

**NEVER STOP**

Once the genetic loop has begun, do not pause to ask the human whether to continue. Keep running generations, inspecting artifacts, and improving the search process until interrupted.
