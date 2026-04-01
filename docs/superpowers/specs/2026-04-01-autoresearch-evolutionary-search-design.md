# Autoresearch Evolutionary Search Design

## Summary

This design changes `autoresearch` from a greedy single-branch research loop into a constrained evolutionary search system.

The current system advances one incumbent model at a time: modify `train.py`, run one experiment, keep the change if `val_bpb` improves, otherwise revert. The new system keeps the same small, fixed-budget training harness, but replaces the search strategy with population-based evolution.

The target design is a hybrid approach:

- Use an explicit genome to encode numeric hyperparameters, categorical choices, and selected implementation variants.
- Keep `prepare.py` frozen as the source of data loading and evaluation truth.
- Refactor `train.py` into a stable training skeleton plus controlled evolvable slots.
- Allow the agent to invent new slot implementations, but only within bounded extension points.
- Rank individuals lexicographically: `val_bpb` first, then run stability, then memory, then complexity.

This preserves the original project philosophy of small, real, repeatable experiments while making search less myopic than hill climbing.

## Goals

- Replace greedy local search with population-based evolutionary search.
- Preserve the fixed 5-minute training budget and `val_bpb` comparison framework.
- Maintain reproducibility and heredity by making search state explicit.
- Allow bounded architectural creativity without allowing arbitrary uncontrolled code drift.
- Keep the system simple enough to run on a single machine with a fixed population size per generation.

## Non-Goals

- This is not a general NAS platform.
- This is not unrestricted code evolution over the entire repository.
- This does not change `prepare.py`, data prep, or evaluation semantics.
- This does not introduce distributed training, asynchronous cluster scheduling, or multi-objective Pareto optimization in the initial version.
- This does not attempt AST-level patch evolution in the first implementation.

## Current State

Today the repository has three functional layers:

- `prepare.py` provides constants, tokenizer loading, dataloading, and the authoritative `evaluate_bpb` path.
- `train.py` contains model definition, optimizer, schedules, and the training loop.
- `program.md` tells an agent to run a single-branch loop with keep-or-revert behavior.

The search policy is therefore a greedy accept-if-better loop. It is easy to operate, but it has standard hill-climbing weaknesses:

- it explores one incumbent at a time
- it has no persistent diversity
- it can get trapped in local optima
- it can discard promising but temporarily worse directions

## Proposed Architecture

The new system splits responsibilities into four layers.

### 1. Frozen Evaluation Layer

`prepare.py` remains unchanged. It continues to define:

- `MAX_SEQ_LEN`
- `TIME_BUDGET`
- tokenizer and dataloader behavior
- `evaluate_bpb`

This layer is the immutable environment and fitness evaluator.

### 2. Train Skeleton Layer

`train.py` is refactored from a freely edited script into a stable skeleton that:

- loads a genome or rendered experiment config
- selects slot implementations
- instantiates the model and optimizer according to that genome
- runs training for the same fixed budget
- prints the same summary metrics

The skeleton owns orchestration, not open-ended research logic.

### Rendering Mechanism

The initial implementation uses runtime config loading plus registry lookup, not code generation.

- The evolution runner writes one `experiment.json` file per individual under `artifacts/<individual_id>/experiment.json`.
- `train.py` accepts an explicit `--experiment <path>` argument. If omitted, it uses the baseline genome bundled with the repository.
- The experiment file contains:
  - `schema_version`
  - genome content
  - derived scalar values needed by the train skeleton
  - selected slot IDs
- `train.py` validates the experiment schema, resolves slot IDs through a Python registry, and instantiates the corresponding modules at runtime.
- No templating or source-to-source code generation is used in the first version.

This decision is on the critical path. It keeps the skeleton stable, makes artifacts inspectable, and avoids a separate code-generation layer.

### 3. Evolution Layer

A new population management layer owns:

- generation creation
- parent selection
- crossover
- mutation
- elitism
- experiment scheduling
- result aggregation
- lineage tracking

This layer becomes the new search controller.

### 4. Agent Innovation Layer

The agent no longer mutates `train.py` arbitrarily. Instead, it can:

- propose new slot implementations
- propose new mutation operators
- propose search-bias adjustments
- analyze lineage and identify dead or promising families

The agent becomes a constrained inventor and curator, not the direct executor of every code edit.

## Genome Design

Each individual must have an explicit genome. The genome is the unit of heredity and must be serializable.

The genome is split into three gene families.

### Schema Versioning and IDs

Every genome record must include:

- `schema_version`
- `individual_id`
- `generation`
- `parents`

The initial schema version is `1.0`.

The initial ID rule is:

- `individual_id = g{generation:04d}-i{ordinal:02d}-{genome_hash8}`
- `genome_hash8` is the first 8 hex characters of the canonical JSON hash of the genome payload excluding run metadata
- if a collision still occurs, append `-r{n:02d}`

This keeps IDs readable, generation-local, and globally unique enough for the first implementation.

### Numeric Genes

These are scalar or integer parameters such as:

- `depth`
- `aspect_ratio`
- `embedding_lr`
- `unembedding_lr`
- `matrix_lr`
- `scalar_lr`
- `weight_decay`
- `warmup_ratio`
- `warmdown_ratio`
- `final_lr_frac`
- `total_batch_size`
- `device_batch_size`

These map directly to the existing hyperparameter region in `train.py`.

### Categorical Genes

These choose from a bounded set of legal values such as:

- `window_pattern`
- `head_dim`
- `kv_head_mode`
- `logit_softcap`
- `optimizer_grouping_strategy`

These preserve comparability while still allowing structural variety.

### Slot Genes

Each slot gene points to a specific implementation ID from a controlled implementation library. Initial slots should include:

- `attention`
- `mlp`
- `schedule`
- `init`

Future slots can include optimizer policy variants, but the first version should avoid slotting too much of the training loop itself.

### Example Genome Shape

```json
{
  "schema_version": "1.0",
  "individual_id": "g0042-i03-a1b2c3d4",
  "generation": 42,
  "parents": ["g0041-i01-b7c91a2e", "g0041-i06-9ad0c4bf"],
  "numeric": {
    "depth": 6,
    "aspect_ratio": 80,
    "embedding_lr": 0.45,
    "matrix_lr": 0.03
  },
  "categorical": {
    "window_pattern": "SLL",
    "head_dim": 128
  },
  "slots": {
    "attention": "sdpa_windowed_v1",
    "mlp": "relu2_mlp_v1",
    "schedule": "warmdown_linear_v1",
    "init": "default_init_v1"
  },
  "metadata": {
    "created_by": "crossover+mutation",
    "notes": "attention from parent A, schedule from parent B"
  }
}
```

## Slot System

Slots are controlled extension points inside the training skeleton.

The first implementation should slot only model-local or schedule-local behavior, not full training orchestration. This keeps crash risk manageable.

### Slot Boundaries

Recommended first-wave slots:

- `attention`
  - example variants: baseline SDPA, SDPA with value residual gating variants, alternative short-window patterns
- `mlp`
  - example variants: `relu^2`, SwiGLU, GEGLU, different expansion ratios
- `schedule`
  - example variants: linear warmdown, cosine decay, flat-plus-cooldown
- `init`
  - example variants: current init, alternative projection init, scalar init adjustments

Deferred slots:

- optimizer implementation internals
- dataloader behavior
- training loop control flow

These are deferred because they expand the failure surface disproportionately.

### Slot Library Rules

- Every slot implementation has a stable ID.
- Every slot implementation must satisfy a documented interface.
- Slot implementations are reusable across generations.
- New implementations added by the agent are versioned rather than overwritten.

This makes evolutionary reuse possible and lineage interpretable.

### Slot Interface Contracts

The first-wave slots must implement explicit Python interfaces.

#### Attention Slot

Factory signature:

```python
def build_attention(slot_config: dict, model_config: GPTConfig, layer_idx: int) -> nn.Module: ...
```

Returned module contract:

```python
def forward(
    self,
    x: torch.Tensor,
    *,
    ve: torch.Tensor | None,
    cos_sin: tuple[torch.Tensor, torch.Tensor],
    window_size: tuple[int, int],
) -> torch.Tensor: ...
```

Invariants:

- input `x` shape is `[B, T, C]`
- output shape must also be `[B, T, C]`
- no parameter allocation inside `forward`
- module must preserve device placement
- module may assume causal language-model semantics only

#### MLP Slot

Factory signature:

```python
def build_mlp(slot_config: dict, model_config: GPTConfig, layer_idx: int) -> nn.Module: ...
```

Returned module contract:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

Invariants:

- input and output shapes are both `[B, T, C]`
- no parameter allocation inside `forward`
- device and dtype must remain consistent with the surrounding block

#### Schedule Slot

Factory signature:

```python
def build_schedule(slot_config: dict, genome: dict) -> SchedulePolicy: ...
```

Returned object contract:

```python
class SchedulePolicy(Protocol):
    def lr_multiplier(self, progress: float) -> float: ...
    def muon_momentum(self, step: int) -> float: ...
    def weight_decay(self, progress: float, base_weight_decay: float) -> float: ...
```

Invariants:

- `progress` is always clamped to `[0.0, 1.0]`
- returned values must be finite scalars
- schedule policy must be pure with respect to run state

#### Init Slot

Factory signature:

```python
def apply_init(model: nn.Module, slot_config: dict, model_config: GPTConfig) -> None: ...
```

Invariants:

- initialization happens in place
- init logic cannot change model topology
- init must leave every trainable parameter initialized exactly once

These contracts are the blocking interface definitions needed before Phase 2 implementation begins.

## Evolution Loop

The initial search strategy is a fixed-size generational GA with elitism.

### Per-Generation Flow

1. Start from generation `g` with population size `P`.
2. Preserve top `E` elite individuals unchanged.
3. Select parents from the top-ranked region using tournament selection or rank-based sampling.
4. Generate `P - E` offspring through crossover and mutation.
5. Optionally reserve a small number of offspring slots for agent-invented novel variants.
6. Render each genome into a runnable experiment.
7. Run each individual under the same fixed 5-minute budget.
8. Parse and store output metrics.
9. Rank the completed population.
10. Advance to generation `g + 1`.

### Execution Model

The first implementation executes individuals sequentially on one accelerator.

- one individual runs at a time
- the default executor is single-process and single-device
- generation wall-clock time is therefore approximately `population_size * 5 minutes` plus startup and evaluation overhead

With the recommended initial defaults, this means roughly 30 to 40 minutes per generation.

Parallel execution across multiple devices or mixed executors is explicitly deferred, but the runner should keep its executor boundary narrow enough that a future parallel backend can replace the sequential executor without rewriting genome or ranking logic.

### Selection

The recommended initial policy is tournament selection with a small tournament size such as 3 or 4.

Rationale:

- simple to implement
- robust under noisy evaluations
- less fragile than probability proportional to raw score

### Crossover

Recommended crossover rules:

- numeric genes: per-gene random alpha blend with `alpha ~ Uniform(0.25, 0.75)`
- categorical genes: parent-wise inheritance
- slot genes: parent-wise inheritance by slot

The child should not differ in every field. Heredity must remain visible.

### Mutation

Recommended mutation rules:

- numeric genes: bounded perturbation or proportional scaling
- categorical genes: discrete jump within allowed values
- slot genes: swap to another valid implementation ID

The initial mutation rate is `0.15` at the individual level. Each mutated individual should change only a small number of genes so offspring remain traceable to parents.

### Elitism

At least one elite individual should be preserved each generation. Preserving two elites is reasonable for the first implementation.

This prevents population collapse from evaluation noise.

## Fitness and Ranking

The ranking scheme is lexicographic rather than full Pareto optimization.

### Ranking Order

1. Lower `val_bpb`
2. Successful, stable completion over crash or invalid run
3. Lower peak memory usage
4. Lower complexity score

This matches the earlier design decision: `val_bpb` is primary, but not the only thing that matters.

### Stability Rules

An individual is unstable if:

- the run crashes
- the summary output is incomplete
- the training loop exceeds the allowed timeout
- the raw training loss becomes `NaN` or `Inf`
- the raw training loss exceeds `100` after the first 10 warmup steps

Unstable individuals rank below any valid run regardless of nominal partial output.

### Complexity Score

The initial complexity score is a deterministic weighted heuristic:

```text
complexity_score =
    1.0 * non_default_slot_count +
    2.0 * custom_slot_count +
    4.0 * max(0, (num_params / baseline_num_params) - 1.0)
```

Definitions:

- `non_default_slot_count`: number of slot selections that differ from the baseline genome
- `custom_slot_count`: number of selected slot IDs marked as agent-authored or experimental
- `baseline_num_params`: parameter count of the repository baseline genome under the current skeleton release

The score is rounded to 3 decimal places before ranking. Lower is better.

This operationalizes the existing simplicity principle from the current project.

## Data and Artifact Model

The current flat `results.tsv` is not enough for population search. The new system should persist richer state.

### Required Artifacts

- `population/current_generation.json`
  - current generation genomes and lineage
- `population/archive/`
  - prior generation snapshots
- `results/runs.tsv`
  - one row per executed individual
- `artifacts/<individual_id>/`
  - rendered config, run log, summary output, optional derived metadata
- `library/slots/`
  - slot implementation registry and code

### Retention Policy

- `population/archive/*.json` generation snapshots are retained indefinitely in v1 because the metadata is small.
- `artifacts/<individual_id>/experiment.json` is retained indefinitely for reproducibility.
- large run logs may be pruned or compressed after summary extraction unless the individual is:
  - an elite
  - a generation winner
  - a failed run needed for debugging a new experimental slot

### Run Record Fields

Each run record should include at minimum:

- `individual_id`
- `schema_version`
- `generation`
- `parents`
- `genome_hash`
- `val_bpb`
- `status`
- `training_seconds`
- `total_seconds`
- `peak_vram_mb`
- `num_params`
- `complexity_score`
- `description`

This is the minimum information needed to analyze lineage and reproduce results.

## Agent Responsibilities

The agent role changes materially in the new design.

### Automation Boundary

The automated evolution loop owns:

- parent selection
- crossover
- mutation
- experiment rendering
- execution
- ranking
- archive persistence

AI assistance is opt-in and out-of-band relative to the per-generation loop. In the initial version, the agent does not autonomously inject new active slot implementations during the same automated generation cycle.

### Breeder

The agent can propose better mutation or crossover heuristics based on observed search dynamics.

### Inventor

The agent can add new slot implementations within controlled extension boundaries.

### Curator

The agent can inspect generations and recommend:

- retiring dead families
- increasing exploration around promising families
- suppressing unstable implementation branches

The agent should not directly rewrite arbitrary skeleton logic during normal search.

## Failure Handling

The design must explicitly prevent evolutionary drift into an unusable codebase.

### Guardrails

- The train skeleton must remain stable and minimally edited.
- Slot implementations must pass interface validation before entering the active library.
- Crash-heavy new slot variants should be quarantined after repeated failures.
- Timeout and invalid-output handling must be first-class, not ad hoc.

### Slot Validation Gate

Agent-authored slot implementations do not enter the active library directly. They enter a staging area first.

Promotion from staging to active requires all of:

1. syntax and import validation
2. registry signature validation against the slot contract
3. successful construction test on the current runtime device
4. a tiny forward or forward-backward smoke test for the slot type
5. successful dry-run integration under the train skeleton for at least 1 to 2 optimizer steps
6. explicit operator promotion in v1

The automated evolution loop may only sample from the active slot library.

### Failure Classification

Runs should be classified as:

- `success`
- `crash`
- `timeout`
- `invalid_output`
- `unstable`

This is more useful than a single `crash` bucket once the system is population-based.

## Testing Strategy

The initial implementation needs fast validation before long experiments.

### Unit-Level Coverage

- genome schema validation
- slot registry lookup and interface checks
- crossover correctness
- mutation bound correctness
- ranking logic
- lineage serialization

### Integration-Level Coverage

- render one genome into a runnable config
- run a tiny smoke evaluation path
- execute one small generation with a tiny population and mocked metrics

### End-to-End Validation

- run a real generation with a small population
- verify artifact layout
- verify elite carryover and offspring creation
- verify ranking and archive persistence

## Migration Plan

Implementation should happen in phases to limit risk.

### Phase 1

Extract the current editable hyperparameters from `train.py` into an explicit genome/config object while preserving current behavior. This phase also introduces:

- schema versioning
- `individual_id` generation
- `experiment.json` rendering
- runtime registry lookup in `train.py`

### Phase 2

Introduce the first slot family set:

- `mlp`
- `schedule`
- optionally `attention`

Keep the rest of the skeleton unchanged.

### Phase 3

Implement the population runner:

- generation state
- selection
- crossover
- mutation
- elitism
- result recording

### Phase 4

Add agent-authored slot invention and slot library versioning.

### Phase 5

Tune search heuristics only after the lineage and artifact model are stable.

## Key Risks

- Too much freedom in slots will destroy comparability and reproducibility.
- Too little freedom will reduce the system to parameter search with extra overhead.
- Overly aggressive mutation will erase heredity.
- Population size that is too large will make runtime impractical on a single machine.
- Complexity scoring that is too elaborate will become a second optimizer to debug.

The first version should optimize for control and observability, not maximum novelty.

## Recommended Initial Defaults

The first implementation should assume:

- `population_size = 6`
- `elite_count = 2`
- `tournament_size = 3`
- `crossover_rate = 0.75`
- `mutation_rate = 0.15`
- `innovation_slots = 1`
- sequential single-device execution
- lexicographic ranking instead of Pareto fronts

The suggested operating range is population sizes from 4 to 8. `6` is the default because it preserves diversity while keeping generation time within roughly half an hour on the current 5-minute evaluation budget.

## Open Design Decisions Resolved Here

The following choices are considered decided for this design:

- use a hybrid evolutionary system, not pure greedy search
- use explicit genomes
- use controlled evolvable slots instead of unrestricted full-file rewriting
- use runtime JSON experiment rendering plus slot registry lookup
- optimize lexicographically with `val_bpb` first
- use fixed population size per generation
- stage the implementation rather than replacing everything at once

## Acceptance Criteria

This design is successful if the implemented system can:

- represent each candidate as a serializable genome
- generate a new population from prior populations without arbitrary manual edits
- render genomes into repeatable experiments
- rank individuals consistently using the agreed ordering
- preserve lineage and artifacts across generations
- allow bounded agent-authored innovation without breaking the search loop

## Out of Scope for the First Spec

- asynchronous island models
- Pareto frontier maintenance
- distributed workers
- AST-level genetic operators
- unrestricted code synthesis across the entire training file
- automatic research paper ingestion to generate operators
