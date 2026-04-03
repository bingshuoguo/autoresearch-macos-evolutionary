"""Sequential genetic-search runner primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .artifacts import (
    append_run_record,
    write_experiment_artifact,
    write_registry_manifest_artifact,
    write_run_log_artifact,
    write_summary_artifact,
)
from .ranking import compute_complexity_score, rank_population
from .registry import SlotManifestEntry, load_active_registry, slot_registry_hash
from .schema import (
    SCHEMA_VERSION,
    build_individual_id,
    canonical_genome_json,
    normalize_genome,
)

DEFAULT_POPULATION_SIZE = 6
DEFAULT_ELITE_COUNT = 2
DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_GENERATION_LIMIT = 1
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_REGISTRY_MANIFEST = Path("library/slots/active_registry.json")
NUMERIC_BLEND_ALPHA_MIN = 0.25
NUMERIC_BLEND_ALPHA_MAX = 0.75
INDIVIDUAL_MUTATION_RATE = 0.15
INTEGER_NUMERIC_FIELDS = {
    "depth",
    "aspect_ratio",
    "total_batch_size",
    "device_batch_size",
}

BASELINE_GENOME_TEMPLATE = {
    "numeric": {
        "depth": 4,
        "aspect_ratio": 64,
        "total_batch_size": 2**16,
        "device_batch_size": 16,
        "embedding_lr": 0.6,
        "unembedding_lr": 0.004,
        "matrix_lr": 0.04,
        "scalar_lr": 0.5,
        "weight_decay": 0.2,
        "warmup_ratio": 0.0,
        "warmdown_ratio": 0.5,
        "final_lr_frac": 0.0,
    },
    "categorical": {
        "window_pattern": "L",
        "head_dim": 128,
        "kv_head_mode": "mha",
        "logit_softcap": "tanh_15",
        "optimizer_grouping_strategy": "baseline_v1",
    },
    "slots": {
        "attention": "baseline_attention_v1",
        "mlp": "baseline_mlp_v1",
        "schedule": "baseline_schedule_v1",
        "init": "baseline_init_v1",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the initial genetic runner."""

    parser = argparse.ArgumentParser(description="Autoresearch genetic runner")
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    parser.add_argument("--tournament-size", type=int, default=DEFAULT_TOURNAMENT_SIZE)
    parser.add_argument("--generation-limit", type=int, default=DEFAULT_GENERATION_LIMIT)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--registry-manifest",
        type=Path,
        default=DEFAULT_REGISTRY_MANIFEST,
        help="Path to the active slot registry manifest",
    )
    args = parser.parse_args(argv)

    if args.population_size < 1:
        raise SystemExit("--population-size must be >= 1")
    if args.elite_count < 0:
        raise SystemExit("--elite-count must be >= 0")
    if args.elite_count > args.population_size:
        raise SystemExit("--elite-count must be <= --population-size")
    if args.tournament_size < 1:
        raise SystemExit("--tournament-size must be >= 1")
    if args.generation_limit < 1:
        raise SystemExit("--generation-limit must be >= 1")
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be >= 1")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be >= 1")

    return args


def tournament_select(population: list[dict[str, Any]], tournament_size: int, rng_seed: int) -> dict[str, Any]:
    """Select one parent from a tournament sampled from the ranked population."""

    if not population:
        raise ValueError("population must not be empty")
    if tournament_size < 1:
        raise ValueError("tournament_size must be >= 1")
    if tournament_size > len(population):
        raise ValueError("tournament_size must be <= population size")

    rng = random.Random(rng_seed)
    sample = rng.sample(population, k=tournament_size)
    return min(sample, key=lambda item: item["fitness_rank"])


def write_generation_state(
    root: Path | str,
    *,
    generation: int,
    population: list[dict[str, Any]],
    repo_commit: str | None = None,
    repo_tree_hash: str | None = None,
    slot_registry_hash_value: str | None = None,
) -> tuple[Path, Path]:
    """Persist the current generation snapshot and an archived copy."""

    population_dir = Path(root) / "population"
    archive_dir = population_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "generation": generation,
        "population": population,
    }
    if repo_commit is not None:
        snapshot["repo_commit"] = repo_commit
    if repo_tree_hash is not None:
        snapshot["repo_tree_hash"] = repo_tree_hash
    if slot_registry_hash_value is not None:
        snapshot["slot_registry_hash"] = slot_registry_hash_value
    payload = json.dumps(snapshot, sort_keys=True, indent=2) + "\n"

    current_path = population_dir / "current_generation.json"
    current_path.write_text(payload, encoding="utf-8")

    archive_path = archive_dir / f"g{generation:04d}.json"
    archive_path.write_text(payload, encoding="utf-8")

    return current_path, archive_path


def load_current_generation_state(root: Path | str) -> dict[str, Any] | None:
    """Load the latest persisted generation state, if present."""

    current_path = Path(root) / "population" / "current_generation.json"
    if not current_path.exists():
        return None
    return json.loads(current_path.read_text(encoding="utf-8"))


def _core_genome(genome: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": genome.get("schema_version", SCHEMA_VERSION),
        "generation": genome["generation"],
        "parents": list(genome.get("parents", [])),
        "numeric": dict(genome["numeric"]),
        "categorical": dict(genome["categorical"]),
        "slots": dict(genome["slots"]),
        "metadata": dict(genome.get("metadata", {})),
    }


def _annotate_genome(
    genome: dict[str, Any],
    *,
    generation: int,
    ordinal: int,
    max_seq_len: int,
) -> dict[str, Any]:
    raw = _core_genome(genome)
    raw["generation"] = generation
    normalized = normalize_genome(raw, max_seq_len=max_seq_len)
    canonical_json = canonical_genome_json(normalized)
    genome_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    normalized["individual_id"] = build_individual_id(generation, ordinal, canonical_json)
    normalized["genome_hash"] = genome_hash
    normalized["metadata"] = dict(raw.get("metadata", {}))
    return normalized


def _scale_numeric(value: float, factor: float) -> float:
    return round(value * factor, 6)


def _quantize_total_batch_size(total_batch_size: int, device_batch_size: int, max_seq_len: int) -> int:
    step_tokens = max(device_batch_size * max_seq_len, 1)
    accum_steps = max(1, round(total_batch_size / step_tokens))
    return accum_steps * step_tokens


def _source_genome(record: dict[str, Any]) -> dict[str, Any]:
    genome = record.get("genome", record)
    return _core_genome(genome)


def _blend_numeric_value(field_name: str, left: Any, right: Any, rng: random.Random) -> Any:
    alpha = rng.uniform(NUMERIC_BLEND_ALPHA_MIN, NUMERIC_BLEND_ALPHA_MAX)
    blended = alpha * float(left) + (1.0 - alpha) * float(right)
    if field_name in INTEGER_NUMERIC_FIELDS:
        return max(1, int(round(blended)))
    return round(blended, 6)


def _crossover_genomes(
    left_parent: dict[str, Any],
    right_parent: dict[str, Any],
    *,
    generation: int,
    rng: random.Random,
    max_seq_len: int,
    slot_options: dict[str, list[str]],
) -> dict[str, Any]:
    left = _source_genome(left_parent)
    right = _source_genome(right_parent)
    child = {
        "schema_version": SCHEMA_VERSION,
        "generation": generation,
        "parents": [
            left_parent["individual_id"],
            right_parent["individual_id"],
        ],
        "numeric": {},
        "categorical": {},
        "slots": {},
        "metadata": {
            "created_by": "crossover+mutation",
            "notes": f"parents={left_parent['individual_id']},{right_parent['individual_id']}",
        },
    }

    for field_name in left["numeric"]:
        child["numeric"][field_name] = _blend_numeric_value(
            field_name,
            left["numeric"][field_name],
            right["numeric"][field_name],
            rng,
        )

    for field_name in left["categorical"]:
        child["categorical"][field_name] = rng.choice(
            [left["categorical"][field_name], right["categorical"][field_name]]
        )

    for slot_name in left["slots"]:
        inherited = rng.choice([left["slots"][slot_name], right["slots"][slot_name]])
        child["slots"][slot_name] = inherited
        if slot_name in slot_options and inherited not in slot_options[slot_name]:
            child["slots"][slot_name] = slot_options[slot_name][0]

    child["numeric"]["depth"] = max(1, int(child["numeric"]["depth"]))
    child["numeric"]["aspect_ratio"] = max(1, int(child["numeric"]["aspect_ratio"]))
    child["numeric"]["device_batch_size"] = max(1, int(child["numeric"]["device_batch_size"]))
    child["numeric"]["total_batch_size"] = _quantize_total_batch_size(
        int(child["numeric"]["total_batch_size"]),
        int(child["numeric"]["device_batch_size"]),
        max_seq_len,
    )
    return child


def _mutate_child_genome(
    genome: dict[str, Any],
    *,
    rng: random.Random,
    max_seq_len: int,
    slot_options: dict[str, list[str]],
) -> dict[str, Any]:
    child = _core_genome(genome)
    if rng.random() >= INDIVIDUAL_MUTATION_RATE:
        return child

    mutation_kind = rng.choice(["numeric", "categorical", "slot"])
    if mutation_kind == "numeric":
        field_name = rng.choice(list(child["numeric"]))
        if field_name == "depth":
            child["numeric"][field_name] = max(1, int(child["numeric"][field_name]) + rng.choice([-1, 1]))
        elif field_name == "aspect_ratio":
            child["numeric"][field_name] = max(1, int(child["numeric"][field_name]) + rng.choice([-16, 16]))
        elif field_name == "device_batch_size":
            child["numeric"][field_name] = max(1, int(child["numeric"][field_name]) + rng.choice([-8, 8]))
        elif field_name == "total_batch_size":
            child["numeric"][field_name] = _quantize_total_batch_size(
                int(child["numeric"][field_name]) * rng.choice([1, 2]),
                int(child["numeric"]["device_batch_size"]),
                max_seq_len,
            )
        else:
            child["numeric"][field_name] = _scale_numeric(
                float(child["numeric"][field_name]),
                rng.choice([0.8, 1.2]),
            )
    elif mutation_kind == "categorical":
        field_name = rng.choice(list(child["categorical"]))
        if field_name == "window_pattern":
            child["categorical"][field_name] = rng.choice(["L", "S", "SL", "LS"])
    else:
        slot_name = rng.choice(list(child["slots"]))
        options = slot_options.get(slot_name, [child["slots"][slot_name]])
        if options:
            child["slots"][slot_name] = rng.choice(options)

    child["numeric"]["total_batch_size"] = _quantize_total_batch_size(
        int(child["numeric"]["total_batch_size"]),
        int(child["numeric"]["device_batch_size"]),
        max_seq_len,
    )
    return child


def _mutate_seed_genome(
    baseline: dict[str, Any],
    *,
    rng: random.Random,
    max_seq_len: int,
) -> dict[str, Any]:
    child = _core_genome(baseline)
    child["metadata"] = {
        "created_by": "seed_mutation",
        "notes": "baseline-derived seed mutation",
    }

    child["numeric"]["depth"] = max(1, child["numeric"]["depth"] + rng.choice([-1, 0, 1]))
    child["numeric"]["aspect_ratio"] = rng.choice([48, 64, 80, 96])
    child["numeric"]["device_batch_size"] = rng.choice([8, 16])
    accum_steps = rng.choice([1, 2, 4])
    child["numeric"]["total_batch_size"] = (
        child["numeric"]["device_batch_size"] * max_seq_len * accum_steps
    )
    child["numeric"]["embedding_lr"] = _scale_numeric(
        child["numeric"]["embedding_lr"],
        rng.choice([0.8, 1.0, 1.2]),
    )
    child["numeric"]["matrix_lr"] = _scale_numeric(
        child["numeric"]["matrix_lr"],
        rng.choice([0.75, 1.0, 1.25]),
    )
    child["numeric"]["warmup_ratio"] = rng.choice([0.0, 0.05, 0.1])
    child["numeric"]["warmdown_ratio"] = rng.choice([0.25, 0.5, 0.75])
    child["numeric"]["final_lr_frac"] = rng.choice([0.0, 0.05, 0.1])
    child["categorical"]["window_pattern"] = rng.choice(["L", "S", "SL", "LS"])
    return child


def create_initial_population(
    *,
    population_size: int,
    generation: int,
    rng_seed: int,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    """Create the seed population for a full generation run."""

    if population_size < 1:
        raise ValueError("population_size must be >= 1")

    baseline = {
        "schema_version": SCHEMA_VERSION,
        "generation": generation,
        "parents": [],
        "numeric": dict(BASELINE_GENOME_TEMPLATE["numeric"]),
        "categorical": dict(BASELINE_GENOME_TEMPLATE["categorical"]),
        "slots": dict(BASELINE_GENOME_TEMPLATE["slots"]),
        "metadata": {
            "created_by": "baseline_seed",
            "notes": "repository baseline seed genome",
        },
    }
    population: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    rng = random.Random(rng_seed)
    ordinal = 1

    while len(population) < population_size:
        if ordinal == 1:
            candidate = baseline
        else:
            candidate = _mutate_seed_genome(baseline, rng=rng, max_seq_len=max_seq_len)

        annotated = _annotate_genome(
            candidate,
            generation=generation,
            ordinal=ordinal,
            max_seq_len=max_seq_len,
        )
        if annotated["individual_id"] not in seen_ids:
            seen_ids.add(annotated["individual_id"])
            population.append(annotated)
        ordinal += 1

    return population


def breed_next_generation(
    ranked_population: list[dict[str, Any]],
    *,
    generation: int,
    population_size: int,
    elite_count: int,
    tournament_size: int,
    rng_seed: int,
    max_seq_len: int,
    registry_entries: list[SlotManifestEntry] | None = None,
) -> list[dict[str, Any]]:
    """Breed the next generation from a ranked prior population."""

    if not ranked_population:
        raise ValueError("ranked_population must not be empty")
    if elite_count < 0 or elite_count > population_size:
        raise ValueError("elite_count must satisfy 0 <= elite_count <= population_size")

    rng = random.Random(rng_seed)
    slot_options: dict[str, list[str]] = {}
    if registry_entries:
        for entry in registry_entries:
            slot_options.setdefault(entry.slot_type, []).append(entry.slot_id)

    next_population: list[dict[str, Any]] = []
    for ordinal, elite in enumerate(ranked_population[:elite_count], start=1):
        elite_genome = _source_genome(elite)
        elite_genome["generation"] = generation
        elite_genome["parents"] = [elite["individual_id"]]
        elite_genome["metadata"] = {
            "created_by": "elitism",
            "notes": f"elite copy of {elite['individual_id']}",
        }
        next_population.append(
            _annotate_genome(
                elite_genome,
                generation=generation,
                ordinal=ordinal,
                max_seq_len=max_seq_len,
            )
        )

    ordinal = elite_count + 1
    while len(next_population) < population_size:
        left_parent = tournament_select(
            ranked_population,
            tournament_size=min(tournament_size, len(ranked_population)),
            rng_seed=rng.randint(0, 1_000_000_000),
        )
        right_parent = tournament_select(
            ranked_population,
            tournament_size=min(tournament_size, len(ranked_population)),
            rng_seed=rng.randint(0, 1_000_000_000),
        )
        if len(ranked_population) > 1 and right_parent["individual_id"] == left_parent["individual_id"]:
            alternatives = [
                candidate
                for candidate in ranked_population
                if candidate["individual_id"] != left_parent["individual_id"]
            ]
            right_parent = rng.choice(alternatives)

        child = _crossover_genomes(
            left_parent,
            right_parent,
            generation=generation,
            rng=rng,
            max_seq_len=max_seq_len,
            slot_options=slot_options,
        )
        child = _mutate_child_genome(
            child,
            rng=rng,
            max_seq_len=max_seq_len,
            slot_options=slot_options,
        )
        next_population.append(
            _annotate_genome(
                child,
                generation=generation,
                ordinal=ordinal,
                max_seq_len=max_seq_len,
            )
        )
        ordinal += 1

    return next_population


def parse_training_summary(log_output: str) -> dict[str, Any]:
    """Parse the summary block emitted by train.py into a structured result."""

    parsed: dict[str, str] = {}
    for line in log_output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()

    required = {
        "val_bpb",
        "training_seconds",
        "total_seconds",
        "peak_vram_mb",
        "num_params_M",
    }
    if not required.issubset(parsed):
        return {
            "status": "invalid_output",
            "val_bpb": None,
            "training_seconds": None,
            "total_seconds": None,
            "peak_vram_mb": None,
            "num_params": None,
            "description": "missing summary fields",
            "log_output": log_output,
        }

    peak_vram_mb = float(parsed["peak_vram_mb"])
    if peak_vram_mb == 0.0:
        peak_vram_mb = None

    return {
        "status": "success",
        "val_bpb": float(parsed["val_bpb"]),
        "training_seconds": float(parsed["training_seconds"]),
        "total_seconds": float(parsed["total_seconds"]),
        "peak_vram_mb": peak_vram_mb,
        "num_params": int(round(float(parsed["num_params_M"]) * 1_000_000)),
        "description": "completed",
        "log_output": log_output,
    }


def execute_experiment(
    experiment_path: Path,
    timeout_seconds: int,
    *,
    cwd: Path | str,
) -> dict[str, Any]:
    """Run train.py for one experiment artifact and classify the outcome."""

    try:
        result = subprocess.run(
            ["uv", "run", "python", "train.py", "--experiment", str(experiment_path)],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + (exc.stderr or "")
        return {
            "status": "timeout",
            "val_bpb": None,
            "training_seconds": None,
            "total_seconds": None,
            "peak_vram_mb": None,
            "num_params": None,
            "description": "run timed out",
            "log_output": combined,
        }

    combined_output = result.stdout + result.stderr
    if result.returncode == 0:
        return parse_training_summary(combined_output)
    if "FAIL" in combined_output:
        return {
            "status": "unstable",
            "val_bpb": None,
            "training_seconds": None,
            "total_seconds": None,
            "peak_vram_mb": None,
            "num_params": None,
            "description": "loss diverged",
            "log_output": combined_output,
        }
    return {
        "status": "crash",
        "val_bpb": None,
        "training_seconds": None,
        "total_seconds": None,
        "peak_vram_mb": None,
        "num_params": None,
        "description": f"process exited with code {result.returncode}",
        "log_output": combined_output,
    }


def build_default_registry(root: Path | str) -> list[SlotManifestEntry]:
    """Build a deterministic baseline registry manifest from the current train.py."""

    train_path = Path(root) / "train.py"
    source_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
    return [
        SlotManifestEntry(
            slot_id="baseline_attention_v1",
            slot_type="attention",
            implementation_version="1",
            source_file="train.py",
            source_sha256=source_sha256,
        ),
        SlotManifestEntry(
            slot_id="baseline_mlp_v1",
            slot_type="mlp",
            implementation_version="1",
            source_file="train.py",
            source_sha256=source_sha256,
        ),
        SlotManifestEntry(
            slot_id="baseline_schedule_v1",
            slot_type="schedule",
            implementation_version="1",
            source_file="train.py",
            source_sha256=source_sha256,
        ),
        SlotManifestEntry(
            slot_id="baseline_init_v1",
            slot_type="init",
            implementation_version="1",
            source_file="train.py",
            source_sha256=source_sha256,
        ),
    ]


def load_registry_entries(root: Path | str, manifest_path: Path | str | None = None) -> list[SlotManifestEntry]:
    """Load the active registry manifest, or synthesize the baseline registry."""

    if manifest_path is None:
        manifest = Path(root) / DEFAULT_REGISTRY_MANIFEST
    else:
        manifest = Path(manifest_path)
        if not manifest.is_absolute():
            manifest = Path(root) / manifest

    if manifest.exists():
        return load_active_registry(manifest)
    return build_default_registry(root)


def resolve_repo_identity(root: Path | str) -> tuple[str, str]:
    """Return the current commit SHA and tree SHA for reproducibility."""

    repo_root = Path(root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, tree


def _count_non_default_slots(slots: dict[str, Any], baseline_slots: dict[str, Any]) -> int:
    return sum(1 for key, value in slots.items() if baseline_slots.get(key) != value)


def _count_custom_slots(slots: dict[str, Any]) -> int:
    return sum(1 for value in slots.values() if not str(value).startswith("baseline_"))


def run_generation(
    root: Path | str,
    *,
    generation: int,
    population: list[dict[str, Any]],
    registry_entries: list[SlotManifestEntry],
    repo_commit: str,
    repo_tree_hash: str,
    run_experiment: Callable[[Path, int], dict[str, Any]] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Execute one complete generation and persist ranked population state."""

    repo_root = Path(root)
    registry_hash = slot_registry_hash(registry_entries)
    registry_manifest = [asdict(entry) for entry in registry_entries]
    executor = run_experiment or (lambda experiment_path, limit: execute_experiment(experiment_path, limit, cwd=repo_root))

    executed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for genome in population:
        individual_id = genome["individual_id"]
        experiment_path = write_experiment_artifact(repo_root, individual_id, genome)
        write_registry_manifest_artifact(repo_root, individual_id, registry_manifest)
        outcome = executor(experiment_path, timeout_seconds)
        write_run_log_artifact(repo_root, individual_id, outcome.get("log_output", ""))
        write_summary_artifact(
            repo_root,
            individual_id,
            {key: value for key, value in outcome.items() if key != "log_output"},
        )
        executed.append((genome, outcome))

    baseline_slots = population[0]["slots"]
    baseline_num_params = next(
        (outcome.get("num_params") for _, outcome in executed if outcome.get("num_params") is not None),
        1,
    )

    run_records: list[dict[str, Any]] = []
    for genome, outcome in executed:
        record = {
            "individual_id": genome["individual_id"],
            "schema_version": genome.get("schema_version", SCHEMA_VERSION),
            "generation": generation,
            "parents": list(genome.get("parents", [])),
            "genome_hash": genome["genome_hash"],
            "repo_commit": repo_commit,
            "repo_tree_hash": repo_tree_hash,
            "slot_registry_hash": registry_hash,
            "val_bpb": outcome["val_bpb"],
            "is_valid": outcome["status"] == "success",
            "status": outcome["status"],
            "training_seconds": outcome["training_seconds"],
            "total_seconds": outcome["total_seconds"],
            "peak_vram_mb": outcome["peak_vram_mb"],
            "num_params": outcome["num_params"],
            "complexity_score": compute_complexity_score(
                non_default_slot_count=_count_non_default_slots(genome["slots"], baseline_slots),
                custom_slot_count=_count_custom_slots(genome["slots"]),
                num_params=outcome["num_params"] or baseline_num_params,
                baseline_num_params=baseline_num_params,
            ),
            "description": outcome.get("description", genome.get("metadata", {}).get("notes", "")),
            "genome": genome,
        }
        append_run_record(repo_root, record)
        run_records.append(record)

    ranked_population = rank_population(run_records)
    for fitness_rank, record in enumerate(ranked_population):
        record["fitness_rank"] = fitness_rank

    write_generation_state(
        repo_root,
        generation=generation,
        population=ranked_population,
        repo_commit=repo_commit,
        repo_tree_hash=repo_tree_hash,
        slot_registry_hash_value=registry_hash,
    )
    return {
        "generation": generation,
        "repo_commit": repo_commit,
        "repo_tree_hash": repo_tree_hash,
        "slot_registry_hash": registry_hash,
        "population": ranked_population,
    }
