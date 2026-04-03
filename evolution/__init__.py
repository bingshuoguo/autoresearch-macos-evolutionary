"""Evolutionary search primitives."""

from .schema import (
    SCHEMA_VERSION,
    ValidationError,
    build_individual_id,
    canonical_genome_json,
    normalize_genome,
)
from .registry import (
    REQUIRED_KEYS,
    SlotManifestEntry,
    load_active_registry,
    slot_registry_hash,
)
from .ranking import (
    INVALID_STATUS_ORDER,
    VALID_STATUS,
    compute_complexity_score,
    rank_population,
    validate_run_record,
)
from .runner import (
    DEFAULT_ELITE_COUNT,
    DEFAULT_GENERATION_LIMIT,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_TOURNAMENT_SIZE,
    parse_args,
    tournament_select,
    write_generation_state,
)
from .artifacts import (
    RUN_RECORD_FIELDS,
    append_run_record,
    ensure_artifact_dir,
    write_experiment_artifact,
    write_registry_manifest_artifact,
)

__all__ = [
    "SCHEMA_VERSION",
    "ValidationError",
    "build_individual_id",
    "canonical_genome_json",
    "normalize_genome",
    "REQUIRED_KEYS",
    "SlotManifestEntry",
    "load_active_registry",
    "slot_registry_hash",
    "INVALID_STATUS_ORDER",
    "VALID_STATUS",
    "compute_complexity_score",
    "rank_population",
    "validate_run_record",
    "DEFAULT_ELITE_COUNT",
    "DEFAULT_GENERATION_LIMIT",
    "DEFAULT_POPULATION_SIZE",
    "DEFAULT_TOURNAMENT_SIZE",
    "parse_args",
    "tournament_select",
    "write_generation_state",
    "RUN_RECORD_FIELDS",
    "append_run_record",
    "ensure_artifact_dir",
    "write_experiment_artifact",
    "write_registry_manifest_artifact",
]
