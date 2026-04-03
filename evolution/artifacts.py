"""Artifact and run-record persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ranking import validate_run_record
from .schema import canonical_genome_json

RUN_RECORD_FIELDS = [
    "individual_id",
    "schema_version",
    "generation",
    "parents",
    "genome_hash",
    "repo_commit",
    "repo_tree_hash",
    "slot_registry_hash",
    "val_bpb",
    "is_valid",
    "status",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "num_params",
    "complexity_score",
    "description",
]


def ensure_artifact_dir(root: Path | str, individual_id: str) -> Path:
    """Create and return the artifact directory for one individual."""

    artifact_dir = Path(root) / "artifacts" / individual_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(
        json.dumps(_json_ready(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_experiment_artifact(root: Path | str, individual_id: str, experiment: Any) -> Path:
    """Persist the rendered experiment config for one individual."""

    artifact_dir = ensure_artifact_dir(root, individual_id)
    return _write_json(artifact_dir / "experiment.json", experiment)


def write_registry_manifest_artifact(
    root: Path | str,
    individual_id: str,
    manifest: Sequence[Any],
) -> Path:
    """Persist the active registry manifest alongside an individual run."""

    artifact_dir = ensure_artifact_dir(root, individual_id)
    return _write_json(artifact_dir / "registry_manifest.json", list(manifest))


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _format_run_record_field(field_name: str, value: Any) -> str:
    if field_name == "parents":
        return canonical_genome_json(list(value))
    return _format_scalar(value)


def append_run_record(root: Path | str, record: Mapping[str, Any]) -> Path:
    """Append one run record to results/runs.tsv, creating the header if needed."""

    for field_name in RUN_RECORD_FIELDS:
        if field_name not in record:
            raise ValueError(f"missing run record field: {field_name}")

    validate_run_record(record)

    results_dir = Path(root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_path = results_dir / "runs.tsv"

    if not runs_path.exists():
        runs_path.write_text("\t".join(RUN_RECORD_FIELDS) + "\n", encoding="utf-8")

    row = "\t".join(
        _format_run_record_field(field_name, record[field_name])
        for field_name in RUN_RECORD_FIELDS
    )
    with runs_path.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")

    return runs_path
