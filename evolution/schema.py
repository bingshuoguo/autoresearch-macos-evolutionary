"""Genome schema and normalization helpers for evolutionary search."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any

SCHEMA_VERSION = "1.0"
ALLOWED_WINDOW_CHARS = {"S", "L"}


class ValidationError(ValueError):
    """Raised when a genome cannot be normalized or validated."""


def canonical_genome_json(payload: dict) -> str:
    """Return a stable canonical JSON encoding for hashing and deduping."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_individual_id(generation: int, ordinal: int, canonical_json: str) -> str:
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:8]
    return f"g{generation:04d}-i{ordinal:02d}-{digest}"


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be an integer") from exc


def _require_field(mapping: dict, key: str, field_name: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValidationError(f"missing required genome field: {field_name}") from exc


def _repeat_pattern(pattern: str, depth: int) -> str:
    repeated = (pattern * depth)[:depth]
    if depth == 1:
        return "L"
    return repeated[:-1] + "L"


def normalize_genome(genome: dict, *, max_seq_len: int) -> dict:
    """Normalize a genome into a canonical executable phenotype."""

    try:
        numeric = genome["numeric"]
        categorical = genome["categorical"]
        slots = genome["slots"]
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValidationError(f"missing required genome field: {missing_key}") from exc

    generation = _require_field(genome, "generation", "generation")
    depth = _require_field(numeric, "depth", "numeric.depth")
    aspect_ratio = _require_field(numeric, "aspect_ratio", "numeric.aspect_ratio")
    total_batch_size = _require_field(numeric, "total_batch_size", "numeric.total_batch_size")
    device_batch_size = _require_field(numeric, "device_batch_size", "numeric.device_batch_size")
    head_dim = _require_field(categorical, "head_dim", "categorical.head_dim")
    raw_window_pattern = _require_field(categorical, "window_pattern", "categorical.window_pattern")

    depth = _coerce_int(depth, "depth")
    aspect_ratio = _coerce_int(aspect_ratio, "aspect_ratio")
    total_batch_size = _coerce_int(total_batch_size, "total_batch_size")
    device_batch_size = _coerce_int(device_batch_size, "device_batch_size")
    head_dim = _coerce_int(head_dim, "head_dim")

    raw_window_pattern = str(raw_window_pattern).upper()
    if not raw_window_pattern:
        raise ValidationError("window_pattern must be non-empty")
    if any(ch not in ALLOWED_WINDOW_CHARS for ch in raw_window_pattern):
        raise ValidationError("window_pattern must contain only S/L")

    if depth < 1:
        raise ValidationError("depth must be >= 1")
    if head_dim < 1:
        raise ValidationError("head_dim must be >= 1")
    if device_batch_size < 1:
        raise ValidationError("device_batch_size must be >= 1")

    tokens_per_fwdbwd = device_batch_size * max_seq_len
    if total_batch_size < tokens_per_fwdbwd:
        raise ValidationError("total_batch_size must be >= device_batch_size * max_seq_len")
    if total_batch_size % tokens_per_fwdbwd != 0:
        raise ValidationError("total_batch_size must be a multiple of device_batch_size * max_seq_len")

    base_dim = depth * aspect_ratio
    model_dim = math.ceil(base_dim / head_dim) * head_dim
    num_heads = model_dim // head_dim
    resolved_window_pattern = _repeat_pattern(raw_window_pattern, depth)

    normalized = {
        "schema_version": genome.get("schema_version", SCHEMA_VERSION),
        "generation": _coerce_int(generation, "generation"),
        "parents": list(genome.get("parents", [])),
        "numeric": dict(numeric),
        "categorical": dict(categorical, window_pattern=raw_window_pattern),
        "slots": deepcopy(slots),
        "derived": {
            "model_dim": model_dim,
            "num_heads": num_heads,
            "resolved_window_pattern": resolved_window_pattern,
            "tokens_per_fwdbwd": tokens_per_fwdbwd,
        },
    }
    return normalized
