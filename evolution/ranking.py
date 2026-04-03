"""Ranking helpers for evolutionary runs."""

from __future__ import annotations

from functools import cmp_to_key
from typing import Any, Mapping

VALID_STATUS = "success"
INVALID_STATUS_ORDER = {
    "invalid_output": 0,
    "unstable": 1,
    "timeout": 2,
    "crash": 3,
}
ALL_STATUSES = {VALID_STATUS, *INVALID_STATUS_ORDER}


def compute_complexity_score(
    *,
    non_default_slot_count: int,
    custom_slot_count: int,
    num_params: int | float,
    baseline_num_params: int | float,
) -> float:
    """Return the spec-defined deterministic complexity score."""

    if baseline_num_params <= 0:
        raise ValueError("baseline_num_params must be > 0")

    score = (
        1.0 * non_default_slot_count
        + 2.0 * custom_slot_count
        + 4.0 * max(0.0, (float(num_params) / float(baseline_num_params)) - 1.0)
    )
    return round(score, 3)


def validate_run_record(record: Mapping[str, Any]) -> None:
    """Validate the ranking-critical fields on a run record."""

    status = record["status"]
    if status not in ALL_STATUSES:
        raise ValueError(f"unknown run status: {status}")

    expected_validity = status == VALID_STATUS
    if record["is_valid"] != expected_validity:
        raise ValueError("is_valid must match status == 'success'")

    if expected_validity:
        if record["val_bpb"] is None:
            raise ValueError("success records must include val_bpb")
        float(record["val_bpb"])
    elif record["val_bpb"] is not None:
        raise ValueError("non-success records must store val_bpb as null")

    float(record["complexity_score"])

    if record.get("peak_vram_mb") is not None:
        float(record["peak_vram_mb"])


def _compare_ordered(left: Any, right: Any) -> int:
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def _compare_valid_records(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    by_bpb = _compare_ordered(float(left["val_bpb"]), float(right["val_bpb"]))
    if by_bpb:
        return by_bpb

    left_resource = left.get("peak_vram_mb")
    right_resource = right.get("peak_vram_mb")
    if left_resource is not None and right_resource is not None:
        by_resource = _compare_ordered(float(left_resource), float(right_resource))
        if by_resource:
            return by_resource

    by_complexity = _compare_ordered(
        float(left["complexity_score"]),
        float(right["complexity_score"]),
    )
    if by_complexity:
        return by_complexity

    return _compare_ordered(str(left["individual_id"]), str(right["individual_id"]))


def _invalid_sort_key(record: Mapping[str, Any]) -> tuple[int, str]:
    return (INVALID_STATUS_ORDER[record["status"]], str(record["individual_id"]))


def rank_population(population: list[dict]) -> list[dict]:
    """Return a fully ranked population with valid runs ahead of invalid runs."""

    validated = list(population)
    for record in validated:
        validate_run_record(record)

    valid = sorted(
        (record for record in validated if record["is_valid"]),
        key=cmp_to_key(_compare_valid_records),
    )
    invalid = sorted(
        (record for record in validated if not record["is_valid"]),
        key=_invalid_sort_key,
    )
    return valid + invalid
