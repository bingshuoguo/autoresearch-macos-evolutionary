"""Sequential evolutionary runner primitives."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

DEFAULT_POPULATION_SIZE = 6
DEFAULT_ELITE_COUNT = 2
DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_GENERATION_LIMIT = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the initial evolutionary runner."""

    parser = argparse.ArgumentParser(description="Autoresearch evolutionary runner")
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    parser.add_argument("--tournament-size", type=int, default=DEFAULT_TOURNAMENT_SIZE)
    parser.add_argument("--generation-limit", type=int, default=DEFAULT_GENERATION_LIMIT)
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
) -> tuple[Path, Path]:
    """Persist the current generation snapshot and an archived copy."""

    population_dir = Path(root) / "population"
    archive_dir = population_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "generation": generation,
        "population": population,
    }
    payload = json.dumps(snapshot, sort_keys=True, indent=2) + "\n"

    current_path = population_dir / "current_generation.json"
    current_path.write_text(payload, encoding="utf-8")

    archive_path = archive_dir / f"g{generation:04d}.json"
    archive_path.write_text(payload, encoding="utf-8")

    return current_path, archive_path
