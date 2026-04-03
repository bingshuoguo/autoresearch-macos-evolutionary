"""CLI entrypoint for the genetic search runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

from evolution.runner import (
    breed_next_generation,
    create_initial_population,
    load_current_generation_state,
    load_registry_entries,
    parse_args,
    resolve_repo_identity,
    run_generation,
)


def assert_clean_worktree() -> None:
    """Refuse genetic-search runs on a dirty worktree by default."""

    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise SystemExit("Refusing to run evolve.py on a dirty worktree")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    assert_clean_worktree()

    repo_root = Path(__file__).resolve().parent
    registry_entries = load_registry_entries(repo_root, args.registry_manifest)
    repo_commit, repo_tree_hash = resolve_repo_identity(repo_root)
    previous_state = load_current_generation_state(repo_root)

    if previous_state is None:
        next_generation = 0
        population = create_initial_population(
            population_size=args.population_size,
            generation=next_generation,
            rng_seed=args.rng_seed,
            max_seq_len=args.max_seq_len,
        )
    else:
        next_generation = int(previous_state["generation"]) + 1
        population = breed_next_generation(
            previous_state["population"],
            generation=next_generation,
            population_size=args.population_size,
            elite_count=args.elite_count,
            tournament_size=args.tournament_size,
            rng_seed=args.rng_seed + next_generation,
            max_seq_len=args.max_seq_len,
            registry_entries=registry_entries,
        )

    state = None
    for generation in range(next_generation, next_generation + args.generation_limit):
        if generation != next_generation:
            population = breed_next_generation(
                state["population"],
                generation=generation,
                population_size=args.population_size,
                elite_count=args.elite_count,
                tournament_size=args.tournament_size,
                rng_seed=args.rng_seed + generation,
                max_seq_len=args.max_seq_len,
                registry_entries=registry_entries,
            )
        state = run_generation(
            repo_root,
            generation=generation,
            population=population,
            registry_entries=registry_entries,
            repo_commit=repo_commit,
            repo_tree_hash=repo_tree_hash,
            timeout_seconds=args.timeout_seconds,
        )

    winner = state["population"][0]
    print(
        " ".join(
            [
                f"generation={state['generation']}",
                f"winner={winner['individual_id']}",
                f"status={winner['status']}",
                f"val_bpb={winner['val_bpb']}",
            ]
        )
    )


if __name__ == "__main__":
    main()
