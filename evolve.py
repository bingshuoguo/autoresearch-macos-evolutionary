"""CLI entrypoint for the initial evolutionary runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

from evolution.runner import (
    create_initial_population,
    load_registry_entries,
    parse_args,
    resolve_repo_identity,
    run_generation,
)


def assert_clean_worktree() -> None:
    """Refuse evolutionary runs on a dirty worktree by default."""

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
    if args.generation_limit != 1:
        raise SystemExit("generation-limit > 1 is not implemented yet")

    repo_root = Path(__file__).resolve().parent
    registry_entries = load_registry_entries(repo_root, args.registry_manifest)
    repo_commit, repo_tree_hash = resolve_repo_identity(repo_root)
    population = create_initial_population(
        population_size=args.population_size,
        generation=0,
        rng_seed=args.rng_seed,
        max_seq_len=args.max_seq_len,
    )
    state = run_generation(
        repo_root,
        generation=0,
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
                "generation=0",
                f"winner={winner['individual_id']}",
                f"status={winner['status']}",
                f"val_bpb={winner['val_bpb']}",
            ]
        )
    )


if __name__ == "__main__":
    main()
