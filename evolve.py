"""CLI entrypoint for the initial evolutionary runner."""

from __future__ import annotations

import subprocess

from evolution.runner import parse_args


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
    print(
        " ".join(
            [
                f"population_size={args.population_size}",
                f"elite_count={args.elite_count}",
                f"tournament_size={args.tournament_size}",
                f"generation_limit={args.generation_limit}",
            ]
        )
    )


if __name__ == "__main__":
    main()
