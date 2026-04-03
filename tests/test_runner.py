import json
from subprocess import CompletedProcess
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from evolve import assert_clean_worktree
from evolution.runner import parse_args, tournament_select, write_generation_state


class RunnerTests(unittest.TestCase):
    def test_tournament_select_returns_lowest_ranked_sampled_member(self):
        population = [
            {"individual_id": "a", "fitness_rank": 0},
            {"individual_id": "b", "fitness_rank": 1},
            {"individual_id": "c", "fitness_rank": 2},
        ]

        selected = tournament_select(population, tournament_size=2, rng_seed=7)

        self.assertEqual(selected["individual_id"], "a")

    def test_parse_args_uses_expected_defaults(self):
        args = parse_args([])

        self.assertEqual(args.population_size, 6)
        self.assertEqual(args.elite_count, 2)
        self.assertEqual(args.tournament_size, 3)
        self.assertEqual(args.generation_limit, 1)

    def test_write_generation_state_updates_current_and_archive(self):
        population = [
            {"individual_id": "g0001-i01-abc12345", "fitness_rank": 0},
            {"individual_id": "g0001-i02-def67890", "fitness_rank": 1},
        ]

        with TemporaryDirectory() as tmpdir:
            current_path, archive_path = write_generation_state(
                Path(tmpdir),
                generation=1,
                population=population,
            )

            current_payload = json.loads(current_path.read_text(encoding="utf-8"))
            archive_payload = json.loads(archive_path.read_text(encoding="utf-8"))

            self.assertEqual(current_payload["generation"], 1)
            self.assertEqual(archive_payload["generation"], 1)
            self.assertEqual(current_payload["population"], population)
            self.assertEqual(archive_payload["population"], population)
            self.assertEqual(archive_path.name, "g0001.json")

    @patch("subprocess.run")
    def test_assert_clean_worktree_raises_on_dirty_output(self, mock_run):
        mock_run.return_value = CompletedProcess(
            args=["git", "status", "--short"],
            returncode=0,
            stdout=" M evolve.py\n",
            stderr="",
        )

        with self.assertRaises(SystemExit):
            assert_clean_worktree()


if __name__ == "__main__":
    unittest.main()
