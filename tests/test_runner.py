import json
from subprocess import CompletedProcess
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from evolve import assert_clean_worktree
from evolution.registry import SlotManifestEntry, slot_registry_hash
from evolution.runner import (
    create_initial_population,
    parse_args,
    parse_training_summary,
    run_generation,
    tournament_select,
    write_generation_state,
)


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

    def test_create_initial_population_produces_unique_individual_ids(self):
        population = create_initial_population(
            population_size=4,
            generation=0,
            rng_seed=7,
            max_seq_len=2048,
        )

        self.assertEqual(len(population), 4)
        self.assertEqual(len({item["individual_id"] for item in population}), 4)
        self.assertEqual(population[0]["generation"], 0)
        self.assertEqual(population[0]["parents"], [])

    def test_parse_training_summary_extracts_expected_fields(self):
        summary = parse_training_summary(
            "\n".join(
                [
                    "---",
                    "val_bpb:          1.234567",
                    "training_seconds: 300.1",
                    "total_seconds:    325.9",
                    "peak_vram_mb:     0.0",
                    "num_params_M:     50.3",
                    "depth:            8",
                ]
            )
        )

        self.assertEqual(summary["status"], "success")
        self.assertAlmostEqual(summary["val_bpb"], 1.234567)
        self.assertEqual(summary["peak_vram_mb"], None)
        self.assertEqual(summary["num_params"], 50300000)

    def test_run_generation_persists_ranked_results_and_artifacts(self):
        registry_entries = [
            SlotManifestEntry(
                slot_id="baseline_attention_v1",
                slot_type="attention",
                implementation_version="1",
                source_file="library/slots/attention.py",
                source_sha256="aaa",
            ),
            SlotManifestEntry(
                slot_id="baseline_mlp_v1",
                slot_type="mlp",
                implementation_version="1",
                source_file="library/slots/mlp.py",
                source_sha256="bbb",
            ),
            SlotManifestEntry(
                slot_id="baseline_schedule_v1",
                slot_type="schedule",
                implementation_version="1",
                source_file="library/slots/schedule.py",
                source_sha256="ccc",
            ),
            SlotManifestEntry(
                slot_id="baseline_init_v1",
                slot_type="init",
                implementation_version="1",
                source_file="library/slots/init.py",
                source_sha256="ddd",
            ),
        ]
        population = create_initial_population(
            population_size=2,
            generation=0,
            rng_seed=3,
            max_seq_len=2048,
        )

        def fake_executor(experiment_path: Path, timeout_seconds: int):
            payload = json.loads(experiment_path.read_text(encoding="utf-8"))
            score = 1.0 if payload["individual_id"] == population[0]["individual_id"] else 1.5
            return {
                "status": "success",
                "val_bpb": score,
                "training_seconds": 300.0,
                "total_seconds": 320.0,
                "peak_vram_mb": None,
                "num_params": 123456,
                "description": "fake executor",
            }

        with TemporaryDirectory() as tmpdir:
            state = run_generation(
                Path(tmpdir),
                generation=0,
                population=population,
                registry_entries=registry_entries,
                repo_commit="commit-a",
                repo_tree_hash="tree-a",
                run_experiment=fake_executor,
                timeout_seconds=600,
            )

            self.assertEqual(state["generation"], 0)
            self.assertEqual(state["repo_commit"], "commit-a")
            self.assertEqual(state["slot_registry_hash"], slot_registry_hash(registry_entries))
            self.assertEqual(state["population"][0]["val_bpb"], 1.0)
            self.assertEqual(state["population"][0]["status"], "success")

            current_path = Path(tmpdir) / "population" / "current_generation.json"
            archive_path = Path(tmpdir) / "population" / "archive" / "g0000.json"
            runs_path = Path(tmpdir) / "results" / "runs.tsv"
            artifact_dir = Path(tmpdir) / "artifacts" / population[0]["individual_id"]

            self.assertTrue(current_path.exists())
            self.assertTrue(archive_path.exists())
            self.assertTrue(runs_path.exists())
            self.assertTrue((artifact_dir / "experiment.json").exists())
            self.assertTrue((artifact_dir / "registry_manifest.json").exists())
            self.assertTrue((artifact_dir / "run.log").exists())
            self.assertTrue((artifact_dir / "summary.json").exists())

            lines = runs_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
