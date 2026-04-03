import tempfile
import unittest
from pathlib import Path

from evolution.artifacts import (
    RUN_RECORD_FIELDS,
    append_run_record,
    ensure_artifact_dir,
    write_experiment_artifact,
    write_registry_manifest_artifact,
)
from evolution.ranking import compute_complexity_score, rank_population


class RankingTests(unittest.TestCase):
    def test_valid_runs_outrank_invalid_runs(self):
        population = [
            {
                "individual_id": "bad",
                "status": "crash",
                "is_valid": False,
                "val_bpb": None,
                "peak_vram_mb": None,
                "complexity_score": 0.0,
            },
            {
                "individual_id": "good",
                "status": "success",
                "is_valid": True,
                "val_bpb": 1.2,
                "peak_vram_mb": None,
                "complexity_score": 2.0,
            },
        ]

        ranked = rank_population(population)

        self.assertEqual(ranked[0]["individual_id"], "good")

    def test_null_resource_skips_resource_tiebreak(self):
        population = [
            {
                "individual_id": "a",
                "status": "success",
                "is_valid": True,
                "val_bpb": 1.0,
                "peak_vram_mb": None,
                "complexity_score": 1.0,
            },
            {
                "individual_id": "b",
                "status": "success",
                "is_valid": True,
                "val_bpb": 1.0,
                "peak_vram_mb": 500.0,
                "complexity_score": 2.0,
            },
        ]

        ranked = rank_population(population)

        self.assertEqual(ranked[0]["individual_id"], "a")

    def test_invalid_runs_have_deterministic_status_ordering(self):
        population = [
            {
                "individual_id": "crash-a",
                "status": "crash",
                "is_valid": False,
                "val_bpb": None,
                "peak_vram_mb": None,
                "complexity_score": 0.0,
            },
            {
                "individual_id": "invalid-output-a",
                "status": "invalid_output",
                "is_valid": False,
                "val_bpb": None,
                "peak_vram_mb": None,
                "complexity_score": 0.0,
            },
            {
                "individual_id": "timeout-a",
                "status": "timeout",
                "is_valid": False,
                "val_bpb": None,
                "peak_vram_mb": None,
                "complexity_score": 0.0,
            },
            {
                "individual_id": "unstable-a",
                "status": "unstable",
                "is_valid": False,
                "val_bpb": None,
                "peak_vram_mb": None,
                "complexity_score": 0.0,
            },
        ]

        ranked = rank_population(population)

        self.assertEqual(
            [item["individual_id"] for item in ranked],
            ["invalid-output-a", "unstable-a", "timeout-a", "crash-a"],
        )

    def test_invalid_run_rejects_placeholder_val_bpb(self):
        population = [
            {
                "individual_id": "bad",
                "status": "crash",
                "is_valid": False,
                "val_bpb": 0.0,
                "peak_vram_mb": None,
                "complexity_score": 0.0,
            }
        ]

        with self.assertRaises(ValueError):
            rank_population(population)

    def test_complexity_score_matches_spec_formula(self):
        score = compute_complexity_score(
            non_default_slot_count=2,
            custom_slot_count=1,
            num_params=150,
            baseline_num_params=100,
        )

        self.assertEqual(score, 6.0)


class ArtifactWriterTests(unittest.TestCase):
    def test_artifact_helpers_create_files_and_append_tsv_with_stable_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = ensure_artifact_dir(root, "g0001-i01-abc12345")
            self.assertEqual(artifact_dir, root / "artifacts" / "g0001-i01-abc12345")
            self.assertTrue(artifact_dir.is_dir())

            experiment_path = write_experiment_artifact(
                root,
                "g0001-i01-abc12345",
                {"individual_id": "g0001-i01-abc12345", "generation": 1},
            )
            registry_path = write_registry_manifest_artifact(
                root,
                "g0001-i01-abc12345",
                [{"slot_id": "baseline_attention_v1", "slot_type": "attention"}],
            )

            self.assertEqual(experiment_path.name, "experiment.json")
            self.assertEqual(registry_path.name, "registry_manifest.json")
            self.assertIn('"generation": 1', experiment_path.read_text(encoding="utf-8"))
            self.assertIn('"slot_id": "baseline_attention_v1"', registry_path.read_text(encoding="utf-8"))

            first_record = {
                "individual_id": "g0001-i01-abc12345",
                "schema_version": "1.0",
                "generation": 1,
                "parents": [],
                "genome_hash": "hash-a",
                "repo_commit": "commit-a",
                "repo_tree_hash": "tree-a",
                "slot_registry_hash": "registry-a",
                "val_bpb": 1.2345,
                "is_valid": True,
                "status": "success",
                "training_seconds": 300.0,
                "total_seconds": 320.0,
                "peak_vram_mb": None,
                "num_params": 1234567,
                "complexity_score": 1.0,
                "description": "winner",
            }
            second_record = {
                "individual_id": "g0001-i02-def67890",
                "schema_version": "1.0",
                "generation": 1,
                "parents": ["g0000-i01-00000000", "g0000-i02-11111111"],
                "genome_hash": "hash-b",
                "repo_commit": "commit-b",
                "repo_tree_hash": "tree-b",
                "slot_registry_hash": "registry-b",
                "val_bpb": None,
                "is_valid": False,
                "status": "crash",
                "training_seconds": None,
                "total_seconds": 12.0,
                "peak_vram_mb": None,
                "num_params": 7654321,
                "complexity_score": 3.0,
                "description": "crashed",
            }

            runs_path = append_run_record(root, first_record)
            append_run_record(root, second_record)

            lines = runs_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split("\t"), RUN_RECORD_FIELDS)
            self.assertEqual(len(lines), 3)
            self.assertIn("[]", lines[1])
            self.assertIn('["g0000-i01-00000000","g0000-i02-11111111"]', lines[2])
            self.assertIn("\tnull\t", lines[2])


if __name__ == "__main__":
    unittest.main()
