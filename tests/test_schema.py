import hashlib
import unittest

from evolution.schema import (
    SCHEMA_VERSION,
    ValidationError,
    build_individual_id,
    canonical_genome_json,
    normalize_genome,
)


class SchemaTests(unittest.TestCase):
    def test_normalize_genome_derives_model_fields_and_resolves_pattern(self):
        genome = {
            "schema_version": SCHEMA_VERSION,
            "generation": 3,
            "parents": ["g0002-i00-deadbeef"],
            "numeric": {
                "depth": 4,
                "aspect_ratio": 64,
                "total_batch_size": 65536,
                "device_batch_size": 16,
            },
            "categorical": {
                "head_dim": 16,
                "window_pattern": "sL",
            },
            "slots": {
                "attention": "baseline",
            },
        }

        normalized = normalize_genome(genome, max_seq_len=2048)

        self.assertEqual(normalized["categorical"]["window_pattern"], "SL")
        self.assertEqual(normalized["derived"]["model_dim"], 256)
        self.assertEqual(normalized["derived"]["num_heads"], 16)
        self.assertEqual(normalized["derived"]["resolved_window_pattern"], "SLSL")
        self.assertEqual(normalized["derived"]["tokens_per_fwdbwd"], 32768)

    def test_invalid_batch_multiple_raises_validation_error(self):
        genome = {
            "schema_version": SCHEMA_VERSION,
            "generation": 0,
            "parents": [],
            "numeric": {
                "depth": 4,
                "aspect_ratio": 64,
                "total_batch_size": 50000,
                "device_batch_size": 16,
            },
            "categorical": {
                "head_dim": 16,
                "window_pattern": "L",
            },
            "slots": {},
        }

        with self.assertRaises(ValidationError):
            normalize_genome(genome, max_seq_len=2048)

    def test_missing_required_nested_keys_raise_validation_error(self):
        base_genome = {
            "schema_version": SCHEMA_VERSION,
            "generation": 0,
            "parents": [],
            "numeric": {
                "depth": 4,
                "aspect_ratio": 64,
                "total_batch_size": 65536,
                "device_batch_size": 16,
            },
            "categorical": {
                "head_dim": 16,
                "window_pattern": "L",
            },
            "slots": {},
        }

        cases = [
            ("generation", lambda genome: genome.pop("generation")),
            ("numeric.depth", lambda genome: genome["numeric"].pop("depth")),
            ("categorical.head_dim", lambda genome: genome["categorical"].pop("head_dim")),
            ("categorical.window_pattern", lambda genome: genome["categorical"].pop("window_pattern")),
        ]

        for expected_field, mutator in cases:
            with self.subTest(expected_field=expected_field):
                genome = {
                    "schema_version": base_genome["schema_version"],
                    "generation": base_genome["generation"],
                    "parents": list(base_genome["parents"]),
                    "numeric": dict(base_genome["numeric"]),
                    "categorical": dict(base_genome["categorical"]),
                    "slots": dict(base_genome["slots"]),
                }
                mutator(genome)

                with self.assertRaises(ValidationError) as ctx:
                    normalize_genome(genome, max_seq_len=2048)

                self.assertIn(expected_field, str(ctx.exception))

    def test_boolean_numeric_fields_raise_validation_error(self):
        genome = {
            "schema_version": SCHEMA_VERSION,
            "generation": 0,
            "parents": [],
            "numeric": {
                "depth": True,
                "aspect_ratio": 64,
                "total_batch_size": 65536,
                "device_batch_size": 16,
            },
            "categorical": {
                "head_dim": 16,
                "window_pattern": "L",
            },
            "slots": {},
        }

        with self.assertRaises(ValidationError) as ctx:
            normalize_genome(genome, max_seq_len=2048)

        self.assertIn("depth", str(ctx.exception))

    def test_build_individual_id_uses_canonical_hash(self):
        payload = {
            "generation": 2,
            "schema_version": SCHEMA_VERSION,
            "parents": [],
            "slots": {"attention": "baseline"},
            "numeric": {
                "device_batch_size": 16,
                "depth": 4,
                "aspect_ratio": 64,
                "total_batch_size": 65536,
            },
            "categorical": {
                "window_pattern": "sL",
                "head_dim": 16,
            },
        }

        canonical = canonical_genome_json(payload)
        individual_id = build_individual_id(2, 1, canonical)
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

        self.assertEqual(individual_id, f"g0002-i01-{expected_hash}")


if __name__ == "__main__":
    unittest.main()
