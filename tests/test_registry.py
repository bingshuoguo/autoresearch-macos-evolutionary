import tempfile
import unittest
from pathlib import Path

from evolution.registry import load_active_registry, slot_registry_hash


class RegistryTests(unittest.TestCase):
    def test_registry_hash_is_deterministic_for_same_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "active.json"
            manifest.write_text(
                """
                [
                  {"slot_id":"baseline_mlp_v1","slot_type":"mlp","implementation_version":"1","source_file":"mlp.py","source_sha256":"bbb"},
                  {"slot_id":"baseline_attention_v1","slot_type":"attention","implementation_version":"1","source_file":"attention.py","source_sha256":"aaa"}
                ]
                """.strip(),
                encoding="utf-8",
            )

            entries = load_active_registry(manifest)
            digest1 = slot_registry_hash(entries)
            digest2 = slot_registry_hash(list(reversed(entries)))

            self.assertEqual(digest1, digest2)

    def test_duplicate_identity_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "active.json"
            manifest.write_text(
                """
                [
                  {"slot_id":"baseline_attention_v1","slot_type":"attention","implementation_version":"1","source_file":"attention.py","source_sha256":"aaa"},
                  {"slot_id":"baseline_attention_v1","slot_type":"attention","implementation_version":"2","source_file":"attention_v2.py","source_sha256":"bbb"}
                ]
                """.strip(),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_active_registry(manifest)

    def test_invalid_field_type_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "active.json"
            manifest.write_text(
                """
                [
                  {"slot_id":"baseline_attention_v1","slot_type":"attention","implementation_version":1,"source_file":"attention.py","source_sha256":"aaa"}
                ]
                """.strip(),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_active_registry(manifest)

    def test_missing_required_key_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "active.json"
            manifest.write_text('[{"slot_id":"only_id"}]', encoding="utf-8")

            with self.assertRaises(ValueError):
                load_active_registry(manifest)


if __name__ == "__main__":
    unittest.main()
