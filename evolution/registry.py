"""Slot registry identity and validation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_KEYS = {
    "slot_id",
    "slot_type",
    "implementation_version",
    "source_file",
    "source_sha256",
}


@dataclass(frozen=True)
class SlotManifestEntry:
    slot_id: str
    slot_type: str
    implementation_version: str
    source_file: str
    source_sha256: str


def load_active_registry(path: Path) -> list[SlotManifestEntry]:
    """Load and validate an active slot registry manifest from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("registry manifest must be a JSON array")

    entries: list[SlotManifestEntry] = []
    seen_identities: set[tuple[str, str]] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("registry manifest entries must be JSON objects")

        missing = REQUIRED_KEYS - set(raw)
        if missing:
            raise ValueError(f"missing registry keys: {sorted(missing)}")

        for field_name in REQUIRED_KEYS:
            if not isinstance(raw[field_name], str):
                raise ValueError(f"registry field {field_name} must be a string")

        identity = (raw["slot_type"], raw["slot_id"])
        if identity in seen_identities:
            raise ValueError(f"duplicate registry identity: {identity}")
        seen_identities.add(identity)

        entries.append(
            SlotManifestEntry(
                slot_id=raw["slot_id"],
                slot_type=raw["slot_type"],
                implementation_version=raw["implementation_version"],
                source_file=raw["source_file"],
                source_sha256=raw["source_sha256"],
            )
        )

    return entries


def slot_registry_hash(entries: list[SlotManifestEntry]) -> str:
    """Return a deterministic hash for the ordered active slot registry."""

    ordered_entries = sorted(entries, key=lambda entry: (entry.slot_type, entry.slot_id))
    canonical_manifest = json.dumps(
        [asdict(entry) for entry in ordered_entries],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_manifest.encode("utf-8")).hexdigest()
