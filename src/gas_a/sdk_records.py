"""SDK 1.0.4 declaration limits; detailed evidence stays in uploaded state."""
import hashlib
import json

from .sensor_training.io import append_jsonl

TEXT_BYTES = 4000
CONFIG_BYTES = 65536


def configuration_bytes(value):
    # Match the SDK's default ensure_ascii=True, including escaped Chinese text.
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def bounded_configuration(value, state, round_index, phase):
    encoded = configuration_bytes(value)
    if len(encoded) <= CONFIG_BYTES:
        return value
    fingerprint = "sha256:" + hashlib.sha256(encoded).hexdigest()
    append_jsonl(state / "configuration-evidence.jsonl", {
        "round_index": round_index, "phase": phase, "sha256": fingerprint,
        "configuration": value,
    })
    # Only small scalar summaries go on the wire. Never truncate serialized JSON
    # or remove evidence: the complete object is in the SDK-uploaded state.
    summary = {}
    for key, item in value.items():
        if (len(summary) < 32 and isinstance(key, str) and len(key.encode("utf-8")) <= 80
                and (item is None or type(item) in (str, bool, int, float))
                and len(configuration_bytes(item)) <= 512):
            summary[key] = item
    result = {"summary": summary, "configuration_evidence": {
        "state_file": "configuration-evidence.jsonl", "round_index": round_index,
        "phase": phase, "sha256": fingerprint, "serialized_bytes": len(encoded),
    }}
    if len(configuration_bytes(result)) > CONFIG_BYTES:
        raise ValueError("Compact SDK configuration exceeds 65536 bytes")
    return result
