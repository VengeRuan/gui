"""Versioned, embedded provenance for reloadable SD HDF5 files."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


PROVENANCE_ATTRIBUTE = "sd_provenance_json"
PROVENANCE_SCHEMA = "sd-h5-provenance"
PROVENANCE_VERSION = 1

_COMPACT_INTEGER_KEYS = {
    "channel_ids", "destination_channel_ids", "good_channel_ids",
    "filtered_channel_ids", "columns",
}


def compact_integer_ranges(values) -> str:
    """Encode integer collections as ``1-10,15,20-30``."""
    numbers = sorted({int(value) for value in values})
    if not numbers:
        return ""
    parts = []
    first = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        parts.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = number
    parts.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(parts)


def compact_provenance_operations(operations) -> list[dict]:
    """Return operations with verbose integer channel lists range-encoded."""
    compacted = _copy_json(operations)
    for operation in compacted if isinstance(compacted, list) else []:
        if not isinstance(operation, dict):
            continue
        for key in _COMPACT_INTEGER_KEYS.intersection(operation):
            value = operation[key]
            if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
                operation[key] = compact_integer_ranges(value)
    return compacted


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def _copy_json(value):
    return json.loads(json.dumps(_json_value(value), ensure_ascii=True, sort_keys=True))


def read_h5_provenance(h5) -> dict | None:
    """Read an embedded provenance document without making it mandatory."""
    raw = h5.attrs.get(PROVENANCE_ATTRIBUTE)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        document = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict) or document.get("schema") != PROVENANCE_SCHEMA:
        return None
    try:
        version = int(document.get("version", -1))
    except (TypeError, ValueError):
        return None
    if version != PROVENANCE_VERSION:
        return None
    return document


def build_h5_provenance(
    *,
    stage: str,
    fs: float,
    unit: str,
    channel_ids,
    source: dict | None = None,
    operation: dict | None = None,
) -> dict:
    """Create the first provenance document for an HDF5 data product."""
    document = {
        "schema": PROVENANCE_SCHEMA,
        "version": PROVENANCE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {
            "stage": str(stage),
            "sampling_rate_hz": float(fs),
            "unit": str(unit),
            "channel_ids": [int(value) for value in channel_ids],
        },
        "lineage": [],
        "operations": [],
    }
    if source:
        document["lineage"].append(_copy_json(source))
    if operation:
        document["operations"].extend(compact_provenance_operations([operation]))
    return document


def derive_h5_provenance(
    parent: dict | None,
    *,
    stage: str,
    fs: float,
    unit: str,
    channel_ids,
    source: dict | None = None,
    operation: dict | None = None,
) -> dict:
    """Append an operation while preserving the parent document's lineage."""
    if parent is None:
        return build_h5_provenance(
            stage=stage, fs=fs, unit=unit, channel_ids=channel_ids,
            source=source, operation=operation,
        )
    document = _copy_json(parent)
    try:
        parent_version = int(document.get("version", -1))
    except (TypeError, ValueError):
        parent_version = -1
    if document.get("schema") != PROVENANCE_SCHEMA or parent_version != PROVENANCE_VERSION:
        return build_h5_provenance(
            stage=stage, fs=fs, unit=unit, channel_ids=channel_ids,
            source=source, operation=operation,
        )
    document["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    document["data"] = {
        "stage": str(stage),
        "sampling_rate_hz": float(fs),
        "unit": str(unit),
        "channel_ids": [int(value) for value in channel_ids],
    }
    document.setdefault("lineage", [])
    document.setdefault("operations", [])
    if source:
        source_value = _copy_json(source)
        if not document["lineage"] or document["lineage"][-1] != source_value:
            document["lineage"].append(source_value)
    if operation:
        document["operations"].extend(compact_provenance_operations([operation]))
    document["operations"] = compact_provenance_operations(document["operations"])
    return document


def write_h5_provenance(h5, provenance: dict) -> None:
    """Embed canonical JSON in the HDF5 root attributes."""
    document = _copy_json(provenance)
    document["operations"] = compact_provenance_operations(document.get("operations", []))
    h5.attrs["provenance_schema"] = PROVENANCE_SCHEMA
    h5.attrs["provenance_version"] = PROVENANCE_VERSION
    h5.attrs[PROVENANCE_ATTRIBUTE] = json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
