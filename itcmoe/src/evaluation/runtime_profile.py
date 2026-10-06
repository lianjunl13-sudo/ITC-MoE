"""itcmoe inference counters and runtime statistics."""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

_ENABLED = os.getenv("ITCMOE_PROFILE", "0") == "1"
_PROFILE_PATH_TEXT = os.getenv("ITCMOE_PROFILE_PATH", "").strip()
_DATASET = os.getenv("ITCMOE_PROFILE_DATASET", "unknown").strip()
_FLUSH_EVERY = max(
    1,
    int(os.getenv("ITCMOE_PROFILE_FLUSH_EVERY", "4096")),
)



_MOE_LAYERS = max(1, int(os.getenv("ITCMOE_PROFILE_MOE_LAYERS", "1")))

_PROFILE_PATH = (
    Path(_PROFILE_PATH_TEXT).expanduser()
    if _PROFILE_PATH_TEXT
    else None
)

_PID = os.getpid()
_HOSTNAME = 'anonymous'
_STARTED_AT = time.time()

_LOCK = threading.Lock()

_STATE = {
    "forward_count": 0,
    "accepted_token_count": 0,
    "activated_expert_count": 0,
}

_EVENT_COUNT = 0
_DIRTY = False


def enabled() -> bool:
    """Return whether profiling is enabled for this process."""
    return bool(_ENABLED and _PROFILE_PATH is not None)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _read_existing(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_snapshot_unlocked() -> None:
    global _DIRTY

    if not enabled() or not _DIRTY:
        return

    assert _PROFILE_PATH is not None

    _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(_PROFILE_PATH) + ".lock")

    process_snapshot = {
        "pid": _PID,
        "hostname": _HOSTNAME,
        "dataset": _DATASET,
        "started_at_unix": _STARTED_AT,
        "updated_at_unix": time.time(),
        **_STATE,
    }

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        existing = _read_existing(_PROFILE_PATH)
        processes = existing.get("processes", {})

        if not isinstance(processes, dict):
            processes = {}

        processes[str(_PID)] = process_snapshot

        totals = {
            "forward_count": 0,
            "accepted_token_count": 0,
            "activated_expert_count": 0,
        }

        for item in processes.values():
            if not isinstance(item, dict):
                continue

            for key in totals:
                totals[key] += int(item.get(key, 0))

        forward_count = totals["forward_count"]
        accepted = totals["accepted_token_count"]
        activated = totals["activated_expert_count"]
        
        activated_per_layer = activated / _MOE_LAYERS

        payload = {
            "dataset": _DATASET,
            "profile_path": str(_PROFILE_PATH),
            "moe_layers": _MOE_LAYERS,
            "processes": processes,
            "totals": totals,
            "metrics": {
                "APF": _safe_ratio(activated_per_layer, forward_count),
                "TPF": _safe_ratio(accepted, forward_count),
                "APT": _safe_ratio(activated_per_layer, accepted),
            },
            "formulas": {
                "APF": (
                    "activated_expert_count / forward_count / moe_layers"
                ),
                "TPF": (
                    "accepted_token_count / forward_count"
                ),
                "APT": (
                    "activated_expert_count / accepted_token_count / moe_layers"
                ),
            },
            "updated_at_unix": time.time(),
        }

        tmp_path = Path(
            str(_PROFILE_PATH) + f".tmp.{_PID}"
        )

        tmp_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        os.replace(tmp_path, _PROFILE_PATH)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    _DIRTY = False


def flush() -> None:
    """Write the current counters to the JSON profile file."""
    with _LOCK:
        _write_snapshot_unlocked()


def _add(key: str, value: int) -> None:
    global _EVENT_COUNT
    global _DIRTY

    if not enabled():
        return

    amount = int(value)

    if amount < 0:
        raise ValueError(
            f"profile counter increment must be non-negative: "
            f"{key}={amount}"
        )

    with _LOCK:
        _STATE[key] += amount
        _EVENT_COUNT += 1
        _DIRTY = True

        if _EVENT_COUNT % _FLUSH_EVERY == 0:
            _write_snapshot_unlocked()


def add_forward(value: int = 1) -> None:
    _add("forward_count", value)


def add_accepted_tokens(value: int) -> None:
    _add("accepted_token_count", value)


def add_activated_experts(value: int) -> None:
    _add("activated_expert_count", value)


atexit.register(flush)
