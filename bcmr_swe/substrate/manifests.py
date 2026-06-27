"""Manifest helpers for BCMR experiment bootstrap and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


ManifestCatalog = dict[tuple[str, str], dict[str, Any]]


def manifest_score(payload: dict[str, Any], path: Path) -> float:
    score = 0.0
    if payload.get("environment_valid") is True:
        score += 30.0
    oracle_status = payload.get("oracle_snapshot_status") or {}
    if isinstance(oracle_status, dict) and not oracle_status.get("error"):
        score += 10.0
    oracle_preflight = payload.get("oracle_patch_preflight") or {}
    if isinstance(oracle_preflight, dict) and oracle_preflight.get("oracle_success"):
        score += 20.0
    if payload.get("dataset_name"):
        score += 5.0
    score += path.stat().st_mtime / 1_000_000_000.0
    return score


def load_manifest_catalog(artifact_root: str | Path, *, pattern: str = "real_eval*.json") -> ManifestCatalog:
    artifact_root = Path(artifact_root)
    catalog: ManifestCatalog = {}
    for path in sorted(artifact_root.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _is_manifest_payload(payload):
            continue
        fault = str(payload.get("fault_injection") or "none")
        key = (str(payload["instance_id"]), fault)
        score = manifest_score(payload, path)
        current = catalog.get(key)
        if current is None or score > current["score"]:
            catalog[key] = {"path": path, "payload": payload, "score": score}
    return catalog


def select_manifest_entry(
    catalog: ManifestCatalog,
    *,
    instance_id: str,
    fault: str = "none",
) -> dict[str, Any] | None:
    preferred: list[tuple[str, str]] = []
    if fault != "none":
        preferred.append((instance_id, fault))
    preferred.append((instance_id, "none"))
    for key in preferred:
        entry = catalog.get(key)
        if entry is not None:
            return entry
    candidates = [entry for (candidate_id, _), entry in catalog.items() if candidate_id == instance_id]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("score", 0.0)))


def run_manifest_command(snapshot: str | Path, command: str, *, timeout: int = 1800) -> dict[str, Any]:
    snapshot = Path(snapshot)
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(snapshot),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(snapshot),
            "returncode": 124,
            "output": (exc.stdout or "") + (exc.stderr or "") + f"\nTimed out after {timeout} seconds.",
        }
    return {
        "command": command,
        "cwd": str(snapshot),
        "returncode": int(completed.returncode),
        "output": (completed.stdout or "") + (completed.stderr or ""),
    }


def resolve_oracle_snapshot(manifest: dict[str, Any]) -> Path:
    explicit = manifest.get("oracle_snapshot")
    if explicit:
        return Path(str(explicit))
    return Path(str(manifest["source_snapshot"]) + "__oracle")


def run_oracle_preflight(manifest: dict[str, Any], *, timeout: int = 1800) -> dict[str, Any]:
    oracle_snapshot = resolve_oracle_snapshot(manifest)
    if not oracle_snapshot.exists():
        return {"available": False, "oracle_snapshot": str(oracle_snapshot)}
    fail_to_pass = run_manifest_command(oracle_snapshot, str(manifest["test_command"]), timeout=timeout)
    oracle = run_manifest_command(oracle_snapshot, str(manifest["oracle_command"]), timeout=timeout)
    return {
        "available": True,
        "oracle_snapshot": str(oracle_snapshot),
        "instance_id": str(manifest.get("instance_id", "")),
        "fail_to_pass_success": fail_to_pass["returncode"] == 0,
        "fail_to_pass_returncode": fail_to_pass["returncode"],
        "oracle_success": oracle["returncode"] == 0,
        "oracle_returncode": oracle["returncode"],
        "fail_to_pass_output": fail_to_pass["output"],
        "oracle_output": oracle["output"],
    }


def _is_manifest_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"instance_id", "problem_statement", "source_snapshot", "test_command", "oracle_command", "oracle_patch_files"}
    return required.issubset(payload.keys())
