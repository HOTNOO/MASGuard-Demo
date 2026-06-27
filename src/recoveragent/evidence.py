"""Evidence extraction for failed LLM repair-agent trajectories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from recoveragent.models import EvidenceBundle, path_text


SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp"}
TEST_TARGET_RE = re.compile(r"([\w./-]+::[\w:\[\].-]+|[\w./-]*test[\w./-]*\.\w+)")
FILE_RE = re.compile(r"([\w./-]+\.(?:py|js|ts|java|go|rs|c|cpp|h|hpp))")


def extract_evidence(
    *,
    trajectory_path: Path,
    repo_path: Path,
    log_path: Path,
    patch_path: Path,
) -> EvidenceBundle:
    """Extract a compact evidence bundle from trajectory, repo, log, and patch files."""

    trajectory = _load_json(trajectory_path)
    log_text = _read_text(log_path)
    patch_text = _read_text(patch_path)
    tool_calls = _tool_calls(trajectory)
    patch_added, patch_removed = _patch_lines(patch_text)
    touched_files = _touched_files(patch_text)
    failure_lines = _failure_lines(log_text)
    stack_trace_files = _dedupe(FILE_RE.findall(log_text))
    test_targets = _dedupe(TEST_TARGET_RE.findall(log_text))
    repo_files = _repo_files(repo_path)

    signals = {
        "has_test_failure": bool(_contains_any(log_text, ("failed", "failure", "assertionerror", "traceback"))),
        "has_collection_or_target_error": bool(
            _contains_any(log_text, ("not found", "no tests ran", "collected 0", "invalid test"))
        ),
        "has_environment_error": bool(
            _contains_any(log_text, ("modulenotfounderror", "no module named", "timeout", "permission denied"))
        ),
        "has_patch": bool(touched_files or patch_added or patch_removed),
        "patch_touches_source": any(Path(item).suffix in SOURCE_EXTENSIONS and "test" not in item.lower() for item in touched_files),
        "patch_touches_tests": any("test" in item.lower() for item in touched_files),
        "stack_trace_not_touched": bool(set(stack_trace_files) - set(touched_files)),
        "repeated_tool_failure": _repeated_tool_failure(tool_calls),
        "patcher_no_effective_diff": _patcher_no_effective_diff(trajectory, tool_calls, log_text, patch_text),
    }

    return EvidenceBundle(
        trajectory_path=path_text(trajectory_path),
        repo_path=path_text(repo_path),
        log_path=path_text(log_path),
        patch_path=path_text(patch_path),
        issue=str(trajectory.get("issue", "") or ""),
        tool_calls=tool_calls,
        touched_files=touched_files,
        test_targets=test_targets,
        failure_lines=failure_lines,
        stack_trace_files=stack_trace_files,
        patch_added_lines=patch_added[:20],
        patch_removed_lines=patch_removed[:20],
        repo_files=repo_files[:200],
        signals=signals,
    )


def build_evidence_graph(bundle: EvidenceBundle) -> dict[str, Any]:
    """Build a small display graph from the extracted evidence bundle."""

    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []

    def add_node(node_id: str, label: str, kind: str, detail: str = "") -> None:
        if not any(node["id"] == node_id for node in nodes):
            nodes.append({"id": node_id, "label": label, "kind": kind, "detail": detail})

    def add_edge(src: str, dst: str, label: str) -> None:
        if src != dst:
            edges.append({"source": src, "target": dst, "label": label})

    add_node("trajectory", "failed trajectory", "input", bundle.issue[:140])
    add_node("log", "test/build log", "input", (bundle.failure_lines[0] if bundle.failure_lines else "")[:140])
    add_node("patch", "candidate patch", "input", ", ".join(bundle.touched_files[:4]))
    add_edge("trajectory", "log", "validated by")
    add_edge("trajectory", "patch", "produced")

    for index, call in enumerate(bundle.tool_calls[:8], start=1):
        node_id = f"tool_{index}"
        add_node(node_id, call.get("role") or f"tool {index}", "tool", call.get("command", ""))
        add_edge("trajectory", node_id, call.get("status") or "observed")

    for index, file_name in enumerate(bundle.stack_trace_files[:8], start=1):
        node_id = f"stack_{index}"
        add_node(node_id, file_name, "stack_trace", "file mentioned by failing log")
        add_edge("log", node_id, "mentions")

    for index, file_name in enumerate(bundle.touched_files[:8], start=1):
        node_id = f"patch_file_{index}"
        add_node(node_id, file_name, "patch_file", "file modified by candidate patch")
        add_edge("patch", node_id, "touches")

    stack_set = set(bundle.stack_trace_files)
    touched_set = set(bundle.touched_files)
    for file_name in sorted(stack_set & touched_set):
        stack_id = next((n["id"] for n in nodes if n["label"] == file_name and n["kind"] == "stack_trace"), "")
        patch_id = next((n["id"] for n in nodes if n["label"] == file_name and n["kind"] == "patch_file"), "")
        if stack_id and patch_id:
            add_edge(patch_id, stack_id, "matches failing evidence")

    for file_name in sorted(stack_set - touched_set)[:6]:
        stack_id = next((n["id"] for n in nodes if n["label"] == file_name and n["kind"] == "stack_trace"), "")
        if stack_id:
            add_edge(stack_id, "patch", "not touched by patch")

    return {"schema": "recoveragent.evidence_graph.v1", "nodes": nodes, "edges": edges}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"trajectory must be a JSON object: {path}")
    return data


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _tool_calls(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    raw = trajectory.get("tool_calls", []) or trajectory.get("commands", []) or []
    calls: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        calls.append(
            {
                "role": str(item.get("role", item.get("agent", "")) or ""),
                "command": str(item.get("command", item.get("tool", "")) or ""),
                "status": str(item.get("status", "") or ""),
                "summary": str(item.get("summary", "") or "")[:300],
            }
        )
    return calls


def _patch_lines(text: str) -> tuple[list[str], list[str]]:
    added: list[str] = []
    removed: list[str] = []
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return added, removed


def _touched_files(text: str) -> list[str]:
    files: list[str] = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files.append(parts[2].removeprefix("a/"))
                files.append(parts[3].removeprefix("b/"))
        elif line.startswith("+++ b/") or line.startswith("--- a/"):
            files.append(line[6:])
    return _dedupe(files)


def _failure_lines(text: str) -> list[str]:
    needles = ("error", "failed", "failure", "traceback", "assert", "not found", "timeout")
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(needle in line.lower() for needle in needles):
            out.append(line[:240])
        if len(out) >= 24:
            break
    return out


def _repo_files(repo_path: Path) -> list[str]:
    if not repo_path.exists():
        raise FileNotFoundError(repo_path)
    if repo_path.is_file():
        return [repo_path.name]
    files: list[str] = []
    for path in sorted(repo_path.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            files.append(str(path.relative_to(repo_path)))
    return files


def _repeated_tool_failure(calls: list[dict[str, Any]]) -> bool:
    failed_commands = [
        item["command"]
        for item in calls
        if item.get("status", "").lower() in {"failed", "error", "timeout"}
    ]
    return len(failed_commands) != len(set(failed_commands))


def _patcher_no_effective_diff(trajectory: dict[str, Any], calls: list[dict[str, Any]], log_text: str, patch_text: str) -> bool:
    patch_summary = trajectory.get("patch_summary")
    if isinstance(patch_summary, dict) and str(patch_summary.get("patch_scope", "") or "") == "no_diff":
        return True
    if "diff --git " in patch_text or "\n+++ " in patch_text:
        return False
    combined = "\n".join(
        [
            log_text,
            patch_text,
            json.dumps(trajectory, ensure_ascii=False),
            "\n".join(str(call.get("summary", "")) for call in calls),
        ]
    ).lower()
    return any(
        needle in combined
        for needle in (
            "no_effective_patch",
            "no effective patch",
            "masguard_strict_abstain_no_edit",
            "patch produced no git diff",
            "patch_scope\": \"no_diff",
        )
    )


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out
