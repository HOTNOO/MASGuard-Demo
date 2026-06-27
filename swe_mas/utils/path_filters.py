"""通用路径分类与补丁目标过滤工具。"""

from __future__ import annotations

import re
from pathlib import Path


_WORKSPACE_MARKERS = (
    "/testbed/",
    "/workspace/",
    "/repo/",
    "/source/",
)


def normalize_repo_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    normalized = normalized.strip("`'\" \t\r\n,;")
    if not normalized:
        return ""
    for marker in _WORKSPACE_MARKERS:
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]
            break
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("./")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def repo_path_variants(path: str) -> list[str]:
    """Return generic cleanup variants for LLM-extracted repo paths.

    Recovery prompts can inherit escaped-newline or transcript artifacts such
    as ``/nlib/foo.py`` or ``finish/nsrc/foo.py``.  We keep the transform
    conservative and workspace existence checks decide which variant survives.
    """

    normalized = normalize_repo_path(path)
    if not normalized:
        return []
    variants = [normalized]
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return variants

    common_roots = {"src", "lib", "pkg", "package", "packages", "tests", "test", "doc", "docs"}
    first = parts[0]
    if first.startswith("n") and first[1:] in common_roots:
        variants.append("/".join([first[1:]] + parts[1:]))
    if len(parts) >= 2 and first.lower() in {"finish", "final", "answer", "result"}:
        second = parts[1]
        if second.startswith("n") and second[1:] in common_roots:
            variants.append("/".join([second[1:]] + parts[2:]))
        variants.append("/".join(parts[1:]))
    return list(dict.fromkeys(variant for variant in variants if variant))


def existing_repo_source_paths(paths: list[str], workspace: str | Path | None) -> list[str]:
    """Return canonical source paths that actually exist in ``workspace``.

    If no workspace is available or none of the candidates exists, callers can
    fall back to ``ordered_canonical_source_paths``.  This avoids promoting
    placeholder paths from old transcripts into the recovery action state.
    """

    if workspace is None:
        return []
    root = Path(workspace)
    if not root.exists():
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        for variant in repo_path_variants(raw_path):
            if variant in seen:
                continue
            if is_generated_path(variant) or is_test_path(variant) or not looks_like_source_path(variant):
                continue
            if (root / variant).is_file():
                seen.add(variant)
                ordered.append(variant)
                break
    return ordered


def is_test_path(path: str) -> bool:
    parts = [part.lower() for part in Path(normalize_repo_path(path)).parts]
    name = parts[-1] if parts else ""
    return (
        any(part in {"test", "tests", "testing"} for part in parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def is_generated_path(path: str) -> bool:
    normalized = normalize_repo_path(path).lower()
    parts = [part for part in Path(normalized).parts if part]
    generated_markers = {
        "build",
        "dist",
        "__pycache__",
        ".eggs",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        "generated",
        "gen",
        "out",
        "target",
        "site-packages",
    }
    if any(part in generated_markers for part in parts):
        return True
    return any(
        token in normalized
        for token in (
            "/build/lib/",
            "/site-packages/",
            "/generated/",
            "/dist/",
            "/target/",
        )
    )


def looks_like_source_path(path: str) -> bool:
    normalized = normalize_repo_path(path)
    suffix = Path(normalized).suffix.lower()
    if suffix in {
        ".py",
        ".pyi",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".hh",
        ".java",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".rb",
        ".php",
    }:
        return True
    parts = [part.lower() for part in Path(normalized).parts]
    return any(part in {"src", "lib", "package", "packages"} for part in parts)


def canonical_source_paths(paths: list[str]) -> list[str]:
    preferred: list[str] = []
    fallback: list[str] = []
    for raw_path in paths:
        path = normalize_repo_path(raw_path)
        if not path or is_generated_path(path) or is_test_path(path):
            continue
        if looks_like_source_path(path):
            preferred.append(path)
        else:
            fallback.append(path)
    chosen = preferred or fallback
    return sorted(dict.fromkeys(chosen))


def ordered_canonical_source_paths(paths: list[str]) -> list[str]:
    preferred: list[str] = []
    fallback: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = normalize_repo_path(raw_path)
        if not path or path in seen or is_generated_path(path) or is_test_path(path):
            continue
        seen.add(path)
        if looks_like_source_path(path):
            preferred.append(path)
        else:
            fallback.append(path)
    return preferred or fallback


def classify_changed_files(changed_files: list[str]) -> dict[str, list[str]]:
    source_files: list[str] = []
    test_files: list[str] = []
    generated_files: list[str] = []
    other_files: list[str] = []

    for raw_path in changed_files:
        path = normalize_repo_path(raw_path)
        if not path:
            continue
        if is_test_path(path):
            test_files.append(path)
        elif is_generated_path(path):
            generated_files.append(path)
        elif looks_like_source_path(path):
            source_files.append(path)
        else:
            other_files.append(path)

    return {
        "source_files": sorted(dict.fromkeys(source_files)),
        "test_files": sorted(dict.fromkeys(test_files)),
        "generated_files": sorted(dict.fromkeys(generated_files)),
        "other_files": sorted(dict.fromkeys(other_files)),
        "effective_files": sorted(dict.fromkeys(source_files + test_files + other_files)),
    }


def parse_unified_diff_paths(diff_text: str) -> list[str]:
    paths: list[str] = []
    pattern = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
    for line in str(diff_text or "").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        right = normalize_repo_path(match.group(2))
        if right and right != "/dev/null":
            paths.append(right)
    return sorted(dict.fromkeys(paths))


def parse_git_status_paths(status_output: str) -> list[str]:
    paths: list[str] = []
    for raw_line in str(status_output or "").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        path_part = line[3:] if len(line) > 3 else line
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        normalized = normalize_repo_path(path_part)
        if normalized:
            paths.append(normalized)
    return sorted(dict.fromkeys(paths))
