"""Patch and validation helpers exposed by the MASGuard CLI."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class PatchApplyResult:
    patch_path: str
    repo_path: str
    applied: bool
    touched_file: str
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ValidationResult:
    command: str
    repo_path: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def apply_simple_unified_patch(repo_path: Path, patch_path: Path) -> PatchApplyResult:
    """Apply a small single-file unified patch without external dependencies.

    This is intentionally conservative and meant for demo/integration workflows.
    Real systems can use git/apply or their own workspace manager.
    """

    text = patch_path.read_text(encoding="utf-8")
    git_result = _try_external_patch(repo_path, patch_path)
    if git_result.applied:
        return git_result

    current_file: Path | None = None
    removals: list[str] = []
    additions: list[str] = []
    for raw in text.splitlines():
        if raw.startswith("+++ b/"):
            current_file = repo_path / raw.removeprefix("+++ b/")
        elif raw.startswith("--- ") or raw.startswith("diff ") or raw.startswith("@@"):
            continue
        elif raw.startswith("-"):
            removals.append(raw[1:])
        elif raw.startswith("+"):
            additions.append(raw[1:])

    if current_file is None:
        return PatchApplyResult(str(patch_path), str(repo_path), False, "", "patch target not found")
    if not current_file.exists():
        return PatchApplyResult(str(patch_path), str(repo_path), False, str(current_file), "target file missing")

    original = current_file.read_text(encoding="utf-8")
    old = "\n".join(removals)
    new = "\n".join(additions)
    if old not in original:
        return PatchApplyResult(str(patch_path), str(repo_path), False, str(current_file), "hunk did not match")

    current_file.write_text(original.replace(old, new, 1), encoding="utf-8")
    return PatchApplyResult(str(patch_path), str(repo_path), True, str(current_file))


def _try_external_patch(repo_path: Path, patch_path: Path) -> PatchApplyResult:
    """Use git/patch for normal unified diffs when available."""

    touched_file = _first_patch_target(repo_path, patch_path)
    git_stderr = ""
    if (repo_path / ".git").exists():
        git = subprocess.run(
            ["git", "apply", str(patch_path.resolve())],
            cwd=repo_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        git_stderr = git.stderr
        if git.returncode == 0 and _patch_target_changed(touched_file, patch_path):
            return PatchApplyResult(str(patch_path), str(repo_path), True, str(touched_file))

    patch = subprocess.run(
        ["patch", "--no-backup-if-mismatch", "-p1", "-i", str(patch_path.resolve())],
        cwd=repo_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if patch.returncode == 0:
        return PatchApplyResult(str(patch_path), str(repo_path), True, str(touched_file))

    reason = (git_stderr or patch.stderr or patch.stdout).strip().splitlines()
    return PatchApplyResult(
        str(patch_path),
        str(repo_path),
        False,
        str(touched_file),
        reason[0] if reason else "external patch application failed",
    )


def _first_patch_target(repo_path: Path, patch_path: Path) -> Path:
    for raw in patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("+++ b/"):
            return repo_path / raw.removeprefix("+++ b/")
    return repo_path


def _patch_target_changed(touched_file: Path, patch_path: Path) -> bool:
    if not touched_file.exists():
        return False
    text = patch_path.read_text(encoding="utf-8", errors="replace")
    added = [line[1:] for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if not added:
        return True
    current = touched_file.read_text(encoding="utf-8", errors="replace")
    return any(line and line in current for line in added)


def run_validation(repo_path: Path, command: str, timeout: int = 60) -> ValidationResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_path.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")
    argv = shlex.split(command)
    completed = subprocess.run(
        argv,
        cwd=repo_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return ValidationResult(
        command=command,
        repo_path=str(repo_path),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
