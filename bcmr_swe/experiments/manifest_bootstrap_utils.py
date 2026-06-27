"""Local manifest bootstrap helpers for BCMR.

These utilities were originally shared with a legacy PropGuard pipeline.
BCMR keeps a local copy so the runtime no longer depends on
legacy stacks such as ``propguard_swe`` or ``cadi_swe`` for manifest setup.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from bcmr_swe import PROJECT_ROOT


ROW_PAGE_SIZE = 100
DEFAULT_HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "datasets"
TEST_NAME_RE = re.compile(r"^([A-Za-z0-9_]+)\s+\(([A-Za-z0-9_\.]+)\)$")
PATCH_FILE_RE = re.compile(r"^\+\+\+\s+b/(.+)$", re.MULTILINE)
DEFAULT_LOCAL_GIT_EXCLUDES = (
    ".DS_Store",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    ".nox/",
    ".cache/",
    ".coverage",
    ".coverage.*",
    "htmlcov/",
    "*.log",
    "logs/",
    "log/",
    "tmp/",
    "temp/",
)


def normalize_test_entries(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return []


def normalize_target(entry: str) -> str | None:
    stripped = entry.strip()
    if not stripped:
        return None
    match = TEST_NAME_RE.match(stripped)
    if match:
        method_name, class_path = match.groups()
        return f"{class_path}.{method_name}"
    if "::" in stripped and " " not in stripped:
        return stripped
    if re.fullmatch(r"[A-Za-z0-9_\.]+", stripped):
        return stripped
    return None


def collect_test_targets(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    fail_to_pass = [
        target
        for item in normalize_test_entries(row.get("FAIL_TO_PASS"))
        if (target := normalize_target(item))
    ]
    pass_to_pass = [
        target
        for item in normalize_test_entries(row.get("PASS_TO_PASS"))
        if (target := normalize_target(item))
    ]
    return fail_to_pass, pass_to_pass


def extract_patch_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for match in PATCH_FILE_RE.findall(patch_text or ""):
        path = match.strip()
        if not path or path == "/dev/null" or path in files:
            continue
        files.append(path)
    return files


def load_cached_hf_dataset_rows(
    *,
    dataset: str,
    split: str,
    cache_root: Path = DEFAULT_HF_CACHE_ROOT,
) -> list[dict[str, Any]]:
    """Load dataset rows from the local HuggingFace Arrow cache if present."""

    cache_name = dataset.lower().replace("/", "___")
    candidates = sorted(cache_root.glob(f"{cache_name}/default/*/*/*-{split}.arrow"))
    if not candidates:
        return []
    try:
        import pyarrow.ipc as ipc  # type: ignore
    except Exception:
        return []
    with candidates[-1].open("rb") as handle:
        try:
            table = ipc.open_stream(handle).read_all()
        except (OSError, ValueError):
            handle.seek(0)
            table = ipc.open_file(handle).read_all()
    return [dict(item) for item in table.to_pylist() if isinstance(item, dict)]


def build_test_command(*, test_cmd: str, targets: list[str], activate_script: Path | None) -> str:
    command = test_cmd.strip()
    if activate_script is not None and command.startswith("./"):
        python_bin = activate_script.parent / "python"
        script_path, _, remainder = command.partition(" ")
        if python_bin.exists():
            command = f"PYTHONPATH=. {shlex.quote(str(python_bin))} {shlex.quote(script_path[2:])}"
            if remainder.strip():
                command = f"{command} {remainder.strip()}"
    quoted_targets = " ".join(shlex.quote(target) for target in targets)
    if quoted_targets:
        command = f"{command} {quoted_targets}".strip()
    if activate_script is None:
        return command
    return f"source {shlex.quote(str(activate_script))} && {command}"


def resolve_test_cmd(repo: str, version: str, spec_python: Path | None) -> str:
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS  # type: ignore

        return str(MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"])
    except Exception:
        pass

    if spec_python is None:
        raise ModuleNotFoundError("swebench")

    script = (
        "from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS\n"
        f"print(MAP_REPO_VERSION_TO_SPECS[{repo!r}][{version!r}]['test_cmd'])\n"
    )
    completed = subprocess.run(
        [str(spec_python), "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or "failed to resolve test_cmd")
    return completed.stdout.strip().splitlines()[-1]


def apply_patch_text(repo_root: Path, patch_text: str, *, use_git_apply: bool = True) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", delete=False) as handle:
        handle.write(patch_text)
        patch_path = Path(handle.name)
    try:
        command = (
            ["git", "apply", "--whitespace=nowarn", str(patch_path)]
            if use_git_apply
            else ["patch", "-p1", "-i", str(patch_path)]
        )
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode == 0:
            return True, completed.stdout
        return False, completed.stderr or completed.stdout
    finally:
        patch_path.unlink(missing_ok=True)


def materialize_source_snapshot(*, workdir: Path, test_patch: str) -> dict[str, Any]:
    if not test_patch.strip():
        return {"applied": False, "reason": "empty_test_patch"}
    ok, message = apply_patch_text(workdir, test_patch, use_git_apply=True)
    if not ok:
        return {"applied": False, "error": message}
    subprocess.run(["git", "config", "user.email", "setup@bcmr.local"], cwd=workdir, check=False)
    subprocess.run(["git", "config", "user.name", "BCMR Setup"], cwd=workdir, check=False)
    subprocess.run(["git", "add", "-A"], cwd=workdir, check=False)
    completed = subprocess.run(
        ["git", "commit", "-m", "SWE-bench test patch"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        return {"applied": False, "error": completed.stderr or completed.stdout or "git commit failed"}
    return {"applied": True}


def ensure_oracle_snapshot(
    source_snapshot: Path,
    oracle_snapshot: Path,
    patch_text: str,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    if oracle_snapshot.exists() and (refresh or _oracle_snapshot_needs_refresh(oracle_snapshot)):
        shutil.rmtree(oracle_snapshot, ignore_errors=True)
    if oracle_snapshot.exists():
        return {"created": False, "oracle_snapshot": str(oracle_snapshot)}

    clone = subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(source_snapshot), str(oracle_snapshot)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if clone.returncode != 0:
        return {
            "created": False,
            "oracle_snapshot": str(oracle_snapshot),
            "error": clone.stderr or clone.stdout or "git clone failed",
        }

    ok, message = apply_patch_text(oracle_snapshot, patch_text, use_git_apply=True)
    if not ok:
        shutil.rmtree(oracle_snapshot, ignore_errors=True)
        return {
            "created": False,
            "oracle_snapshot": str(oracle_snapshot),
            "error": message,
        }

    subprocess.run(["git", "config", "user.email", "setup@bcmr.local"], cwd=oracle_snapshot, check=False)
    subprocess.run(["git", "config", "user.name", "BCMR Setup"], cwd=oracle_snapshot, check=False)
    subprocess.run(["git", "add", "-A"], cwd=oracle_snapshot, check=False)
    completed = subprocess.run(
        ["git", "commit", "-m", "SWE-bench oracle patch"],
        cwd=oracle_snapshot,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        shutil.rmtree(oracle_snapshot, ignore_errors=True)
        return {
            "created": False,
            "oracle_snapshot": str(oracle_snapshot),
            "error": completed.stderr or completed.stdout or "git commit failed",
        }
    return {"created": True, "oracle_snapshot": str(oracle_snapshot), "committed": True}


def _oracle_snapshot_needs_refresh(snapshot: Path) -> bool:
    git_dir = snapshot / ".git"
    if not git_dir.exists() or not git_dir.is_dir():
        return True
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=snapshot,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        return True
    return bool((completed.stdout or "").strip())


def select_rows(
    *,
    rows: list[dict[str, Any]],
    instance_ids: list[str] | None,
    repo: str | None,
    max_instances: int | None,
) -> list[dict[str, Any]]:
    if instance_ids:
        ordered = []
        by_id = {str(row.get("instance_id")): row for row in rows}
        for instance_id in instance_ids:
            row = by_id.get(instance_id)
            if row is not None:
                ordered.append(row)
        return ordered

    filtered = [row for row in rows if not repo or str(row.get("repo") or "") == repo]
    filtered.sort(key=lambda row: str(row.get("instance_id") or ""))
    if max_instances is not None:
        filtered = filtered[:max_instances]
    return filtered


def build_repo_url(repo_slug: str) -> str:
    return f"https://github.com/{repo_slug}.git"


def _run_git(commands: list[str], cwd: Path, timeout: int = 600) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", *commands],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)
    if completed.returncode == 0:
        return True, completed.stdout
    return False, completed.stderr or completed.stdout


def _install_local_git_excludes(workdir: Path) -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "--git-path", "info/exclude"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return
    exclude_path = Path(completed.stdout.strip())
    if not exclude_path.is_absolute():
        exclude_path = workdir / exclude_path
    existing_lines = exclude_path.read_text(encoding="utf-8").splitlines() if exclude_path.exists() else []
    existing = {line.strip() for line in existing_lines}
    missing = [pattern for pattern in DEFAULT_LOCAL_GIT_EXCLUDES if pattern not in existing]
    if not missing:
        return
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    lines_to_append: list[str] = []
    if existing_lines and existing_lines[-1].strip():
        lines_to_append.append("")
    if "# Local runtime noise" not in existing:
        lines_to_append.append("# Local runtime noise")
    lines_to_append.extend(missing)
    with exclude_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines_to_append) + "\n")


def _ensure_cached_repo(
    repo_slug: str,
    snapshot_root: Path,
    *,
    skip_fetch: bool = False,
) -> tuple[Path | None, str | None]:
    snapshot_root = Path(snapshot_root).resolve()
    safe_repo = repo_slug.replace("/", "__")
    cache_dir = snapshot_root / "_repos" / safe_repo
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f"{safe_repo}.lock"
    acquired = False
    for _ in range(400):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            pass
    try:
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            repo_url = build_repo_url(repo_slug)
            completed = subprocess.run(
                ["git", "clone", repo_url, str(cache_dir)],
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            if completed.returncode != 0:
                return None, completed.stderr or completed.stdout
        elif not skip_fetch:
            _run_git(["fetch", "--all", "--tags", "--prune"], cwd=cache_dir)
    finally:
        if acquired and lock_path.exists():
            lock_path.unlink(missing_ok=True)
    return cache_dir, None


def prepare_instance_repo(
    *,
    instance_id: str,
    repo_slug: str,
    base_commit: str,
    snapshot_root: Path,
    skip_fetch: bool = False,
) -> tuple[Path | None, str | None]:
    snapshot_root = Path(snapshot_root).resolve()
    cache_dir, err = _ensure_cached_repo(repo_slug, snapshot_root, skip_fetch=skip_fetch)
    if cache_dir is None:
        return None, err
    workdir = snapshot_root / instance_id
    if not workdir.exists() or not any(workdir.iterdir()):
        ok, msg = _run_git(["worktree", "add", "--detach", str(workdir), base_commit], cwd=cache_dir)
        if not ok:
            return None, f"worktree add failed: {msg.strip()}"
    else:
        _run_git(["reset", "--hard"], cwd=workdir)
        _run_git(["clean", "-xfd"], cwd=workdir)
        ok, msg = _run_git(["checkout", "-f", base_commit], cwd=workdir)
        if not ok:
            return None, f"checkout failed: {msg.strip()}"
    _install_local_git_excludes(workdir)
    return workdir, None


__all__ = [
    "ROW_PAGE_SIZE",
    "DEFAULT_HF_CACHE_ROOT",
    "build_test_command",
    "collect_test_targets",
    "ensure_oracle_snapshot",
    "extract_patch_files",
    "load_cached_hf_dataset_rows",
    "materialize_source_snapshot",
    "prepare_instance_repo",
    "resolve_test_cmd",
    "select_rows",
]
