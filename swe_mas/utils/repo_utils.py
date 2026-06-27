"""按实例准备源码仓库的工具

- 根据 SWE-bench 实例信息自动构建仓库 URL
- 克隆到指定工作根目录的唯一子目录
- 可选按 base_commit 强制 checkout，避免不同实例互相影响
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Tuple
import time

from swe_mas.utils.logger import get_logger
from swe_mas.utils.git_utils import clone_repo_ssh, convert_https_to_ssh


logger = get_logger(__name__)

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


def _extract_repo_slug(instance: dict) -> str | None:
    """从实例中提取仓库标识，例如 "owner/repo"。

    兼容多种字段名以适配不同数据版本。
    """
    for key in ("repo", "repository", "repo_name", "repo_slug"):
        value = instance.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_base_commit(instance: dict) -> str | None:
    """从实例中提取基础提交哈希，兼容常见字段名。"""
    for key in ("base_commit", "base_commit_sha", "base_sha", "base_commit_id"):
        value = instance.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def build_repo_url(repo_slug: str, use_ssh: bool = False) -> str:
    """构造仓库 URL。

    默认使用 HTTPS；当 use_ssh=True 时转换为 SSH 形式。
    """
    https_url = f"https://github.com/{repo_slug}.git"
    return convert_https_to_ssh(https_url) if use_ssh else https_url


def build_instance_workdir(work_root: str, repo_slug: str, instance_id: str) -> Path:
    """构造实例级工作目录，避免并发实例相互干扰。"""
    safe_repo = repo_slug.replace("/", "__")
    return Path(work_root) / f"{safe_repo}__{instance_id}"


def _run_git(commands: list[str], cwd: Path, timeout: int = 300) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", *commands],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr or result.stdout
    except Exception as e:
        return False, str(e)


def _install_local_git_excludes(workdir: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "info/exclude"],
            cwd=str(workdir),
            text=True,
            capture_output=True,
            timeout=30,
        )
    except Exception:
        return

    if result.returncode != 0:
        return

    exclude_path = Path(result.stdout.strip())
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


def _ensure_cached_repo(repo_slug: str, work_root: str, use_ssh: bool) -> Tuple[Path | None, str | None]:
    """确保本地存在仓库缓存副本（单拷贝），返回缓存路径。

    为加速与节省空间：/work_root/_repos/{owner__repo}
    """
    safe_repo = repo_slug.replace("/", "__")
    cache_dir = Path(work_root) / "_repos" / safe_repo
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    # 文件锁，避免并发克隆同一仓库
    lock_path = cache_dir.parent / (cache_dir.name + ".lock")
    acquired = False
    start = time.time()
    while not acquired and time.time() - start < 300:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
        except FileExistsError:
            time.sleep(0.25)

    try:
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            repo_url = build_repo_url(repo_slug, use_ssh=use_ssh)
            logger.info(f"克隆缓存仓库 {repo_slug} -> {cache_dir}")
            ok = clone_repo_ssh(repo_url, str(cache_dir), use_ssh=use_ssh)
            if not ok:
                # 若出现并发竞争：目录可能已被其他worker克隆好
                if cache_dir.exists() and any(cache_dir.iterdir()):
                    logger.info("检测到缓存目录已存在且非空，视为可用")
                else:
                    return None, f"缓存克隆失败: {repo_slug}"
        else:
            _run_git(["fetch", "--all", "--tags", "--prune"], cwd=cache_dir)
    finally:
        try:
            if acquired and lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass

    return cache_dir, None


def prepare_instance_repo(
    instance: dict,
    work_root: str,
    *,
    use_ssh: bool = False,
    checkout: bool = True,
    clone_timeout_sec: int = 600,
) -> tuple[str | None, bool, str | None]:
    """为单个实例准备独立的源码目录。

    Returns: (workdir, success, error)
    """
    instance_id = instance.get("instance_id", "")
    repo_slug = _extract_repo_slug(instance)
    if not repo_slug:
        return None, False, "实例缺少仓库信息(repo)"

    base_commit = _extract_base_commit(instance)
    work_root_path = Path(work_root)
    work_root_path.mkdir(parents=True, exist_ok=True)

    # 1) 确保缓存仓库存在（单拷贝）
    cache_dir, err = _ensure_cached_repo(repo_slug, work_root, use_ssh)
    if cache_dir is None:
        return None, False, err

    # 2) 为实例创建独立工作树（避免并发干扰）
    workdir = build_instance_workdir(work_root, repo_slug, instance_id)
    workdir.parent.mkdir(parents=True, exist_ok=True)

    if not workdir.exists() or not any(workdir.iterdir()):
        target = base_commit if (checkout and base_commit) else "HEAD"
        ok, msg = _run_git(["worktree", "add", "--detach", str(workdir), target], cwd=cache_dir)
        if not ok:
            return None, False, f"worktree创建失败: {msg.strip()}"
    else:
        # 已存在则重置到目标提交
        _run_git(["reset", "--hard"], cwd=workdir)
        _run_git(["clean", "-xfd"], cwd=workdir)
        if checkout and base_commit:
            ok, msg = _run_git(["checkout", "-f", base_commit], cwd=workdir)
            if not ok:
                return None, False, f"checkout失败: {base_commit} - {msg.strip()}"

    _install_local_git_excludes(workdir)
    return str(workdir), True, None

