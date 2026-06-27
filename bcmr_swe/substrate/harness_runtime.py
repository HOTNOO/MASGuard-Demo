"""Persistent official SWE-bench env-image runtime for BCMR agents."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid
from typing import Any, Iterator


CONTAINER_WORKDIR = "/testbed"
MAX_INLINE_DOCKER_EXEC_CHARS = 32_000


def _slug_for_lock(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:160] or "unknown"


def _offline_editable_install_command(command: str) -> str:
    """Avoid networked build isolation for editable repo setup installs."""
    editable = _editable_install_parts(command)
    if editable is None:
        return command
    pip_prefix, rest, _editable_target = editable
    rewritten = [*pip_prefix, "install", "--no-build-isolation", *rest[1:]]
    return shlex.join(rewritten)


def _editable_install_parts(command: str) -> tuple[list[str], list[str], str] | None:
    normalized = command.strip()
    try:
        parts = shlex.split(normalized)
    except ValueError:
        return None
    pip_prefix = []
    if parts[:3] == ["python", "-m", "pip"]:
        pip_prefix = parts[:3]
        rest = parts[3:]
    elif parts[:1] == ["pip"]:
        pip_prefix = parts[:1]
        rest = parts[1:]
    else:
        return None
    if not rest or rest[0] != "install" or "--no-build-isolation" in rest:
        return None
    if "-e" not in rest:
        return None
    editable_index = rest.index("-e")
    if editable_index + 1 >= len(rest):
        return None
    editable_target = rest[editable_index + 1]
    if editable_target not in {".", ".[test]"}:
        return None
    return pip_prefix, rest, editable_target


def _editable_install_bootstrap_commands(command: str, *, workspace: Path, instance: dict[str, Any] | None = None) -> list[str]:
    """Restore build-isolation dependencies when we deliberately disable isolation.

    Astropy's SWE-bench setup installs ``.[test]`` from a source snapshot. The
    offline rewrite avoids networked build isolation, so the build requirements
    normally installed in isolation must be present in the active env first.
    Keep this narrow to repositories that explicitly declare extension-helpers.
    """
    if _editable_install_parts(command) is None:
        return []
    pyproject = workspace / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8").lower()
    except OSError:
        return []
    if "extension-helpers" not in text or "setuptools_scm" not in text:
        return []
    commands = ["python -m pip install extension-helpers setuptools_scm cython==0.29.22"]
    repo = str((instance or {}).get("repo", "") or "").lower()
    if repo == "astropy/astropy" or (workspace / "astropy").is_dir():
        version = str((instance or {}).get("version", "") or "4.3").strip() or "4.3"
        if ".dev" not in version:
            version = f"{version}.dev0"
        commands.append(f"export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTROPY={shlex.quote(version)}")
    return commands


@dataclass
class HarnessRuntimeConfig:
    dataset_name: str
    split: str = "test"
    dataset_instance: dict[str, Any] | None = None
    namespace: str | None = None
    force_rebuild: bool = False
    max_workers: int = 1
    instance_image_tag: str = "latest"
    env_image_tag: str = "latest"
    run_id: str = "bcmr-runtime"
    container_prefix: str = "bcmr"
    timeout: int = 120
    setup_timeout: int = 1800
    repo_init_lock_timeout: int = 60
    repo_init_lock_root: str = "/tmp/bcmr_harness_repo_init_locks"
    container_start_timeout: int = 180
    container_cleanup_timeout: int = 60
    cwd: str = ""
    encoding: str = "utf-8"
    extra_env: dict[str, str] = field(default_factory=dict)
    docker_binary: str = "docker"
    env_image_key_override: str = ""
    skip_recursive_chmod: bool = False


@dataclass
class _OverrideEnvSpec:
    env_image_key: str
    repo_script_list: list[str]


class OfficialHarnessRuntime:
    """Run BCMR commands in an official SWE-bench env image with a mounted workspace.

    The real BCMR pipeline already materializes source snapshots locally. This runtime
    keeps those snapshot contents intact, reuses the official env image, and applies the
    repo install steps without rebuilding a fragile instance image that clones inside Docker.
    """

    def __init__(self, *, instance_id: str, workspace: str | Path, config: HarnessRuntimeConfig):
        self.instance_id = instance_id
        self.workspace = Path(workspace).resolve()
        self.config = config
        self.config.cwd = str(self.workspace)
        self._container_name = f"{config.container_prefix}.{instance_id.lower()}.{uuid.uuid4().hex[:8]}"
        self._instance = None
        self._spec = None
        self._started = False
        self._initialized = False

    @property
    def container_name(self) -> str:
        return self._container_name

    def image_metadata(self) -> dict[str, Any]:
        if self._spec is None:
            return {}
        image_key = str(getattr(self._spec, "env_image_key", "") or "")
        digest = ""
        image_id = ""
        if image_key:
            completed = subprocess.run(
                [
                    self.config.docker_binary,
                    "image",
                    "inspect",
                    image_key,
                ],
                capture_output=True,
                text=True,
                check=False,
                env=os.environ,
            )
            if completed.returncode == 0:
                try:
                    inspected = json.loads(completed.stdout or "[]")
                except json.JSONDecodeError:
                    inspected = []
                if inspected:
                    first = inspected[0]
                    if isinstance(first, dict):
                        image_id = str(first.get("Id", "") or "")
                        repo_digests = first.get("RepoDigests", []) or []
                        if isinstance(repo_digests, list) and repo_digests:
                            digest = str(repo_digests[0] or "")
        if not digest:
            digest = image_id
        return {
            "dataset_name": self.config.dataset_name,
            "dataset_split": self.config.split,
            "env_image_key": image_key,
            "env_image_digest": digest,
            "env_image_id": image_id,
            "container_name": self._container_name,
        }

    def start(self) -> None:
        if self._started:
            return
        self._ensure_workspace_ready()
        self._ensure_env_image()
        run_cmd = [
            self.config.docker_binary,
            "run",
            "-d",
            "--rm",
            "--name",
            self._container_name,
            "-v",
            f"{self.workspace}:{CONTAINER_WORKDIR}",
            "-w",
            CONTAINER_WORKDIR,
        ]
        network_mode = str(os.environ.get("BCMR_DOCKER_NETWORK_MODE", "") or "").strip()
        if network_mode:
            run_cmd.extend(["--network", network_mode])
        for key, value in sorted(self.config.extra_env.items()):
            run_cmd.extend(["-e", f"{key}={value}"])
        run_cmd.extend([self._spec.env_image_key, "tail", "-f", "/dev/null"])
        start_timeout = max(1, int(self.config.container_start_timeout or 1))
        guarded_run_cmd = [
            "timeout",
            "--kill-after=5s",
            f"{start_timeout}s",
            *run_cmd,
        ]
        try:
            completed = subprocess.run(
                guarded_run_cmd,
                capture_output=True,
                text=True,
                timeout=start_timeout + 10,
                check=False,
                env=os.environ,
            )
        except subprocess.TimeoutExpired as exc:
            recovered = self._recover_created_container_after_run_timeout()
            if recovered:
                self._started = True
                try:
                    self._initialize_repo()
                except Exception:
                    self.close()
                    raise
                return
            raise RuntimeError(
                f"Harness runtime container start timed out after "
                f"{self.config.container_start_timeout} seconds for {self._container_name}:\n"
                f"{(exc.stdout or '')}{(exc.stderr or '')}"
            ) from exc
        if completed.returncode == 124:
            recovered = self._recover_created_container_after_run_timeout()
            if recovered:
                self._started = True
                try:
                    self._initialize_repo()
                except Exception:
                    self.close()
                    raise
                return
            raise RuntimeError(
                f"Harness runtime container start timed out after "
                f"{self.config.container_start_timeout} seconds for {self._container_name}:\n"
                f"{(completed.stdout or '')}{(completed.stderr or '')}"
            )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout or "failed to start harness runtime container")
        self._started = True
        try:
            self._initialize_repo()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if not self._started and not self._container_exists():
            return
        try:
            subprocess.run(
                [self.config.docker_binary, "rm", "-f", self._container_name],
                capture_output=True,
                text=True,
                timeout=self.config.container_cleanup_timeout,
                check=False,
                env=os.environ,
            )
        except subprocess.TimeoutExpired:
            pass
        self._started = False
        self._initialized = False

    def _recover_created_container_after_run_timeout(self) -> bool:
        if not self._wait_for_container_after_run_timeout():
            return False
        if self._container_is_running():
            return True
        try:
            completed = subprocess.run(
                [self.config.docker_binary, "start", self._container_name],
                capture_output=True,
                text=True,
                timeout=self.config.container_start_timeout,
                check=False,
                env=os.environ,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def _wait_for_container_after_run_timeout(self) -> bool:
        deadline = time.monotonic() + min(max(int(self.config.container_cleanup_timeout or 1), 1), 30)
        while True:
            if self._container_exists():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(1)

    def _container_exists(self) -> bool:
        try:
            completed = subprocess.run(
                [self.config.docker_binary, "inspect", self._container_name],
                capture_output=True,
                text=True,
                timeout=min(max(int(self.config.container_cleanup_timeout or 1), 1), 30),
                check=False,
                env=os.environ,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def _container_is_running(self) -> bool:
        try:
            completed = subprocess.run(
                [self.config.docker_binary, "inspect", "-f", "{{.State.Running}}", self._container_name],
                capture_output=True,
                text=True,
                timeout=min(max(int(self.config.container_cleanup_timeout or 1), 1), 30),
                check=False,
                env=os.environ,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0 and completed.stdout.strip().lower() == "true"


    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> dict[str, Any]:
        self.start()
        requested_cwd = cwd or self.config.cwd or str(self.workspace)
        container_cwd = self._map_cwd(requested_cwd)
        shell_command = self._wrap_command(command, container_cwd)
        exec_cmd, stdin_text = self._docker_exec_command(shell_command)
        try:
            completed = subprocess.run(
                exec_cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout or self.config.timeout,
                check=False,
                env=os.environ,
            )
            result = {
                "output": (completed.stdout or "") + (completed.stderr or ""),
                "returncode": int(completed.returncode),
            }
        except subprocess.TimeoutExpired as exc:
            result = {
                "output": (exc.stdout or "") + (exc.stderr or "") + f"\n命令执行超时 ({timeout or self.config.timeout}秒)",
                "returncode": -1,
            }
        self._sync_permissions()
        return result

    def evaluate_manifest(self, manifest: dict[str, Any], *, timeout: int = 1800) -> dict[str, Any]:
        fail_to_pass = self.execute(str(manifest["test_command"]), cwd=str(self.workspace), timeout=timeout)
        oracle = self.execute(str(manifest["oracle_command"]), cwd=str(self.workspace), timeout=timeout)
        return {
            "instance_id": manifest.get("instance_id"),
            "fail_to_pass_returncode": fail_to_pass["returncode"],
            "oracle_returncode": oracle["returncode"],
            "fail_to_pass_output": fail_to_pass["output"],
            "oracle_output": oracle["output"],
            "oracle_success": oracle["returncode"] == 0,
        }

    def get_template_vars(self) -> dict[str, Any]:
        return {
            "cwd": CONTAINER_WORKDIR,
            "host_cwd": str(self.workspace),
            "system": "Linux",
            "machine": "x86_64",
            "python_version": "",
        }

    def _map_cwd(self, cwd: str) -> str:
        if not cwd:
            return CONTAINER_WORKDIR
        cwd_path = Path(cwd).resolve()
        try:
            rel = cwd_path.relative_to(self.workspace)
        except ValueError:
            return cwd
        if str(rel) == ".":
            return CONTAINER_WORKDIR
        return f"{CONTAINER_WORKDIR}/{rel.as_posix()}"

    def _wrap_command(self, command: str, cwd: str) -> str:
        pieces = [
            "set -o pipefail",
            "source /opt/miniconda3/etc/profile.d/conda.sh",
            "conda activate testbed",
            f"cd {shlex.quote(cwd)}",
            command,
        ]
        return " && ".join(pieces)

    def _docker_exec_command(self, shell_command: str) -> tuple[list[str], str | None]:
        if len(shell_command) <= MAX_INLINE_DOCKER_EXEC_CHARS:
            return (
                [self.config.docker_binary, "exec", self._container_name, "bash", "-lc", shell_command],
                None,
            )
        script_path = f"/tmp/bcmr_exec_{uuid.uuid4().hex}.sh"
        loader = f"cat > {shlex.quote(script_path)} && bash {shlex.quote(script_path)}"
        return (
            [self.config.docker_binary, "exec", "-i", self._container_name, "bash", "-lc", loader],
            shell_command + "\n",
        )

    def _ensure_workspace_ready(self) -> None:
        git_dir = self.workspace / ".git"
        if not self.workspace.exists():
            raise FileNotFoundError(f"Workspace does not exist: {self.workspace}")
        if not git_dir.exists() or not git_dir.is_dir():
            raise ValueError(
                "Harness runtime requires a self-contained git workspace. "
                "Use materialize_workspace(..., strategy='git_clone') for official harness mode."
            )

    def _initialize_repo(self) -> None:
        if self._initialized:
            return
        with self._repo_init_lock():
            script = self._build_repo_setup_script()
            setup_timeout = max(1, int(self.config.setup_timeout or 1))
            init_cmd = [
                self.config.docker_binary,
                "exec",
                "-i",
                self._container_name,
                "bash",
                "-lc",
                (
                    "cat >/tmp/bcmr_init_repo.sh && "
                    f"timeout --kill-after=5s {setup_timeout}s bash /tmp/bcmr_init_repo.sh"
                ),
            ]
            try:
                completed = subprocess.run(
                    init_cmd,
                    input=script,
                    text=True,
                    capture_output=True,
                    timeout=setup_timeout + 10,
                    check=False,
                    env=os.environ,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Harness repo initialization timed out after {setup_timeout} seconds:\n"
                    f"{(exc.stdout or '')}{(exc.stderr or '')}"
                ) from exc
            if completed.returncode == 124:
                raise RuntimeError(
                    f"Harness repo initialization timed out after {setup_timeout} seconds:\n"
                    f"{(completed.stdout or '')}{(completed.stderr or '')}"
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    completed.stderr or completed.stdout or "failed to initialize harness workspace repository"
                )
            self._sync_permissions()
            self._initialized = True

    @contextmanager
    def _repo_init_lock(self) -> Iterator[None]:
        lock_root = Path(self.config.repo_init_lock_root)
        lock_root.mkdir(parents=True, exist_ok=True)
        image_key = ""
        if self._spec is not None:
            image_key = str(getattr(self._spec, "env_image_key", "") or "")
        repo_key = _slug_for_lock(image_key or self.instance_id.split("__", 1)[0] or self.instance_id)
        lock_path = lock_root / f"{repo_key}.lock"
        timeout = max(0, int(self.config.repo_init_lock_timeout or 0))
        deadline = time.monotonic() + timeout
        with lock_path.open("w", encoding="utf-8") as lock_file:
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if timeout == 0 or time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Harness repo initialization lock timed out for "
                            f"{repo_key} after {timeout} seconds"
                        ) from exc
                    time.sleep(0.25)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _build_repo_setup_script(self) -> str:
        lines = [
            "#!/bin/bash",
            "set -euxo pipefail",
            f"git config --global --add safe.directory {CONTAINER_WORKDIR}",
        ]
        for command in self._spec.repo_script_list:
            normalized = command.strip()
            if normalized.startswith("git clone -o origin "):
                continue
            if normalized.startswith("git reset --hard "):
                continue
            if normalized.startswith("TARGET_TIMESTAMP="):
                continue
            if normalized.startswith("git tag -l | while read tag;"):
                continue
            if normalized == "git reflog expire --expire=now --all":
                continue
            if normalized == "git gc --prune=now --aggressive":
                continue
            if normalized.startswith("AFTER_TIMESTAMP="):
                continue
            if normalized.startswith("COMMIT_COUNT="):
                continue
            if normalized == '[ "$COMMIT_COUNT" -eq 0 ] || exit 1':
                continue
            if normalized == "git remote remove origin":
                lines.append("git remote remove origin || true")
                continue
            lines.extend(
                _editable_install_bootstrap_commands(
                    command,
                    workspace=self.workspace,
                    instance=dict(self._instance or {}),
                )
            )
            lines.append(_offline_editable_install_command(command))
        if not self.config.skip_recursive_chmod:
            lines.append(f"chmod -R a+rwX {CONTAINER_WORKDIR}")
        return "\n".join(lines) + "\n"

    def _ensure_env_image(self) -> None:
        if self._spec is not None:
            return
        if str(self.config.env_image_key_override or "").strip():
            self._spec = _OverrideEnvSpec(
                env_image_key=str(self.config.env_image_key_override).strip(),
                repo_script_list=[],
            )
            return
        try:
            import docker  # type: ignore
            from swebench.harness.docker_build import build_env_images  # type: ignore
            from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Official harness runtime requires swebench + docker. "
                "Run this script with the swebench_tools interpreter or install official harness deps."
            ) from exc

        if self.config.dataset_instance:
            dataset = [dict(self.config.dataset_instance)]
            row_instance_id = str(dataset[0].get("instance_id", "") or "")
            if row_instance_id and row_instance_id != self.instance_id:
                raise ValueError(
                    f"Configured dataset instance {row_instance_id} does not match runtime instance {self.instance_id}"
                )
        else:
            from swebench.harness.utils import load_swebench_dataset  # type: ignore

            dataset = load_swebench_dataset(
                name=self.config.dataset_name,
                split=self.config.split,
                instance_ids=[self.instance_id],
            )
        if not dataset:
            raise ValueError(f"Instance {self.instance_id} was not found in dataset {self.config.dataset_name}:{self.config.split}")
        self._instance = dataset[0]
        with self._offline_spec_resource_patch(self._instance):
            self._spec = make_test_spec(
                self._instance,
                namespace=self.config.namespace,
                env_image_tag=self.config.env_image_tag,
                instance_image_tag=self.config.instance_image_tag,
            )
        client = docker.from_env()
        build_env_images(
            client,
            dataset=[self._spec],
            force_rebuild=self.config.force_rebuild,
            max_workers=self.config.max_workers,
            namespace=self.config.namespace,
            instance_image_tag=self.config.instance_image_tag,
            env_image_tag=self.config.env_image_tag,
        )

    @contextmanager
    def _offline_spec_resource_patch(self, instance: dict[str, Any]) -> Iterator[None]:
        """Use manifest snapshots for official TestSpec env resources.

        The official SWE-bench TestSpec code fetches requirements/environment
        files from GitHub by commit. Our manifests already pin a local source
        snapshot, so harness-mode runs should be reproducible and should fail
        clearly if the local resource is missing.
        """
        source_snapshot = str(instance.get("source_snapshot", "") or "").strip()
        if not source_snapshot:
            yield
            return
        snapshot = Path(source_snapshot).resolve()
        if not snapshot.exists():
            raise RuntimeError(
                "Harness offline spec resource blocker: source_snapshot does not exist "
                f"for {instance.get('instance_id')}: {snapshot}"
            )
        if not snapshot.is_dir():
            raise RuntimeError(
                "Harness offline spec resource blocker: source_snapshot is not a directory "
                f"for {instance.get('instance_id')}: {snapshot}"
            )

        harness_constants = importlib.import_module("swebench.harness.constants")
        python_spec = importlib.import_module("swebench.harness.test_spec.python")

        original_get_requirements = python_spec.get_requirements_by_commit
        original_get_environment = python_spec.get_environment_yml_by_commit
        expected_repo = str(instance.get("repo", "") or "")
        instance_id = str(instance.get("instance_id", "") or "")

        def _assert_repo(repo: str) -> None:
            if expected_repo and repo != expected_repo:
                raise RuntimeError(
                    "Harness offline spec resource blocker: requested repo does not match "
                    f"manifest repo for {instance_id}: requested={repo}, manifest={expected_repo}"
                )

        def _local_requirements(repo: str, commit: str) -> str:
            _assert_repo(repo)
            req_paths = harness_constants.MAP_REPO_TO_REQS_PATHS.get(repo)
            if not req_paths:
                raise RuntimeError(
                    "Harness offline spec resource blocker: no local requirements path map "
                    f"for repo {repo} at commit {commit} ({instance_id})"
                )
            req_file = self._first_local_spec_file(
                snapshot,
                req_paths,
                repo=repo,
                commit=commit,
                instance_id=instance_id,
                resource_kind="requirements.txt",
            )
            return self._read_local_requirements(snapshot, req_file)

        def _local_environment(repo: str, commit: str, env_name: str) -> str:
            _assert_repo(repo)
            env_paths = harness_constants.MAP_REPO_TO_ENV_YML_PATHS.get(repo)
            if not env_paths:
                raise RuntimeError(
                    "Harness offline spec resource blocker: no local environment.yml path map "
                    f"for repo {repo} at commit {commit} ({instance_id})"
                )
            env_file = self._first_local_spec_file(
                snapshot,
                env_paths,
                repo=repo,
                commit=commit,
                instance_id=instance_id,
                resource_kind="environment.yml",
            )
            lines = env_file.read_text(encoding="utf-8").split("\n")
            cleaned = [f"name: {env_name}" if line.startswith("name:") else line for line in lines]
            return "\n".join(cleaned)

        python_spec.get_requirements_by_commit = _local_requirements
        python_spec.get_environment_yml_by_commit = _local_environment
        try:
            yield
        finally:
            python_spec.get_requirements_by_commit = original_get_requirements
            python_spec.get_environment_yml_by_commit = original_get_environment

    def _first_local_spec_file(
        self,
        snapshot: Path,
        relative_paths: list[str],
        *,
        repo: str,
        commit: str,
        instance_id: str,
        resource_kind: str,
    ) -> Path:
        for relative in relative_paths:
            candidate = (snapshot / relative).resolve()
            try:
                candidate.relative_to(snapshot)
            except ValueError as exc:
                raise RuntimeError(
                    "Harness offline spec resource blocker: local resource path escapes "
                    f"snapshot for {instance_id}: {relative}"
                ) from exc
            if candidate.is_file():
                return candidate
        raise RuntimeError(
            "Harness offline spec resource blocker: missing local "
            f"{resource_kind} for repo {repo} at commit {commit} ({instance_id}); "
            f"searched {relative_paths} under {snapshot}"
        )

    def _read_local_requirements(self, snapshot: Path, req_file: Path) -> str:
        original_req: list[str] = []
        additional_reqs: list[str] = []
        req_dir = req_file.parent

        def exclude_line(line: str) -> bool:
            return any(line.strip().startswith(prefix) for prefix in ["-e .", "#", ".[test"])

        for line in req_file.read_text(encoding="utf-8").split("\n"):
            if line.strip().startswith("-r"):
                file_name = line[len("-r") :].strip()
                nested_req = (req_dir / file_name).resolve()
                try:
                    nested_req.relative_to(snapshot)
                except ValueError as exc:
                    raise RuntimeError(
                        "Harness offline spec resource blocker: nested requirements path escapes "
                        f"snapshot: {file_name}"
                    ) from exc
                if not nested_req.is_file():
                    raise RuntimeError(
                        "Harness offline spec resource blocker: missing nested local requirements "
                        f"{file_name} referenced by {req_file}"
                    )
                for nested_line in nested_req.read_text(encoding="utf-8").split("\n"):
                    if not exclude_line(nested_line):
                        additional_reqs.append(nested_line)
            elif not exclude_line(line):
                original_req.append(line)

        additional_reqs.append("\n".join(original_req))
        return "\n".join(additional_reqs)

    def _sync_permissions(self) -> None:
        if not self._started or self.config.skip_recursive_chmod:
            return
        try:
            subprocess.run(
                [
                    self.config.docker_binary,
                    "exec",
                    self._container_name,
                    "bash",
                    "-lc",
                    f"chmod -R a+rwX {CONTAINER_WORKDIR} || true",
                ],
                capture_output=True,
                text=True,
                timeout=self.config.container_cleanup_timeout,
                check=False,
                env=os.environ,
            )
        except subprocess.TimeoutExpired:
            pass
