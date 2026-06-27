"""Docker-backed harness bridge for BCMR-SWE evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


@dataclass
class DockerHarnessConfig:
    image: str = "python:3.12-slim"
    workspace_mount: str = "/workspace"
    timeout_sec: int = 1800
    docker_binary: str = "docker"


class DockerHarnessBridge:
    """Run evaluation commands inside Docker containers.

    This is intentionally generic so we can plug in the official SWE-bench
    harness image later without rewriting BCMR's recovery pipeline.
    """

    def __init__(self, config: DockerHarnessConfig | None = None):
        self.config = config or DockerHarnessConfig()

    def check_available(self) -> bool:
        try:
            result = subprocess.run(
                [self.config.docker_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def run_shell(self, *, workspace: str | Path, command: str, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
        workspace = Path(workspace).resolve()
        env_args: list[str] = []
        for key, value in sorted((extra_env or {}).items()):
            env_args.extend(["-e", f"{key}={value}"])
        docker_cmd = [
            self.config.docker_binary,
            "run",
            "--rm",
            "-v",
            f"{workspace}:{self.config.workspace_mount}",
            "-w",
            self.config.workspace_mount,
            *env_args,
            self.config.image,
            "bash",
            "-lc",
            command,
        ]
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_sec,
        )
        return {
            "command": command,
            "docker_command": docker_cmd,
            "output": (result.stdout or "") + (result.stderr or ""),
            "returncode": result.returncode,
        }

    def evaluate_manifest(self, manifest: dict[str, Any], *, workspace: str | Path) -> dict[str, Any]:
        fail_to_pass = self.run_shell(workspace=workspace, command=str(manifest["test_command"]))
        oracle = self.run_shell(workspace=workspace, command=str(manifest["oracle_command"]))
        return {
            "instance_id": manifest.get("instance_id"),
            "fail_to_pass_returncode": fail_to_pass["returncode"],
            "oracle_returncode": oracle["returncode"],
            "fail_to_pass_output": fail_to_pass["output"],
            "oracle_output": oracle["output"],
            "oracle_success": oracle["returncode"] == 0,
        }
