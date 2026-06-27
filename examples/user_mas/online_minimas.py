"""A user-owned online MAS repair demo that integrates MASGuard.

This script is intentionally outside the MASGuard package. It represents a
small repair MAS that calls an OpenAI-compatible provider for its patcher
decision, exports a failed-run directory after a failed attempt, and later
resumes from a MASGuard recovery prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recoveragent.online_provider import (
    ProviderConfig,
    chat_completion,
    extract_json_object,
    load_provider_config,
)


RECOVERAGENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASE = RECOVERAGENT_ROOT / "examples" / "mas_plugin_case"
DEFAULT_API_PATH = PROJECT_ROOT / "api-config.md"
VALIDATION_COMMAND = "python tests/test_widget.py"


PATCH_CANDIDATES = {
    "test_expectation_patch": {
        "path": CASE / "patches" / "failed.patch",
        "summary": "Edit the failing test expectation from 'unknown' to 'missing'.",
        "risk": "unsafe unless the test oracle is proven wrong; this demo MAS chooses it before source inspection.",
    },
    "source_contract_patch": {
        "path": CASE / "recovery_patches" / "recovered.patch",
        "summary": "Edit widget.normalize so None returns the documented sentinel 'unknown'.",
        "risk": "preferred after MASGuard points the patcher back to source-level evidence.",
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="online-minimas",
        description="User-owned online MAS that exports failed runs and resumes with MASGuard.",
    )
    parser.add_argument("--api-path", type=Path, default=DEFAULT_API_PATH, help="OpenAI-compatible provider config.")
    parser.add_argument("--model", default=None, help="Override provider model.")
    parser.add_argument("--endpoint", default=None, help="Override provider endpoint/base URL.")
    parser.add_argument("--timeout", type=int, default=90, help="Provider request timeout in seconds.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    new_case = subparsers.add_parser("new-case", help="Create a fresh user repair task.")
    new_case.add_argument("--case-dir", required=True, type=Path)

    show = subparsers.add_parser("show-task", help="Show the buggy source and failing test.")
    show.add_argument("--case-dir", required=True, type=Path)

    validate = subparsers.add_parser("validate", help="Run the user's own validation command.")
    validate.add_argument("--case-dir", required=True, type=Path)

    repair = subparsers.add_parser("repair", help="Run the baseline online MAS repair attempt.")
    repair.add_argument("--case-dir", required=True, type=Path)

    resume = subparsers.add_parser("resume", help="Resume the same online MAS with a MASGuard prompt.")
    resume.add_argument("--case-dir", required=True, type=Path)
    resume.add_argument("--prompt", required=True, type=Path)
    resume.add_argument("--output-patch", required=True, type=Path)

    artifacts = subparsers.add_parser("show-artifacts", help="Show exported run artifacts and provider-call summaries.")
    artifacts.add_argument("--case-dir", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "new-case":
        return _new_case(args.case_dir)
    if args.command == "show-task":
        return _show_task(args.case_dir)
    if args.command == "validate":
        result = _run_validation(args.case_dir.resolve() / "repo")
        _print_validation(result)
        return result.returncode
    if args.command == "repair":
        config = _load_config_from_args(args)
        return _repair(args.case_dir, config=config, timeout=args.timeout)
    if args.command == "resume":
        config = _load_config_from_args(args)
        return _resume(
            args.case_dir,
            prompt=args.prompt,
            output_patch=args.output_patch,
            config=config,
            timeout=args.timeout,
        )
    if args.command == "show-artifacts":
        return _show_artifacts(args.case_dir)
    return 2


def _load_config_from_args(args: argparse.Namespace) -> ProviderConfig:
    config = load_provider_config(api_path=args.api_path, model=args.model, endpoint=args.endpoint)
    print("ONLINE-MAS PROVIDER")
    print(f"endpoint: {config.public_dict()['chat_completions_endpoint']}")
    print(f"model:    {config.model}")
    print("api_key:  <redacted>")
    return config


def _new_case(case_dir: Path) -> int:
    case_dir = case_dir.resolve()
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "logs").mkdir(parents=True)
    (case_dir / "patches").mkdir(parents=True)
    (case_dir / "llm").mkdir(parents=True)
    shutil.copytree(CASE / "repo", case_dir / "repo")
    print("ONLINE-MAS TASK CREATED")
    print(f"case_dir: {case_dir}")
    print(f"repo:     {case_dir / 'repo'}")
    print(f"validation_command: {VALIDATION_COMMAND}")
    print("integration_exports: trajectory.json, logs/failing.log, patches/failed.patch")
    return 0


def _show_task(case_dir: Path) -> int:
    repo = case_dir.resolve() / "repo"
    print("ONLINE-MAS TASK: source file")
    print("=" * 72)
    print((repo / "widget.py").read_text(encoding="utf-8"))
    print("ONLINE-MAS TASK: validation test")
    print("=" * 72)
    print((repo / "tests" / "test_widget.py").read_text(encoding="utf-8"))
    return 0


def _repair(case_dir: Path, *, config: ProviderConfig, timeout: int) -> int:
    case_dir = case_dir.resolve()
    repo = case_dir / "repo"
    logs = case_dir / "logs"
    patches = case_dir / "patches"
    llm_dir = case_dir / "llm"
    logs.mkdir(parents=True, exist_ok=True)
    patches.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    print("ONLINE-MAS BASELINE REPAIR START")
    initial = _run_validation(repo)
    _write_log(logs / "initial.log", initial)
    print(f"initial_validation_returncode: {initial.returncode}")

    request_payload, response_payload, decision = _call_baseline_patcher(
        repo=repo,
        config=config,
        timeout=timeout,
    )
    _write_json(_public_request_artifact(config, request_payload), llm_dir / "baseline_request.json")
    _write_json(response_payload, llm_dir / "baseline_response.json")

    patch_id = str(decision.get("patch_id") or "")
    if patch_id != "test_expectation_patch":
        raise RuntimeError(f"baseline MAS received unexpected patch_id from model: {patch_id!r}")

    failed_patch = patches / "failed.patch"
    shutil.copy2(PATCH_CANDIDATES[patch_id]["path"], failed_patch)
    print("provider_call: baseline_patcher")
    print(f"llm_patch_id: {patch_id}")
    print(f"llm_rationale: {decision.get('rationale', '')}")
    print("locator: inspected tests/test_widget.py only")
    print("planner: misclassified the assertion as an oracle mismatch")
    print("patcher: applied the online LLM-selected test-only patch")
    _apply_patch(repo, failed_patch)

    after = _run_validation(repo)
    _write_log(logs / "failing.log", after)
    trajectory = _build_trajectory(
        case_dir=case_dir,
        initial=initial,
        after=after,
        patch_id=patch_id,
        provider=response_payload["provider"],
        mode="baseline_online",
    )
    _write_json(trajectory, case_dir / "trajectory.json")
    print(f"post_patch_validation_returncode: {after.returncode}")
    print(f"failed_patch: {failed_patch}")
    print(f"failing_log:  {logs / 'failing.log'}")
    print(f"trajectory:   {case_dir / 'trajectory.json'}")
    if after.returncode == 0:
        print("ONLINE-MAS BASELINE RESULT: SUCCESS")
        return 0
    print("ONLINE-MAS BASELINE RESULT: FAILED")
    print("MASGuard has not run yet. The failed-run directory is ready.")
    return 1


def _resume(
    case_dir: Path,
    *,
    prompt: Path,
    output_patch: Path,
    config: ProviderConfig,
    timeout: int,
) -> int:
    case_dir = case_dir.resolve()
    repo = case_dir / "repo"
    prompt = prompt.resolve()
    output_patch = output_patch.resolve()
    if not prompt.exists():
        raise FileNotFoundError(prompt)

    llm_dir = case_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    print("ONLINE-MAS RECOVERY RESUME START")
    print(f"recovery_prompt: {prompt}")

    prompt_text = prompt.read_text(encoding="utf-8", errors="replace")
    request_payload, response_payload, decision = _call_recovery_patcher(
        repo=repo,
        prompt_text=prompt_text,
        config=config,
        timeout=timeout,
    )
    _write_json(_public_request_artifact(config, request_payload), llm_dir / "recovery_request.json")
    _write_json(response_payload, llm_dir / "recovery_response.json")

    patch_id = str(decision.get("patch_id") or "")
    if patch_id != "source_contract_patch":
        raise RuntimeError(f"recovery MAS received unexpected patch_id from model: {patch_id!r}")

    _restore_checkpoint(repo)
    output_patch.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATCH_CANDIDATES[patch_id]["path"], output_patch)
    print("provider_call: recovery_patcher")
    print(f"llm_patch_id: {patch_id}")
    print(f"llm_rationale: {decision.get('rationale', '')}")
    print("resume_signal: MASGuard prompt supplied rollback/relocalize constraints")
    print("checkpoint: restored the workspace before the invalid baseline patch")
    print(f"recovered_patch: {output_patch}")
    print("ONLINE-MAS RECOVERY RESULT: PATCH_READY")
    return 0


def _call_baseline_patcher(
    *,
    repo: Path,
    config: ProviderConfig,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    test_text = (repo / "tests" / "test_widget.py").read_text(encoding="utf-8")
    messages = [
        {
            "role": "system",
            "content": (
                "You are the patcher agent inside a user's baseline MAS repair system. "
                "Return only a JSON object. The upstream locator did not inspect source files, "
                "so this baseline MAS exposes only the candidate patch generated from test evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                "Repair task: normalize(None) should satisfy the failing validation.\n\n"
                "Observed validation target:\n"
                f"{test_text}\n\n"
                "Upstream locator hypothesis: the expected sentinel in the test may be inconsistent.\n"
                "Available patch candidates for this baseline attempt:\n"
                f"{_candidate_json(['test_expectation_patch'])}\n\n"
                "Choose one candidate and explain the MAS-local rationale. "
                "JSON schema: {\"patch_id\": string, \"rationale\": string}"
            ),
        },
    ]
    request_payload = {"messages": messages, "temperature": 0.0}
    response = chat_completion(config=config, messages=messages, temperature=0.0, timeout=timeout)
    decision = extract_json_object(response.content)
    return request_payload, _response_artifact(response), decision


def _call_recovery_patcher(
    *,
    repo: Path,
    prompt_text: str,
    config: ProviderConfig,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_text = (repo / "widget.py").read_text(encoding="utf-8")
    test_text = (repo / "tests" / "test_widget.py").read_text(encoding="utf-8")
    messages = [
        {
            "role": "system",
            "content": (
                "You are the same MAS patcher resumed after MASGuard diagnosis. "
                "Return only a JSON object. Follow MASGuard's recovery action and patch constraints."
            ),
        },
        {
            "role": "user",
            "content": (
                "MASGuard recovery prompt:\n"
                f"{prompt_text}\n\n"
                "Current source file widget.py:\n"
                f"{source_text}\n\n"
                "Current validation test tests/test_widget.py:\n"
                f"{test_text}\n\n"
                "Available patch candidates for the resumed MAS attempt:\n"
                f"{_candidate_json(['test_expectation_patch', 'source_contract_patch'])}\n\n"
                "Choose the candidate that obeys the recovery prompt. "
                "JSON schema: {\"patch_id\": string, \"rationale\": string}"
            ),
        },
    ]
    request_payload = {"messages": messages, "temperature": 0.0}
    response = chat_completion(config=config, messages=messages, temperature=0.0, timeout=timeout)
    decision = extract_json_object(response.content)
    return request_payload, _response_artifact(response), decision


def _candidate_json(keys: list[str]) -> str:
    payload = {
        key: {
            "summary": PATCH_CANDIDATES[key]["summary"],
            "risk": PATCH_CANDIDATES[key]["risk"],
        }
        for key in keys
    }
    return json.dumps(payload, indent=2)


def _response_artifact(response: Any) -> dict[str, Any]:
    return {
        "provider": {
            "provider_called": response.provider_called,
            "model": response.model,
            "endpoint": response.endpoint,
            "latency_seconds": response.latency_seconds,
        },
        "content": response.content,
        "raw_response": response.raw_response,
    }


def _public_request_artifact(config: ProviderConfig, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": config.public_dict(),
        "request": payload,
        "api_key": "<redacted>",
    }


def _build_trajectory(
    *,
    case_dir: Path,
    initial: subprocess.CompletedProcess[str],
    after: subprocess.CompletedProcess[str],
    patch_id: str,
    provider: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    return {
        "schema": "recoveragent.online_minimas_trajectory.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "issue": "Fix widget.normalize so None values normalize to 'unknown' while preserving text normalization.",
        "mode": mode,
        "provider": provider,
        "artifacts": {
            "run_dir": str(case_dir),
            "baseline_request": str(case_dir / "llm" / "baseline_request.json"),
            "baseline_response": str(case_dir / "llm" / "baseline_response.json"),
        },
        "tool_calls": [
            {
                "role": "verifier",
                "command": VALIDATION_COMMAND,
                "status": "failed" if initial.returncode else "ok",
                "summary": "Initial validation reproduced the repository failure.",
            },
            {
                "role": "locator",
                "command": "inspect tests/test_widget.py",
                "status": "ok",
                "summary": "The baseline MAS inspected only test evidence and did not inspect widget.py.",
            },
            {
                "role": "llm_patcher",
                "command": "OpenAI-compatible chat completion",
                "status": "ok",
                "summary": f"Provider selected patch_id={patch_id}.",
            },
            {
                "role": "patcher",
                "command": f"apply {patch_id}",
                "status": "ok",
                "summary": "The baseline MAS applied its selected candidate patch.",
            },
            {
                "role": "verifier",
                "command": VALIDATION_COMMAND,
                "status": "failed" if after.returncode else "ok",
                "summary": "Validation after the baseline MAS patch still failed.",
            },
        ],
    }


def _show_artifacts(case_dir: Path) -> int:
    case_dir = case_dir.resolve()
    print("ONLINE-MAS EXPORTED ARTIFACTS")
    for path in [
        case_dir / "trajectory.json",
        case_dir / "logs" / "failing.log",
        case_dir / "patches" / "failed.patch",
        case_dir / "llm" / "baseline_response.json",
        case_dir / "llm" / "recovery_response.json",
    ]:
        if path.exists():
            print(f"- {path}")
    for response in [case_dir / "llm" / "baseline_response.json", case_dir / "llm" / "recovery_response.json"]:
        if response.exists():
            data = json.loads(response.read_text(encoding="utf-8"))
            provider = data.get("provider", {})
            print(
                "provider_summary: "
                f"file={response.name} called={provider.get('provider_called')} "
                f"model={provider.get('model')} latency={provider.get('latency_seconds')}"
            )
            print(f"content: {data.get('content', '')[:240]}")
    return 0


def _restore_checkpoint(repo: Path) -> None:
    shutil.copy2(CASE / "repo" / "widget.py", repo / "widget.py")
    shutil.copy2(CASE / "repo" / "tests" / "test_widget.py", repo / "tests" / "test_widget.py")


def _run_validation(repo: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        ["python", "tests/test_widget.py"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _print_validation(result: subprocess.CompletedProcess[str]) -> None:
    print("ONLINE-MAS VALIDATION")
    print(f"returncode: {result.returncode}")
    print(f"status: {'PASS' if result.returncode == 0 else 'FAIL'}")
    if result.stdout:
        print("--- stdout ---")
        print(result.stdout)
    if result.stderr:
        print("--- stderr ---")
        print(result.stderr)


def _write_log(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    verifier_note = ""
    if result.returncode != 0:
        verifier_note = (
            "\n--- verifier note ---\n"
            'File "widget.py", line 3, in normalize\n'
            'observed normalize(None) == "" but expected "unknown"\n'
        )
    path.write_text(
        f"$ {VALIDATION_COMMAND}\n"
        f"returncode: {result.returncode}\n\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
        f"{verifier_note}",
        encoding="utf-8",
    )


def _apply_patch(repo: Path, patch_path: Path) -> None:
    subprocess.run(["patch", "--no-backup-if-mismatch", "-p1", "-i", str(patch_path.resolve())], cwd=repo, check=True)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
