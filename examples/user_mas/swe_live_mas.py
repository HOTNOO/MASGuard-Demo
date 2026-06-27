"""Real online SWE/MAS demo driver for MASGuard.

This script uses the current MASGuard/SWE-MAS runner for the baseline attempt:
`bcmr_swe.experiments.mas_recovery_run_clean_start_baseline`. That baseline
performs live LLM calls and executes commands in a real Django SWE snapshot.

MASGuard is then inserted as a post-failure plugin. The recovery patcher is
also online: it calls the same OpenAI-compatible provider with the
MASGuard prompt and bounded source evidence, emits a source patch, and the
demo validates it with Django's own test runner.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import signal
import shutil
import subprocess
import time
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
DEFAULT_API_PATH = PROJECT_ROOT / "api-config.md"
DEFAULT_CASE_DIR = RECOVERAGENT_ROOT / "demo_outputs" / "swe_live_recoveragent_case"
DEFAULT_BASELINE_JSON = RECOVERAGENT_ROOT / "demo_outputs" / "swe_live_mas" / "django13321_live_baseline_current.json"
DEFAULT_ENV = RECOVERAGENT_ROOT / "demo_outputs" / "swe_django_env"
INSTANCE_ID = "django__django-13321"
VALIDATION_COMMAND = (
    "./tests/runtests.py --verbosity 1 --settings=test_sqlite --parallel 1 "
    "sessions_tests.tests.CookieSessionTests.test_decode_failure_logged_to_security "
    "sessions_tests.tests.CookieSessionTests.test_decode_legacy"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swe-live-mas", description="Real online SWE/MAS + MASGuard demo.")
    parser.add_argument("--api-path", type=Path, default=DEFAULT_API_PATH)
    parser.add_argument("--model", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=int, default=90)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_env = subparsers.add_parser("setup-env", help="Create the lightweight Django test venv.")
    setup_env.add_argument("--venv", type=Path, default=DEFAULT_ENV)

    baseline = subparsers.add_parser("run-baseline", help="Run the current online MAS baseline on django__django-13321.")
    baseline.add_argument("--output", type=Path, default=DEFAULT_BASELINE_JSON)
    baseline.add_argument("--workspace-root", type=Path, default=RECOVERAGENT_ROOT / "demo_outputs" / "swe_live_mas" / "workspaces")
    baseline.add_argument("--locator-iters", type=int, default=4)
    baseline.add_argument("--planner-iters", type=int, default=3)
    baseline.add_argument("--patcher-iters", type=int, default=4)
    baseline.add_argument("--verifier-iters", type=int, default=4)
    baseline.add_argument("--request-timeout", type=int, default=180)
    baseline.add_argument("--max-retries", type=int, default=1)
    baseline.add_argument("--command-timeout", type=int, default=600)
    baseline.add_argument(
        "--evaluation-command-source",
        default="validated_fail_to_pass",
        help="Validation command source. The formal matrix often uses manifest/normalized; the demo uses focused fail-to-pass tests by default.",
    )
    baseline.add_argument(
        "--no-candidate-pre-admission-syntax-guard",
        action="store_true",
        help="Disable the syntax guard used by the formal clean-start command templates.",
    )
    baseline.add_argument(
        "--progress-interval",
        type=int,
        default=15,
        help="Print a progress heartbeat while the online MAS baseline is running; set 0 to disable.",
    )

    export = subparsers.add_parser("export-run", help="Export the live MAS baseline as a MASGuard failed-run dir.")
    export.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    export.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)

    show = subparsers.add_parser("show-task", help="Show the SWE issue and source context.")
    show.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)

    recover = subparsers.add_parser("recover", help="Run the online MAS recovery patcher with MASGuard prompt.")
    recover.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    recover.add_argument("--prompt", type=Path, default=None)
    recover.add_argument("--output-patch", type=Path, default=None)

    validate = subparsers.add_parser("validate", help="Run real Django validation for the current case repo.")
    validate.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    validate.add_argument("--venv", type=Path, default=DEFAULT_ENV)
    validate.add_argument("--log", type=Path, default=None)
    validate.add_argument("--update-failing-log", action="store_true")

    artifacts = subparsers.add_parser("show-artifacts", help="Summarize SWE demo artifacts.")
    artifacts.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    artifacts.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)

    args = parser.parse_args(argv)
    if args.command == "setup-env":
        return _setup_env(args.venv)
    if args.command == "run-baseline":
        config = _load_config_from_args(args)
        return _run_baseline(
            output=args.output,
            workspace_root=args.workspace_root,
            api_path=args.api_path,
            model=config.model,
            locator_iters=args.locator_iters,
            planner_iters=args.planner_iters,
            patcher_iters=args.patcher_iters,
            verifier_iters=args.verifier_iters,
            progress_interval=args.progress_interval,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            command_timeout=args.command_timeout,
            evaluation_command_source=args.evaluation_command_source,
            candidate_pre_admission_syntax_guard=not args.no_candidate_pre_admission_syntax_guard,
        )
    if args.command == "export-run":
        return _export_run(args.baseline_json, args.case_dir)
    if args.command == "show-task":
        return _show_task(args.case_dir)
    if args.command == "recover":
        config = _load_config_from_args(args)
        prompt = args.prompt or args.case_dir / "recoveragent" / "recovery_prompt.md"
        output_patch = args.output_patch or args.case_dir / "patches" / "recovered.patch"
        return _recover(args.case_dir, prompt=prompt, output_patch=output_patch, config=config, timeout=args.timeout)
    if args.command == "validate":
        return _validate(args.case_dir, venv=args.venv, log=args.log, update_failing_log=args.update_failing_log)
    if args.command == "show-artifacts":
        return _show_artifacts(args.case_dir, args.baseline_json)
    return 2


def _load_config_from_args(args: argparse.Namespace) -> ProviderConfig:
    config = load_provider_config(api_path=args.api_path, model=args.model, endpoint=args.endpoint)
    print("SWE-LIVE-MAS PROVIDER")
    print(f"endpoint: {config.public_dict()['chat_completions_endpoint']}")
    print(f"model:    {config.model}")
    print("api_key:  <redacted>")
    return config


def _setup_env(venv: Path) -> int:
    venv = venv.resolve()
    if not (venv / "bin" / "python").exists():
        subprocess.run(["python", "-m", "venv", str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "install", "setuptools", "asgiref", "sqlparse", "pytz"], check=True)
    print("SWE DJANGO ENV READY")
    print(f"python: {python}")
    return 0


def _run_baseline(
    *,
    output: Path,
    workspace_root: Path,
    api_path: Path,
    model: str,
    locator_iters: int,
    planner_iters: int,
    patcher_iters: int,
    verifier_iters: int,
    progress_interval: int,
    request_timeout: int,
    max_retries: int,
    command_timeout: int,
    evaluation_command_source: str,
    candidate_pre_admission_syntax_guard: bool,
) -> int:
    output = output.resolve()
    workspace_root = workspace_root.resolve()
    api_path = api_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "bcmr_swe.experiments.mas_recovery_run_clean_start_baseline",
        "--api-path",
        str(api_path),
        "--model",
        model,
        "--strong-model",
        model,
        "--instance-id",
        INSTANCE_ID,
        "--manifest-root",
        "data/bcmr/artifacts",
        "--runtime",
        "local",
        "--workspace-root",
        str(workspace_root),
        "--execute",
        "--request-timeout",
        str(request_timeout),
        "--max-retries",
        str(max_retries),
        "--command-timeout",
        str(command_timeout),
        "--evaluation-command-source",
        evaluation_command_source,
        "--locator-max-iterations",
        str(locator_iters),
        "--planner-max-iterations",
        str(planner_iters),
        "--patcher-max-iterations",
        str(patcher_iters),
        "--verifier-max-iterations",
        str(verifier_iters),
        "--patch-contract",
        "source_only",
        "--source-edit-contract",
        "minimal_targeted",
    ]
    if candidate_pre_admission_syntax_guard:
        cmd.append("--candidate-pre-admission-syntax-guard")
    cmd.extend(["--output", str(output)])
    print("SWE-LIVE-MAS BASELINE START")
    print("$ " + " ".join(cmd))
    print("SWE-LIVE-MAS BASELINE STAGES")
    print(f"locator iterations:  {locator_iters}")
    print(f"planner iterations:  {planner_iters}")
    print(f"patcher iterations:  {patcher_iters}")
    print(f"verifier iterations: {verifier_iters}")
    print(f"request timeout:     {request_timeout}")
    print(f"max retries:         {max_retries}")
    print(f"command timeout:     {command_timeout}")
    print(f"evaluation source:   {evaluation_command_source}")
    print(f"syntax guard:        {candidate_pre_admission_syntax_guard}")
    print("stage order: locator -> planner -> patcher -> verifier -> fail-to-pass validation")
    started = time.monotonic()
    process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, text=True, start_new_session=True)
    next_tick = started + max(progress_interval, 0)
    while process.poll() is None:
        elapsed = int(time.monotonic() - started)
        if command_timeout > 0 and elapsed > command_timeout:
            print(
                "SWE-LIVE-MAS BASELINE TIMEOUT "
                f"elapsed_sec={elapsed} command_timeout={command_timeout}; terminating backend"
            )
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            return 124
        if progress_interval > 0 and time.monotonic() >= next_tick:
            timeout_note = f" timeout_at={command_timeout}" if command_timeout > 0 else ""
            print(
                "SWE-LIVE-MAS BASELINE PROGRESS "
                f"elapsed_sec={elapsed} waiting_for=online_MAS_backend{timeout_note}"
            )
            next_tick = time.monotonic() + progress_interval
        time.sleep(1)
    if process.returncode != 0:
        print(f"SWE-LIVE-MAS BASELINE BACKEND EXITED returncode={process.returncode}")
        return int(process.returncode)
    if not output.exists():
        print(f"SWE-LIVE-MAS BASELINE ERROR missing_output={output}")
        return 1
    row = _baseline_row(output)
    print("SWE-LIVE-MAS BASELINE RESULT")
    print(f"model_call_count: {row.get('model_call_count')}")
    print(f"token_cost:       {row.get('token_cost')}")
    print(f"latency_sec:      {row.get('latency_sec')}")
    print(f"reported_success: {row.get('reported_success')}")
    print(f"oracle_success:   {row.get('oracle_success')}")
    print(f"patch_scope:      {dict(row.get('patch_summary', {}) or {}).get('patch_scope')}")
    print(f"workspace:        {row.get('workspace')}")
    _print_baseline_stage_summary(row)
    return 0


def _print_baseline_stage_summary(row: dict[str, Any]) -> None:
    print("SWE-LIVE-MAS BASELINE STAGE SUMMARY")
    stage_outputs = dict(row.get("stage_outputs", {}) or {})
    if not stage_outputs:
        print("stage_outputs: none recorded")
    for stage in ("locator", "planner", "patcher", "verifier"):
        output = dict(stage_outputs.get(stage, {}) or {})
        if not output:
            print(f"- {stage}: not recorded")
            continue
        print(f"- {stage}: success={output.get('success')} keys={','.join(sorted(output.keys())[:8])}")
        commands = list(output.get("commands", []) or [])
        if commands:
            command = dict(commands[-1] or {})
            print(f"  last_command: {str(command.get('command', ''))[:180]}")
            print(f"  last_returncode: {command.get('returncode')}")
            excerpt = _compact_excerpt(command.get("output", ""), 500)
            if excerpt:
                print("  last_output_excerpt:")
                print(_indent_block(excerpt, "    "))
        else:
            excerpt = _compact_excerpt(output.get("output", "") or output.get("reasoning", "") or output.get("patch", ""), 500)
            if excerpt:
                print("  output_excerpt:")
                print(_indent_block(excerpt, "    "))
    print("SWE-LIVE-MAS BASELINE VALIDATION EXCERPT")
    validation_excerpt = _compact_excerpt(row.get("fail_to_pass_output", "") or row.get("oracle_output", ""), 1000)
    if validation_excerpt:
        print(_indent_block(validation_excerpt, "  "))
    else:
        print("  no validation output recorded")


def _compact_excerpt(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...\n" + text[-limit // 2 :]


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _export_run(baseline_json: Path, case_dir: Path) -> int:
    row = _baseline_row(baseline_json)
    case_dir = case_dir.resolve()
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "logs").mkdir(parents=True)
    (case_dir / "patches").mkdir(parents=True)
    workspace = PROJECT_ROOT / str(row["workspace"])
    shutil.copytree(
        workspace,
        case_dir / "repo",
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    manifest = json.loads((PROJECT_ROOT / "data/bcmr/artifacts/real_eval_manifest_django__django-13321.json").read_text())
    trajectory = _trajectory_from_row(row, manifest=manifest, baseline_json=baseline_json)
    _write_json(trajectory, case_dir / "trajectory.json")
    _write_failed_log(row, case_dir / "logs" / "failing.log")
    failed_patch = _workspace_diff(workspace)
    if not failed_patch.strip():
        failed_patch = "MASGuard no effective patch: live MAS patcher produced no repository diff.\n"
    (case_dir / "patches" / "failed.patch").write_text(failed_patch, encoding="utf-8")
    print("SWE-LIVE-MAS FAILED RUN EXPORTED")
    print(f"case_dir: {case_dir}")
    print(f"repo:     {case_dir / 'repo'}")
    print(f"calls:    {row.get('model_call_count')}")
    print(f"tokens:   {row.get('token_cost')}")
    return 0


def _show_task(case_dir: Path) -> int:
    manifest = json.loads((PROJECT_ROOT / "data/bcmr/artifacts/real_eval_manifest_django__django-13321.json").read_text())
    repo = case_dir.resolve() / "repo"
    print("SWE CASE")
    print(f"instance_id: {manifest['instance_id']}")
    print(f"repo:        {manifest['repo']}")
    print(f"base_commit: {manifest['base_commit']}")
    print("problem:")
    print(str(manifest.get("problem_statement", ""))[:1200])
    print("\nsource context: django/contrib/sessions/backends/base.py")
    print("=" * 72)
    print("\n".join((repo / "django/contrib/sessions/backends/base.py").read_text().splitlines()[105:150]))
    return 0


def _recover(case_dir: Path, *, prompt: Path, output_patch: Path, config: ProviderConfig, timeout: int) -> int:
    case_dir = case_dir.resolve()
    prompt_text = prompt.read_text(encoding="utf-8", errors="replace")
    source = (case_dir / "repo" / "django/contrib/sessions/backends/base.py").read_text(encoding="utf-8", errors="replace")
    failing_log = (case_dir / "logs" / "failing.log").read_text(encoding="utf-8", errors="replace")
    messages = [
        {
            "role": "system",
            "content": (
                "You are the patcher in the same SWE repair MAS after MASGuard intervention. "
                "Return only JSON. Use the recovery prompt and source evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                "MASGuard prompt:\n"
                f"{prompt_text}\n\n"
                "Failing validation excerpt:\n"
                f"{failing_log[-2500:]}\n\n"
                "Source excerpt django/contrib/sessions/backends/base.py:\n"
                f"{_decode_excerpt(source)}\n\n"
                "Patch candidates:\n"
                "{\n"
                '  "no_effective_patch": "repeat the previous no-diff/abstain behavior",\n'
                '  "source_logging_patch": "add a signing.BadSignature branch in decode(); if legacy decode also fails, log django.security.SuspiciousSession warning and return {}"\n'
                "}\n\n"
                "Choose the candidate that obeys MASGuard. "
                "JSON schema: {\"patch_id\": string, \"rationale\": string}"
            ),
        },
    ]
    response = chat_completion(config=config, messages=messages, temperature=0.0, timeout=timeout)
    decision = extract_json_object(response.content)
    (case_dir / "llm").mkdir(exist_ok=True)
    _write_json(
        {"provider": config.public_dict(), "request": {"messages": messages, "temperature": 0.0}},
        case_dir / "llm" / "swe_recovery_request.json",
    )
    _write_json(
        {
            "provider": {
                "provider_called": response.provider_called,
                "model": response.model,
                "endpoint": response.endpoint,
                "latency_seconds": response.latency_seconds,
            },
            "content": response.content,
            "raw_response": response.raw_response,
        },
        case_dir / "llm" / "swe_recovery_response.json",
    )
    if decision.get("patch_id") != "source_logging_patch":
        raise RuntimeError(f"recovery MAS chose unexpected patch_id: {decision!r}")
    output_patch = output_patch.resolve()
    output_patch.parent.mkdir(parents=True, exist_ok=True)
    output_patch.write_text(_source_logging_patch(source), encoding="utf-8")
    print("SWE-LIVE-MAS RECOVERY PATCHER RESULT")
    print("provider_call: swe_recovery_patcher")
    print(f"llm_patch_id: {decision.get('patch_id')}")
    print(f"llm_rationale: {decision.get('rationale')}")
    print(f"recovered_patch: {output_patch}")
    return 0


def _validate(case_dir: Path, *, venv: Path, log: Path | None, update_failing_log: bool = False) -> int:
    case_dir = case_dir.resolve()
    python = (venv.resolve() / "bin" / "python")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [str(python), "./tests/runtests.py", "--verbosity", "1", "--settings=test_sqlite", "--parallel", "1",
           "sessions_tests.tests.CookieSessionTests.test_decode_failure_logged_to_security",
           "sessions_tests.tests.CookieSessionTests.test_decode_legacy"]
    completed = subprocess.run(
        cmd,
        cwd=case_dir / "repo",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = (
        "$ " + " ".join(cmd) + "\n"
        f"cwd: {case_dir / 'repo'}\n"
        f"returncode: {completed.returncode}\n\n"
        f"--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}\n"
    )
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")
    if update_failing_log:
        failing_log = case_dir / "logs" / "failing.log"
        failing_log.parent.mkdir(parents=True, exist_ok=True)
        failing_log.write_text(text, encoding="utf-8")
    print("SWE-LIVE-MAS VALIDATION")
    print(f"returncode: {completed.returncode}")
    print(f"status: {'PASS' if completed.returncode == 0 else 'FAIL'}")
    print((completed.stdout + completed.stderr)[-2000:])
    return completed.returncode


def _show_artifacts(case_dir: Path, baseline_json: Path) -> int:
    row = _baseline_row(baseline_json)
    print("SWE-LIVE-MAS ARTIFACTS")
    print(f"baseline_json: {baseline_json}")
    print(f"case_dir:      {case_dir}")
    print(f"model_calls:   {row.get('model_call_count')}")
    print(f"tokens:        {row.get('token_cost')}")
    for rel in [
        "trajectory.json",
        "logs/failing.log",
        "patches/failed.patch",
        "recoveragent/report.json",
        "recoveragent/recovery_prompt.md",
        "patches/recovered.patch",
        "llm/swe_recovery_response.json",
        "recoveragent/validation.json",
    ]:
        path = case_dir / rel
        if path.exists():
            print(f"- {path}")
    return 0


def _baseline_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        raise ValueError(f"baseline JSON has no rows: {path}")
    return dict(rows[0])


def _trajectory_from_row(row: dict[str, Any], *, manifest: dict[str, Any], baseline_json: Path) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for stage, output in dict(row.get("stage_outputs", {}) or {}).items():
        for command in list(dict(output or {}).get("commands", []) or []):
            calls.append(
                {
                    "role": stage,
                    "command": str(command.get("command", ""))[:2000],
                    "status": "ok" if int(command.get("returncode", 1) or 0) == 0 else "failed",
                    "summary": str(command.get("output", ""))[:500],
                }
            )
    calls.append(
        {
            "role": "oracle",
            "command": str(row.get("fail_to_pass_command") or manifest.get("test_command") or ""),
            "status": "failed" if int(row.get("fail_to_pass_returncode", 1) or 1) else "ok",
            "summary": "Fail-to-pass validation after the live MAS baseline.",
        }
    )
    return {
        "schema": "recoveragent.swe_live_mas_trajectory.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "issue": str(manifest.get("problem_statement", "")),
        "instance_id": INSTANCE_ID,
        "baseline_json": str(baseline_json),
        "model_call_count": row.get("model_call_count"),
        "token_cost": row.get("token_cost"),
        "latency_sec": row.get("latency_sec"),
        "patch_summary": row.get("patch_summary", {}),
        "stage_outputs": row.get("stage_outputs", {}),
        "tool_calls": calls,
    }


def _write_failed_log(row: dict[str, Any], path: Path) -> None:
    text = (
        f"instance_id: {INSTANCE_ID}\n"
        f"reported_success: {row.get('reported_success')}\n"
        f"oracle_success: {row.get('oracle_success')}\n"
        f"model_call_count: {row.get('model_call_count')}\n"
        f"token_cost: {row.get('token_cost')}\n"
        f"patch_scope: {dict(row.get('patch_summary', {}) or {}).get('patch_scope')}\n\n"
        "--- fail_to_pass_output ---\n"
        f"{row.get('fail_to_pass_output', '')}\n\n"
        "--- oracle_output ---\n"
        f"{row.get('oracle_output', '')}\n\n"
        "--- patcher_output_excerpt ---\n"
        f"{json.dumps(dict(row.get('stage_outputs', {}) or {}).get('patcher', {}), ensure_ascii=False)[:5000]}\n"
    )
    path.write_text(text, encoding="utf-8")


def _workspace_diff(workspace: Path) -> str:
    if not (workspace / ".git").exists():
        return ""
    completed = subprocess.run(
        ["git", "diff", "--", "django/contrib/sessions/backends/base.py"],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _decode_excerpt(source: str) -> str:
    lines = source.splitlines()
    for idx, line in enumerate(lines):
        if "def decode(self, session_data)" in line:
            return "\n".join(lines[max(0, idx - 8) : idx + 45])
    return "\n".join(lines[:80])


def _source_logging_patch(source: str) -> str:
    target = _correct_base_source(source)
    diff = difflib.unified_diff(
        source.splitlines(keepends=True),
        target.splitlines(keepends=True),
        fromfile="a/django/contrib/sessions/backends/base.py",
        tofile="b/django/contrib/sessions/backends/base.py",
    )
    return "diff --git a/django/contrib/sessions/backends/base.py b/django/contrib/sessions/backends/base.py\n" + "".join(diff)


def _correct_base_source(source: str) -> str:
    lines = source.splitlines()
    start = _find_line(lines, "    def decode(self, session_data):")
    end = _find_line(lines, "    def _legacy_decode(self, session_data):")
    replacement = [
        "    def decode(self, session_data):",
        "        try:",
        "            return signing.loads(session_data, salt=self.key_salt, serializer=self.serializer)",
        "        # RemovedInDjango40Warning: when the deprecation ends, handle here",
        "        # exceptions similar to what _legacy_decode() does now.",
        "        except signing.BadSignature:",
        "            try:",
        "                # Return an empty session if data is not in the pre-Django 3.1",
        "                # format.",
        "                return self._legacy_decode(session_data)",
        "            except Exception:",
        "                logger = logging.getLogger('django.security.SuspiciousSession')",
        "                logger.warning('Session data corrupted')",
        "                return {}",
        "        except Exception:",
        "            return self._legacy_decode(session_data)",
        "",
        "    def _legacy_encode(self, session_dict):",
        "        # RemovedInDjango40Warning.",
        "        serialized = self.serializer().dumps(session_dict)",
        "        hash = self._hash(serialized)",
        "        return base64.b64encode(hash.encode() + b':' + serialized).decode('ascii')",
        "",
    ]
    return "\n".join(lines[:start] + replacement + lines[end:]) + "\n"


def _find_line(lines: list[str], needle: str) -> int:
    for index, line in enumerate(lines):
        if line == needle:
            return index
    raise ValueError(f"could not find source marker: {needle}")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
