"""Command line interface for MASGuard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recoveragent.diagnosis import diagnose
from recoveragent.evidence import extract_evidence
from recoveragent.llm import DEFAULT_ENDPOINT, DEFAULT_MODEL, maybe_generate_llm_insight
from recoveragent.mas_plugin_demo import run_mas_plugin_demo
from recoveragent.patching import apply_simple_unified_patch, run_validation
from recoveragent.planner import plan_recovery
from recoveragent.prompt import write_recovery_prompt
from recoveragent.report import (
    build_report,
    build_suite_report,
    write_html_report,
    write_html_suite,
    write_markdown_suite,
    write_report,
)
from recoveragent.run_contract import init_run_dir, layout_for, print_layout, validate_run_dir


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES = PACKAGE_ROOT / "examples"
DEFAULT_SNAPSHOT = PACKAGE_ROOT / "assets" / "masguard_snapshot.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=Path(sys.argv[0]).name if argv is None else "masguard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create a standard failed-run directory for MAS integration.")
    init.add_argument("--run-dir", required=True, type=Path, help="Directory where the MAS will export failed-run artifacts.")
    init.add_argument("--force", action="store_true", help="Overwrite placeholder files if they already exist.")

    analyze = subparsers.add_parser("analyze", help="Analyze a failed repair-agent trajectory.")
    analyze.add_argument("--trajectory", required=True, type=Path, help="Path to trajectory JSON.")
    analyze.add_argument("--repo", required=True, type=Path, help="Path to repository snapshot.")
    analyze.add_argument("--log", required=True, type=Path, help="Path to failing test/build log.")
    analyze.add_argument("--patch", required=True, type=Path, help="Path to failed candidate patch.")
    analyze.add_argument("--output", required=True, type=Path, help="Path for JSON analysis report.")
    analyze.add_argument("--html", type=Path, help="Optional path for a self-contained HTML report.")
    _add_llm_args(analyze)

    analyze_run = subparsers.add_parser("analyze-run", help="Analyze a standard failed-run directory.")
    analyze_run.add_argument("--run-dir", required=True, type=Path, help="Directory created by masguard init.")
    analyze_run.add_argument("--no-prompt", action="store_true", help="Do not write recoveragent/recovery_prompt.md.")
    _add_llm_args(analyze_run)

    suite = subparsers.add_parser("suite", help="Analyze every demo case in a cases directory.")
    suite.add_argument("--cases-dir", required=True, type=Path, help="Directory containing demo case folders.")
    suite.add_argument("--output", required=True, type=Path, help="Path for JSON suite report.")
    suite.add_argument("--markdown", type=Path, help="Optional path for Markdown comparison report.")
    suite.add_argument("--html", type=Path, help="Optional path for a self-contained HTML suite report.")
    suite.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT, help="Optional MASGuard snapshot JSON.")
    _add_llm_args(suite)

    prompt = subparsers.add_parser("prompt", help="Export a recovery prompt for an upstream MAS/agent.")
    prompt.add_argument("--report", type=Path, help="MASGuard analysis report JSON.")
    prompt.add_argument("--run-dir", type=Path, help="Standard failed-run directory containing recoveragent/report.json.")
    prompt.add_argument("--output", type=Path, help="Path for recovery prompt Markdown.")

    apply_patch_cmd = subparsers.add_parser("apply-patch", help="Apply a MAS-produced recovery patch to a repository.")
    apply_patch_cmd.add_argument("--repo", type=Path, help="Repository/workspace path.")
    apply_patch_cmd.add_argument("--run-dir", type=Path, help="Standard failed-run directory; uses run-dir/repo.")
    apply_patch_cmd.add_argument("--patch", required=True, type=Path, help="Unified patch path.")
    apply_patch_cmd.add_argument("--output", type=Path, help="Optional JSON result path.")

    validate = subparsers.add_parser("validate", help="Run a validation command inside a repository.")
    validate.add_argument("--repo", type=Path, help="Repository/workspace path.")
    validate.add_argument("--run-dir", type=Path, help="Standard failed-run directory; uses run-dir/repo.")
    validate.add_argument(
        "--command",
        dest="validation_command",
        required=True,
        help='Validation command, e.g. "python tests/test_widget.py".',
    )
    validate.add_argument("--log", type=Path, help="Optional combined stdout/stderr log path.")
    validate.add_argument("--output", type=Path, help="Optional JSON result path.")
    validate.add_argument("--timeout", type=int, default=60, help="Validation timeout in seconds.")

    demo = subparsers.add_parser("demo", help="Replay packaged examples and generate demo artifacts.")
    demo.add_argument("--cases-dir", type=Path, default=DEFAULT_EXAMPLES, help="Directory containing demo cases.")
    demo.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "demo_outputs", help="Output directory.")
    demo.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT, help="MASGuard snapshot JSON.")
    _add_llm_args(demo)

    subparsers.add_parser(
        "backend-info",
        help="Show the MASGuard repository layers and how the SWE/MAS backend connects to the CLI.",
    )

    mas_demo = subparsers.add_parser(
        "mas-plugin-demo",
        help="Run a local MAS failure -> MASGuard sidecar -> recovered test pass demo.",
    )
    mas_demo.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_EXAMPLES / "mas_plugin_case",
        help="Case directory with repo, failed patch, and recovery patch.",
    )
    mas_demo.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "demo_outputs" / "mas_plugin_run",
        help="Output directory for the executable MAS-plugin run.",
    )
    _add_llm_args(mas_demo)

    args = parser.parse_args(argv)
    if args.command == "init":
        layout = init_run_dir(args.run_dir, force=args.force)
        print("Initialized MASGuard failed-run directory.")
        print(print_layout(layout))
        print("\nNext command:")
        print(f"  masguard analyze-run --run-dir {layout.run_dir}")
        return 0
    if args.command == "analyze":
        report = _analyze_case(
            trajectory_path=args.trajectory,
            repo_path=args.repo,
            log_path=args.log,
            patch_path=args.patch,
            use_llm=args.llm,
            llm_model=args.llm_model,
            llm_endpoint=args.llm_endpoint,
            llm_timeout=args.llm_timeout,
        )
        write_report(report, args.output)
        if args.html:
            write_html_report(report, args.html)
        print(f"Wrote MASGuard analysis report to {args.output}")
        if args.html:
            print(f"Wrote MASGuard HTML report to {args.html}")
        print(f"Diagnosis: {report['diagnosis']['failure_type']} ({report['diagnosis']['confidence']})")
        print(f"Recovery action: {report['recovery_plan']['action']}")
        if args.llm:
            print(f"Optional LLM called: {report['llm_insight']['provider_called']}")
        return 0
    if args.command == "analyze-run":
        layout = validate_run_dir(args.run_dir)
        report = _analyze_case(
            trajectory_path=layout.trajectory,
            repo_path=layout.repo,
            log_path=layout.log,
            patch_path=layout.failed_patch,
            use_llm=args.llm,
            llm_model=args.llm_model,
            llm_endpoint=args.llm_endpoint,
            llm_timeout=args.llm_timeout,
        )
        write_report(report, layout.report_json)
        write_html_report(report, layout.report_html)
        prompt_text = ""
        if not args.no_prompt:
            prompt_text = write_recovery_prompt(layout.report_json, layout.recovery_prompt)
        print(f"Wrote MASGuard analysis report to {layout.report_json}")
        print(f"Wrote MASGuard HTML report to {layout.report_html}")
        if not args.no_prompt:
            print(f"Wrote MASGuard recovery prompt to {layout.recovery_prompt}")
            print(f"Prompt lines: {len(prompt_text.splitlines())}")
        print(f"Diagnosis: {report['diagnosis']['failure_type']} ({report['diagnosis']['confidence']})")
        print(f"Recovery action: {report['recovery_plan']['action']}")
        print("\nGive this prompt back to the same MAS patcher:")
        print(f"  {layout.recovery_prompt}")
        return 0
    if args.command == "suite":
        suite_report = _build_suite_from_cases(args)
        snapshot = _load_snapshot(args.snapshot)
        write_report(suite_report, args.output)
        if args.markdown:
            write_markdown_suite(suite_report, args.markdown)
        if args.html:
            write_html_suite(suite_report, args.html, snapshot=snapshot)
        print(f"Wrote MASGuard suite report to {args.output}")
        if args.markdown:
            print(f"Wrote MASGuard suite markdown to {args.markdown}")
        if args.html:
            print(f"Wrote MASGuard suite HTML to {args.html}")
        print(f"Cases analyzed: {suite_report['case_count']}")
        for case in suite_report["cases"]:
            print(
                " - {case_id}: {diagnosis} -> {action}".format(
                    case_id=case["case_id"],
                    diagnosis=case["diagnosis"],
                    action=case["recovery_action"],
                )
            )
        return 0
    if args.command == "prompt":
        report_path, output_path = _resolve_prompt_paths(args)
        prompt_text = write_recovery_prompt(report_path, output_path)
        print(f"Wrote MASGuard recovery prompt to {output_path}")
        print(f"Prompt lines: {len(prompt_text.splitlines())}")
        return 0
    if args.command == "apply-patch":
        repo_path = _resolve_repo(args)
        output_path = args.output
        if output_path is None and args.run_dir:
            output_path = layout_for(args.run_dir).output_dir / "patch_apply.json"
        result = apply_simple_unified_patch(repo_path, args.patch)
        if output_path:
            _write_json(result.to_dict(), output_path)
        print(f"Patch applied: {result.applied}")
        print(f"Touched file: {result.touched_file}")
        if result.reason:
            print(f"Reason: {result.reason}")
        return 0 if result.applied else 1
    if args.command == "validate":
        repo_path = _resolve_repo(args)
        log_path = args.log
        output_path = args.output
        if args.run_dir:
            layout = layout_for(args.run_dir)
            if log_path is None:
                log_path = layout.output_dir / "validation.log"
            if output_path is None:
                output_path = layout.output_dir / "validation.json"
        result = run_validation(repo_path, args.validation_command, timeout=args.timeout)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                f"$ {args.validation_command}\n"
                f"cwd: {repo_path}\n"
                f"returncode: {result.returncode}\n\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}\n",
                encoding="utf-8",
            )
        if output_path:
            _write_json(result.to_dict(), output_path)
        print(f"Validation passed: {result.passed}")
        print(f"Return code: {result.returncode}")
        return result.returncode
    if args.command == "demo":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        single_json = args.output_dir / "case_report.json"
        single_html = args.output_dir / "case_report.html"
        suite_json = args.output_dir / "suite_report.json"
        suite_md = args.output_dir / "comparison.md"
        suite_html = args.output_dir / "demo_dashboard.html"
        snapshot_out = args.output_dir / "masguard_snapshot.json"
        mas_plugin_out = args.output_dir / "mas_plugin_run"
        demo_case = args.cases_dir / "demo_case"

        case_report = _analyze_case(
            trajectory_path=demo_case / "trajectory.json",
            repo_path=demo_case / "repo",
            log_path=demo_case / "logs" / "failing.log",
            patch_path=demo_case / "patches" / "failed.patch",
            use_llm=args.llm,
            llm_model=args.llm_model,
            llm_endpoint=args.llm_endpoint,
            llm_timeout=args.llm_timeout,
        )
        write_report(case_report, single_json)
        write_html_report(case_report, single_html)

        suite_report = _build_suite_from_cases(args)
        snapshot = _load_snapshot(args.snapshot)
        write_report(suite_report, suite_json)
        write_markdown_suite(suite_report, suite_md)
        write_html_suite(suite_report, suite_html, snapshot=snapshot)
        _write_json(snapshot, snapshot_out)
        mas_summary = run_mas_plugin_demo(
            case_dir=args.cases_dir / "mas_plugin_case",
            output_dir=mas_plugin_out,
            use_llm=args.llm,
            llm_model=args.llm_model,
            llm_endpoint=args.llm_endpoint,
            llm_timeout=args.llm_timeout,
        )

        print(f"MASGuard demo artifacts written to {args.output_dir}")
        print(f" - single-case JSON: {single_json}")
        print(f" - single-case HTML: {single_html}")
        print(f" - suite JSON: {suite_json}")
        print(f" - suite Markdown: {suite_md}")
        print(f" - demo dashboard: {suite_html}")
        print(f" - MASGuard snapshot: {snapshot_out}")
        print(f" - MAS plugin run: {mas_plugin_out}")
        print(f"Cases analyzed: {suite_report['case_count']}")
        print(f"Optional LLM called for single case: {case_report['llm_insight']['provider_called']}")
        print(f"MAS plugin recovered success: {mas_summary['success']}")
        return 0
    if args.command == "backend-info":
        print("MASGuard architecture")
        print("")
        print("1. CLI sidecar: src/recoveragent")
        print("   Installed command: masguard")
        print("   Role: failed-run layout, evidence extraction, diagnosis, recovery prompt,")
        print("         patch accounting, validation logging, and deterministic reports.")
        print("")
        print("2. SWE/MAS backend: bcmr_swe")
        print("   Used by: examples/user_mas/swe_live_mas.py run-baseline")
        print("   Role: real SWE runner, online MAS baseline, recovery driver, and")
        print("         experiment-record generation for the 69-instance data package.")
        print("")
        print("3. MAS substrate: swe_mas")
        print("   Role: reusable multi-agent repair components and historical MAS substrate.")
        print("")
        print("Typical flow")
        print("   existing MAS or bcmr_swe MAS fails")
        print("   -> export repo, trajectory.json, logs/failing.log, patches/failed.patch")
        print("   -> masguard analyze-run --run-dir <failed_run>")
        print("   -> same MAS patcher consumes recoveragent/recovery_prompt.md")
        print("   -> masguard apply-patch and masguard validate record the recovered result")
        return 0
    if args.command == "mas-plugin-demo":
        summary = run_mas_plugin_demo(
            case_dir=args.case_dir,
            output_dir=args.output_dir,
            use_llm=args.llm,
            llm_model=args.llm_model,
            llm_endpoint=args.llm_endpoint,
            llm_timeout=args.llm_timeout,
        )
        print(f"MASGuard MAS-plugin demo written to {args.output_dir}")
        for row in summary["flow"]:
            if row["stage"] == "masguard_plugin":
                print(f" - {row['stage']}: {row['diagnosis']} -> {row['recovery_action']}")
            else:
                print(f" - {row['stage']}: returncode={row['returncode']} passed={row['passed']}")
        print(f"Overall recovered success: {summary['success']}")
        return 0
    return 2


def _analyze_case(
    *,
    trajectory_path: Path,
    repo_path: Path,
    log_path: Path,
    patch_path: Path,
    use_llm: bool = False,
    llm_model: str = DEFAULT_MODEL,
    llm_endpoint: str = DEFAULT_ENDPOINT,
    llm_timeout: int = 45,
) -> dict:
    bundle = extract_evidence(
        trajectory_path=trajectory_path,
        repo_path=repo_path,
        log_path=log_path,
        patch_path=patch_path,
    )
    diagnosis = diagnose(bundle)
    plan = plan_recovery(bundle, diagnosis)
    insight = maybe_generate_llm_insight(
        bundle=bundle,
        diagnosis=diagnosis,
        plan=plan,
        enabled=use_llm,
        model=llm_model,
        endpoint=llm_endpoint,
        timeout=llm_timeout,
    )
    return build_report(bundle, diagnosis, plan, insight)


def _build_suite_from_cases(args: argparse.Namespace) -> dict:
    case_reports = []
    for case_dir in _case_dirs(args.cases_dir):
        report = _analyze_case(
            trajectory_path=case_dir / "trajectory.json",
            repo_path=case_dir / "repo",
            log_path=case_dir / "logs" / "failing.log",
            patch_path=case_dir / "patches" / "failed.patch",
            use_llm=args.llm,
            llm_model=args.llm_model,
            llm_endpoint=args.llm_endpoint,
            llm_timeout=args.llm_timeout,
        )
        report["case_id"] = case_dir.name
        case_reports.append(report)
    return build_suite_report(case_reports)


def _case_dirs(cases_dir: Path) -> list[Path]:
    if not cases_dir.exists():
        raise FileNotFoundError(cases_dir)
    case_dirs = [
        path
        for path in sorted(cases_dir.iterdir())
        if path.is_dir()
        and (path / "trajectory.json").exists()
        and (path / "repo").exists()
        and (path / "logs" / "failing.log").exists()
        and (path / "patches" / "failed.patch").exists()
    ]
    if not case_dirs:
        raise ValueError(f"no MASGuard demo cases found under {cases_dir}")
    return case_dirs


def _load_snapshot(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_prompt_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_dir:
        layout = layout_for(args.run_dir)
        report_path = args.report or layout.report_json
        output_path = args.output or layout.recovery_prompt
    else:
        if not args.report:
            raise ValueError("prompt requires --report or --run-dir")
        if not args.output:
            raise ValueError("prompt requires --output unless --run-dir is provided")
        report_path = args.report
        output_path = args.output
    return report_path, output_path


def _resolve_repo(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return layout_for(args.run_dir).repo
    if args.repo:
        return args.repo
    raise ValueError(f"{args.command} requires --repo or --run-dir")


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm", action="store_true", help="Call an optional OpenAI-compatible API for explanation text.")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL, help="Model for optional LLM explanation.")
    parser.add_argument("--llm-endpoint", default=DEFAULT_ENDPOINT, help="OpenAI-compatible chat completions endpoint.")
    parser.add_argument("--llm-timeout", type=int, default=45, help="Timeout for optional LLM explanation.")


if __name__ == "__main__":
    raise SystemExit(main())
