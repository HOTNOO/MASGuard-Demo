from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from recoveragent.cli import main
from recoveragent.diagnosis import diagnose
from recoveragent.evidence import build_evidence_graph, extract_evidence
from recoveragent.mas_plugin_demo import run_mas_plugin_demo
from recoveragent.planner import plan_recovery
from recoveragent.llm import DEFAULT_ENDPOINT, DEFAULT_MODEL
from recoveragent.online_provider import extract_json_object, load_provider_config, normalize_endpoint
from recoveragent.run_contract import layout_for


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo_case"
EXAMPLES = ROOT / "examples"
MAS_PLUGIN = EXAMPLES / "mas_plugin_case"


class RecoverAgentCoreTests(unittest.TestCase):
    def test_extract_evidence_from_demo_case(self) -> None:
        bundle = extract_evidence(
            trajectory_path=DEMO / "trajectory.json",
            repo_path=DEMO / "repo",
            log_path=DEMO / "logs" / "failing.log",
            patch_path=DEMO / "patches" / "failed.patch",
        )

        self.assertIn("widget.py", bundle.stack_trace_files)
        self.assertIn("tests/test_widget.py", bundle.touched_files)
        self.assertTrue(bundle.signals["has_test_failure"])
        self.assertTrue(bundle.signals["stack_trace_not_touched"])
        self.assertTrue(bundle.tool_calls)

    def test_diagnosis_for_demo_case(self) -> None:
        bundle = extract_evidence(
            trajectory_path=DEMO / "trajectory.json",
            repo_path=DEMO / "repo",
            log_path=DEMO / "logs" / "failing.log",
            patch_path=DEMO / "patches" / "failed.patch",
        )

        diagnosis = diagnose(bundle)

        self.assertEqual(diagnosis.failure_type, "fault_localization_failure")
        self.assertEqual(diagnosis.confidence, "medium")
        self.assertTrue(diagnosis.evidence)

    def test_recovery_plan_for_demo_case(self) -> None:
        bundle = extract_evidence(
            trajectory_path=DEMO / "trajectory.json",
            repo_path=DEMO / "repo",
            log_path=DEMO / "logs" / "failing.log",
            patch_path=DEMO / "patches" / "failed.patch",
        )
        diagnosis = diagnose(bundle)

        plan = plan_recovery(bundle, diagnosis)

        self.assertEqual(plan.action, "rollback-and-relocalize-from-evidence")
        self.assertTrue(plan.requires_human_or_agent_execution)
        self.assertIn("must produce and validate", plan.scope_note)

    def test_cli_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            html = Path(tmp) / "report.html"
            code = main(
                [
                    "analyze",
                    "--trajectory",
                    str(DEMO / "trajectory.json"),
                    "--repo",
                    str(DEMO / "repo"),
                    "--log",
                    str(DEMO / "logs" / "failing.log"),
                    "--patch",
                    str(DEMO / "patches" / "failed.patch"),
                    "--output",
                    str(output),
                    "--html",
                    str(html),
                ]
            )

            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "masguard.analysis_report.v2")
            self.assertEqual(report["tool"], "MASGuard")
            self.assertTrue(report["scope_note"]["deterministic_core"])
            self.assertEqual(report["diagnosis"]["failure_type"], "fault_localization_failure")
            self.assertIn("evidence_graph", report)
            self.assertIn("MASGuard Case Report", html.read_text(encoding="utf-8"))

    def test_evidence_graph_contains_patch_and_stack_nodes(self) -> None:
        bundle = extract_evidence(
            trajectory_path=DEMO / "trajectory.json",
            repo_path=DEMO / "repo",
            log_path=DEMO / "logs" / "failing.log",
            patch_path=DEMO / "patches" / "failed.patch",
        )
        graph = build_evidence_graph(bundle)
        labels = {node["label"] for node in graph["nodes"]}

        self.assertIn("widget.py", labels)
        self.assertIn("tests/test_widget.py", labels)
        self.assertGreaterEqual(len(graph["edges"]), 3)

    def test_suite_covers_multiple_failure_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite.json"
            markdown = Path(tmp) / "suite.md"
            code = main(
                [
                    "suite",
                    "--cases-dir",
                    str(EXAMPLES),
                    "--output",
                    str(output),
                    "--markdown",
                    str(markdown),
                    "--html",
                    str(Path(tmp) / "suite.html"),
                ]
            )

            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertGreaterEqual(report["case_count"], 4)
            self.assertIn("fault_localization_failure", report["diagnosis_counts"])
            self.assertIn("validation_target_failure", report["diagnosis_counts"])
            self.assertIn("environment_or_tool_failure", report["diagnosis_counts"])
            self.assertIn("context_drift_or_repeated_mistake", report["diagnosis_counts"])
            self.assertIn("Naive retry risk", markdown.read_text(encoding="utf-8"))

    def test_demo_command_writes_video_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "demo_outputs"
            code = main(
                [
                    "demo",
                    "--cases-dir",
                    str(EXAMPLES),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "case_report.json").exists())
            self.assertTrue((output_dir / "case_report.html").exists())
            self.assertTrue((output_dir / "suite_report.json").exists())
            self.assertTrue((output_dir / "comparison.md").exists())
            self.assertTrue((output_dir / "demo_dashboard.html").exists())
            self.assertIn("MASGuard Experiment Snapshot", (output_dir / "demo_dashboard.html").read_text(encoding="utf-8"))
            self.assertTrue((output_dir / "mas_plugin_run" / "mas_plugin_summary.json").exists())

    def test_backend_info_command(self) -> None:
        code = main(["backend-info"])

        self.assertEqual(code, 0)

    def test_cli_prompt_apply_patch_and_validate_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = tmp_path / "report.json"
            prompt = tmp_path / "recovery_prompt.md"
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "tests").mkdir()
            (workspace / "widget.py").write_text((MAS_PLUGIN / "repo" / "widget.py").read_text(encoding="utf-8"), encoding="utf-8")
            (workspace / "tests" / "test_widget.py").write_text(
                (MAS_PLUGIN / "repo" / "tests" / "test_widget.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            analyze_code = main(
                [
                    "analyze",
                    "--trajectory",
                    str(MAS_PLUGIN / "trajectory.json"),
                    "--repo",
                    str(workspace),
                    "--log",
                    str(MAS_PLUGIN / "logs" / "failing.log"),
                    "--patch",
                    str(MAS_PLUGIN / "patches" / "failed.patch"),
                    "--output",
                    str(report),
                ]
            )
            self.assertEqual(analyze_code, 0)

            prompt_code = main(["prompt", "--report", str(report), "--output", str(prompt)])
            self.assertEqual(prompt_code, 0)
            self.assertIn("rollback-and-relocalize", prompt.read_text(encoding="utf-8"))

            patch_code = main(
                [
                    "apply-patch",
                    "--repo",
                    str(workspace),
                    "--patch",
                    str(MAS_PLUGIN / "recovery_patches" / "recovered.patch"),
                ]
            )
            self.assertEqual(patch_code, 0)

            validate_code = main(
                [
                    "validate",
                    "--repo",
                    str(workspace),
                    "--command",
                    "python tests/test_widget.py",
                ]
            )
            self.assertEqual(validate_code, 0)

    def test_standard_run_dir_cli_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "failed_run"
            init_code = main(["init", "--run-dir", str(run_dir)])
            self.assertEqual(init_code, 0)

            layout = layout_for(run_dir)
            shutil.rmtree(layout.repo)
            shutil.copytree(MAS_PLUGIN / "repo", layout.repo)
            shutil.copy2(MAS_PLUGIN / "trajectory.json", layout.trajectory)
            shutil.copy2(MAS_PLUGIN / "logs" / "failing.log", layout.log)
            shutil.copy2(MAS_PLUGIN / "patches" / "failed.patch", layout.failed_patch)

            analyze_code = main(["analyze-run", "--run-dir", str(run_dir)])
            self.assertEqual(analyze_code, 0)
            self.assertTrue(layout.report_json.exists())
            self.assertTrue(layout.report_html.exists())
            self.assertTrue(layout.recovery_prompt.exists())

            prompt_code = main(["prompt", "--run-dir", str(run_dir)])
            self.assertEqual(prompt_code, 0)

            patch_code = main(
                [
                    "apply-patch",
                    "--run-dir",
                    str(run_dir),
                    "--patch",
                    str(MAS_PLUGIN / "recovery_patches" / "recovered.patch"),
                ]
            )
            self.assertEqual(patch_code, 0)

            validate_code = main(
                [
                    "validate",
                    "--run-dir",
                    str(run_dir),
                    "--command",
                    "python tests/test_widget.py",
                ]
            )
            self.assertEqual(validate_code, 0)
            self.assertTrue((layout.output_dir / "validation.json").exists())

    def test_no_effective_patch_diagnosis_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "repo"
            (repo / "pkg").mkdir(parents=True)
            (repo / "pkg" / "session.py").write_text("def decode(data):\n    return data\n", encoding="utf-8")
            trajectory = tmp_path / "trajectory.json"
            log = tmp_path / "failing.log"
            patch = tmp_path / "failed.patch"
            trajectory.write_text(
                json.dumps(
                    {
                        "issue": "decode() should log and return an empty session for corrupt data.",
                        "patch_summary": {"patch_scope": "no_diff"},
                        "tool_calls": [
                            {
                                "role": "patcher",
                                "command": "python patch.py",
                                "status": "ok",
                                "summary": "MASGUARD_STRICT_ABSTAIN_NO_EDIT; patch produced no git diff.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            log.write_text("FAILED tests/test_session.py::test_decode_failure_logged\n", encoding="utf-8")
            patch.write_text("MASGuard no effective patch: live MAS patcher produced no repository diff.\n", encoding="utf-8")

            bundle = extract_evidence(trajectory_path=trajectory, repo_path=repo, log_path=log, patch_path=patch)
            diagnosis = diagnose(bundle)
            plan = plan_recovery(bundle, diagnosis)

            self.assertTrue(bundle.signals["patcher_no_effective_diff"])
            self.assertEqual(diagnosis.failure_type, "no_effective_patch")
            self.assertEqual(diagnosis.responsible_stage, "patch")
            self.assertEqual(plan.action, "resume-patcher-with-source-contract-evidence")

    def test_mas_plugin_demo_recovers_to_passing_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_mas_plugin_demo(
                case_dir=MAS_PLUGIN,
                output_dir=Path(tmp) / "mas_plugin_run",
                use_llm=False,
                llm_model=DEFAULT_MODEL,
                llm_endpoint=DEFAULT_ENDPOINT,
                llm_timeout=10,
            )

            self.assertTrue(summary["success"])
            self.assertFalse(summary["flow"][0]["passed"])
            self.assertFalse(summary["flow"][1]["passed"])
            self.assertEqual(summary["flow"][2]["diagnosis"], "fault_localization_failure")
            self.assertTrue(summary["flow"][3]["passed"])

    def test_online_provider_config_parser_redacts_and_normalizes_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "api.local.md"
            config_path.write_text(
                "api_key=secret-value\n"
                "endpoint=https://example.test\n"
                "model=demo-model\n"
                'extra_body={"reasoning_split": true}\n',
                encoding="utf-8",
            )

            config = load_provider_config(api_path=config_path)

            self.assertEqual(config.model, "demo-model")
            self.assertEqual(normalize_endpoint(config.endpoint), "https://example.test/v1/chat/completions")
            self.assertTrue(config.extra_body["reasoning_split"])
            self.assertEqual(config.public_dict()["api_key"], "<redacted>")

    def test_extract_json_object_from_fenced_model_response(self) -> None:
        data = extract_json_object('```json\n{"patch_id": "source_contract_patch", "rationale": "ok"}\n```')

        self.assertEqual(data["patch_id"], "source_contract_patch")


if __name__ == "__main__":
    unittest.main()
