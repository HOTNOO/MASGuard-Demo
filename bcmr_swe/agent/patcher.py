"""Patcher adapter built from existing planner + implementer agents."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import re
from typing import Any, Protocol

from bcmr_swe.recovery.patch_contract import parse_patch_contract_prompt
from bcmr_swe.recovery.patch_intent import parse_patch_intent_prompt
from swe_mas.utils.command_classification import (
    looks_like_validation_command,
    looks_like_write_command,
)
from swe_mas.utils.path_filters import (
    canonical_source_paths,
    classify_changed_files,
    existing_repo_source_paths,
    normalize_repo_path,
    parse_unified_diff_paths,
)


logger = logging.getLogger(__name__)


class PatcherProtocol(Protocol):
    def patch(
        self,
        issue: str,
        workspace: str,
        *,
        located_files: str,
        recovery_context: str = "",
        escalation_level: int = 0,
    ) -> dict[str, Any]:
        ...


@dataclass
class PlannerPatcherAdapter:
    """Compose swe_mas planner and implementer into a BCMR patcher stage."""

    model: Any
    executor: Any
    planner_model: Any | None = None
    implementer_model: Any | None = None
    recorder: Any | None = None
    session_id: str | None = None
    max_plan_iterations: int = 4
    max_patch_iterations: int = 8
    enable_recovery_fresh_diff_audit: bool = False
    recovery_execution_profile: str = "normal"
    enable_recovery_retry: bool = True

    def _compact_located_files(self, located_files: str) -> str:
        raw = str(located_files or "").strip()
        if not raw:
            return ""
        workspace = str(getattr(getattr(self.executor, "config", None), "cwd", "") or "")
        raw_paths = parse_unified_diff_paths(raw)
        raw_paths.extend(re.findall(r"[A-Za-z0-9_./-]+\.pyi?", raw))
        canonical = existing_repo_source_paths(raw_paths, workspace) or canonical_source_paths(raw_paths)
        if canonical:
            return "\n".join(f"- {path}" for path in canonical[:10])
        return raw[:1200]

    def _compose_recovery_issue(
        self,
        *,
        issue: str,
        located_files: str,
        recovery_context: str,
        previous_failure_mode: str = "",
        previous_attempt_feedback: str = "",
    ) -> str:
        issue_with_context = issue.strip()
        if not recovery_context:
            return issue_with_context
        compact_located_files = self._compact_located_files(located_files)

        contract_lines = [
            "[RECOVERY CONTRACT]",
            "当前是在失败后的恢复场景中继续修复，不是从头重新解题。",
            "优先把 located_files 视为规范源码目标；若仓库中存在重复副本或镜像目录，默认不要修改那些副本。",
            "除非计划明确要求，不要把 build/、dist/、target/、generated/、site-packages/、__pycache__ 等目录当主编辑目标。",
            "如果没有形成针对规范源码目标的有效 diff，不要把任务视为完成。",
            f"Recovery execution profile: {self.recovery_execution_profile or 'normal'}",
        ]
        if self._is_masguard_source_edit_contract(recovery_context):
            contract_lines.extend(
                [
                    "MASGuard candidate admission rule: a reply without a fresh canonical source diff is a rejected candidate, not a partial success.",
                    "When the evidence is uncertain, spend at most one bounded source-target probe, then either make a minimal source edit or emit an explicit abstention reason.",
                    "A no-op script, stale diff, test-only edit, or terminal completion marker without a fresh source diff must not be submitted as the final answer.",
                ]
            )
        if compact_located_files:
            contract_lines.append(f"Canonical source targets:\n{compact_located_files}")
        if previous_failure_mode:
            contract_lines.append(f"Previous ineffective attempt: {previous_failure_mode}")
        if previous_attempt_feedback:
            contract_lines.append(previous_attempt_feedback)
        if self.recovery_execution_profile == "focused_source_repair":
            cfr_contract = self._extract_cfr_replay_contract(recovery_context)
            preferred_source_paths = cfr_contract.get("preferred_source_paths", [])
            read_only_paths = cfr_contract.get("read_only_evidence_paths", [])
            failing_tests = cfr_contract.get("failing_tests", [])
            negative_facts = cfr_contract.get("negative_facts", [])
            if preferred_source_paths:
                contract_lines.append(
                    "CFR focused source-repair targets:\n"
                    + "\n".join(f"- {path}" for path in preferred_source_paths[:8])
                )
                contract_lines.append(
                    "Hard recovery rule: make the next source diff inside one CFR focused source-repair target unless the file does not exist or focused evidence proves a different source file is required."
                )
            if read_only_paths:
                contract_lines.append(
                    "CFR read-only evidence paths; inspect but do not edit:\n"
                    + "\n".join(f"- {path}" for path in read_only_paths[:8])
                )
            if failing_tests:
                contract_lines.append(
                    "CFR focused failing tests to run or reason from:\n"
                    + "\n".join(f"- {test}" for test in failing_tests[:8])
                )
            if negative_facts:
                contract_lines.append(
                    "CFR negative facts from the polluted replay; do not repeat these assumptions:\n"
                    + "\n".join(f"- {fact}" for fact in negative_facts[:8])
                )
            contract_lines.extend(
                [
                    "CFR focused source-repair profile: do not edit tests, generated files, copied build outputs, or downstream evidence files.",
                    "CFR focused source-repair profile: after a source edit, run the most focused relevant validation available; if it fails, revise the same source diff before widening scope.",
                ]
            )
        precondition_match = re.search(r"Replay precondition:\s*([A-Za-z0-9_./-]+)", recovery_context)
        replay_precondition = precondition_match.group(1).strip() if precondition_match else ""
        if replay_precondition == "evidence_bounded_scope_expand":
            contract_lines.extend(
                [
                    "CAR evidence-bounded scope expansion: this is not an open-ended repository search.",
                    "Use the latest failure evidence to relocalize the smallest adjacent source boundary, then produce a fresh source diff and focused validation.",
                    "If no source edit is possible, the plan must state the focused evidence that proves why.",
                ]
            )
        parc_contract = parse_patch_contract_prompt(recovery_context)
        patch_intent = parse_patch_intent_prompt(recovery_context)
        if parc_contract:
            suspect_paths = [
                normalize_repo_path(str(item))
                for item in list(parc_contract.get("suspect_paths", []) or [])[:8]
                if normalize_repo_path(str(item))
            ]
            if suspect_paths:
                contract_lines.append(
                    "PARC suspect source boundary:\n"
                    + "\n".join(f"- {path}" for path in suspect_paths)
                )
                contract_lines.append(
                    "Hard recovery rule: make the fresh source diff inside this boundary unless you first refresh localization with evidence."
                )
            max_files = int(parc_contract.get("max_fresh_source_files", 3) or 3)
            contract_lines.append(
                f"Hard recovery rule: keep the fresh source diff local, normally one source file and never more than {max_files} source files."
            )
            forbidden = ", ".join(list(parc_contract.get("forbidden_path_classes", []) or [])[:4])
            if forbidden:
                contract_lines.append(f"Hard recovery rule: do not edit forbidden path classes: {forbidden}.")
        if patch_intent:
            selected_action = str(patch_intent.get("selected_action", "") or "")
            latest_revision = str(patch_intent.get("latest_revision_type", "") or "")
            target_paths = [
                normalize_repo_path(str(item))
                for item in list(patch_intent.get("target_paths", []) or [])[:8]
                if normalize_repo_path(str(item))
            ]
            candidate_paths = [
                normalize_repo_path(str(item))
                for item in list(patch_intent.get("candidate_source_paths", []) or [])[:8]
                if normalize_repo_path(str(item))
            ]
            avoid_paths = [
                normalize_repo_path(str(item))
                for item in list(patch_intent.get("avoid_target_paths", []) or [])[:8]
                if normalize_repo_path(str(item))
            ]
            directives = {
                str(item).strip().upper()
                for item in list(patch_intent.get("directives", []) or [])
                if str(item).strip()
            }
            contract_lines.append(
                f"CAR action intent: selected_action={selected_action or 'UNKNOWN'}, belief_revision={latest_revision or 'UNKNOWN'}."
            )
            if target_paths:
                contract_lines.append(
                    "CAR intended source targets:\n"
                    + "\n".join(f"- {path}" for path in target_paths)
                )
            if candidate_paths:
                contract_lines.append(
                    "CAR preserved source candidate paths:\n"
                    + "\n".join(f"- {path}" for path in candidate_paths)
                )
            if avoid_paths:
                contract_lines.append(
                    "CAR revoked or stale target paths; do not return to them without new evidence:\n"
                    + "\n".join(f"- {path}" for path in avoid_paths)
                )
            if bool(patch_intent.get("require_target_touch", False)) and target_paths:
                contract_lines.append(
                    "Hard recovery rule: the next fresh source diff must touch one CAR intended source target, "
                    "unless you first run evidence that proves the target must change."
                )
            if bool(patch_intent.get("require_evidence_before_retarget", False)):
                contract_lines.append(
                    "Hard recovery rule: a previous belief was revoked or made no progress; do not reuse or retarget it without focused evidence."
                )
            if bool(patch_intent.get("preserve_candidate_source", False)) or "PRESERVE_AND_REFINE_SOURCE_CANDIDATE" in directives:
                contract_lines.append(
                    "CAR candidate-preserving mode: keep the useful parts of the existing source candidate and make the smallest evidence-backed correction."
                )
            if selected_action in {"LOCAL_REPAIR", "REPAIR_LOCAL"}:
                contract_lines.append(
                    "CAR local repair semantics: edit the intended source boundary directly, run focused validation, and avoid broad search unless evidence invalidates the boundary."
                )
            elif selected_action in {"SCOPE_EXPAND", "EXPAND_SCOPE"}:
                contract_lines.append(
                    "CAR scope expansion semantics: refresh localization first, then patch only the newly evidence-backed canonical source target."
                )
            max_files = int(patch_intent.get("max_fresh_source_files", 3) or 3)
            contract_lines.append(
                f"Hard recovery rule: CAR patch intent allows at most {max_files} fresh source files in this replay."
            )
        contract_lines.append(recovery_context.strip())
        if "Candidate-preserving refine:" in recovery_context:
            contract_lines.extend(
                [
                    "候选补丁精修模式：不要从头大范围探索；先保留已有源码候选中仍然合理的部分。",
                    "用最新失败证据定位最小修正点；若 focused test 仍失败，优先修正源码候选而不是新增测试文件。",
                ]
            )
        contract = "\n".join(line for line in contract_lines if line)
        return f"{contract}\n\n---\n\n{issue_with_context}"

    def _extract_cfr_replay_contract(self, recovery_context: str) -> dict[str, list[str]]:
        if "[CFR replay contract]" not in str(recovery_context or ""):
            return {}
        parsed: dict[str, list[str]] = {}
        wanted = {
            "preferred_source_paths",
            "read_only_evidence_paths",
            "failing_tests",
            "negative_facts",
        }
        for raw_line in str(recovery_context or "").splitlines():
            line = raw_line.strip()
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            if key not in wanted:
                continue
            values = [
                item.strip()
                for item in raw_value.split(",")
                if item.strip() and item.strip().lower() != "none"
            ]
            if values:
                parsed[key] = values
        return parsed

    def _needs_recovery_retry(self, impl_result: dict[str, Any], recovery_context: str) -> bool:
        if not recovery_context.strip():
            return False
        summary = dict(impl_result.get("patch_summary") or {})
        failure_mode = str(summary.get("failure_mode", "") or "")
        retryable = {
            "no_effective_patch",
            "no_fresh_patch",
            "no_fresh_source_patch",
            "generated_only_patch",
        }
        if self.recovery_execution_profile == "focused_source_repair":
            retryable.update(
                {
                    "test_only_patch",
                    "mixed_patch",
                    "source_mixed",
                    "non_source_only",
                }
            )
        return failure_mode in retryable

    def _recovery_retry_feedback(self, impl_result: dict[str, Any], recovery_context: str) -> str:
        if not recovery_context.strip():
            return ""
        summary = dict(impl_result.get("patch_summary") or {})
        failure_mode = str(summary.get("failure_mode", "") or "").strip()
        if not failure_mode:
            return ""
        lines = [
            "[RECOVERY RETRY FEEDBACK]",
            f"Previous recovery attempt failed with failure_mode={failure_mode}.",
        ]
        fresh_files = [
            normalize_repo_path(str(path))
            for path in list(summary.get("fresh_changed_files", []) or [])
            if normalize_repo_path(str(path))
        ]
        fresh_classes = dict(summary.get("fresh_changed_file_classes", {}) or {})
        fresh_source_files = [
            normalize_repo_path(str(path))
            for path in list(fresh_classes.get("source_files", []) or [])
            if normalize_repo_path(str(path))
        ]
        changed_files = [
            normalize_repo_path(str(path))
            for path in list(summary.get("changed_files", []) or [])
            if normalize_repo_path(str(path))
        ]
        audit = dict(summary.get("recovery_fresh_diff_audit", {}) or {})
        replay_diff_changed = bool(audit.get("replay_diff_changed", False))
        if audit:
            lines.append(
                "Fresh-diff audit: the replay is judged against the pre-replay diff baseline, not against the outer workspace diff."
            )
            if not replay_diff_changed:
                lines.append(
                    "The final diff digest did not change during the replay; do not repeat or re-emit the pre-existing patch."
                )
        if changed_files and not fresh_files:
            lines.append(
                "Observed changed files are stale with respect to this replay: "
                + ", ".join(changed_files[:6])
            )
        elif fresh_files:
            lines.append("Fresh changed files from the failed attempt: " + ", ".join(fresh_files[:6]))
        if not fresh_source_files:
            lines.append("The next attempt must create a baseline-fresh canonical source diff.")
        else:
            lines.append("Fresh source files from the failed attempt: " + ", ".join(fresh_source_files[:6]))
        commands = [dict(item) for item in list(impl_result.get("commands", []) or []) if isinstance(item, dict)]
        write_commands = [
            str(item.get("command", "") or "")
            for item in commands
            if looks_like_write_command(str(item.get("command", "") or ""))
        ]
        validation_commands = [
            str(item.get("command", "") or "")
            for item in commands
            if looks_like_validation_command(str(item.get("command", "") or ""))
        ]
        if write_commands and not fresh_source_files:
            lines.append(
                "The previous attempt issued write commands but left no baseline-fresh source diff; revise the actual source content, not just the command sequence."
            )
        if not validation_commands:
            lines.append("After the source edit, run the focused validation command when available.")
        patch_intent = parse_patch_intent_prompt(recovery_context)
        if patch_intent:
            intended_paths = [
                normalize_repo_path(str(item))
                for item in (
                    list(patch_intent.get("candidate_source_paths", []) or [])
                    or list(patch_intent.get("target_paths", []) or [])
                )[:8]
                if normalize_repo_path(str(item))
            ]
            if intended_paths:
                lines.append(
                    "Retry target boundary: make the fresh source diff touch one of "
                    + ", ".join(intended_paths[:6])
                    + "."
                )
        lines.append(
            "Do not finish the retry unless the final git diff contains a new source change relative to the replay baseline and validation has been attempted."
        )
        return "\n".join(lines)

    def _is_post_evidence_source_repair_intent(self, recovery_context: str) -> bool:
        patch_intent = parse_patch_intent_prompt(recovery_context)
        if not patch_intent:
            return False
        selected_action = str(patch_intent.get("selected_action", "") or "")
        if selected_action not in {"LOCAL_REPAIR", "REPAIR_LOCAL"}:
            return False
        directives = {
            str(item or "").strip().upper()
            for item in list(patch_intent.get("directives", []) or [])
            if str(item or "").strip()
        }
        return bool(
            "POST_EVIDENCE_SOURCE_REPAIR" in directives
            or "DO_NOT_SPEND_REPLAY_ON_READONLY_DIAGNOSIS" in directives
        )

    def _patch_intent_plan_audit(
        self,
        plan: str,
        recovery_context: str,
    ) -> dict[str, Any]:
        patch_intent = parse_patch_intent_prompt(recovery_context)
        if not patch_intent:
            return {"intent_present": False, "satisfied": True, "flags": []}
        plan_text = str(plan or "")
        plan_lower = plan_text.lower()
        target_paths = [
            normalize_repo_path(str(item))
            for item in list(patch_intent.get("target_paths", []) or [])[:8]
            if normalize_repo_path(str(item))
        ]
        candidate_paths = [
            normalize_repo_path(str(item))
            for item in list(patch_intent.get("candidate_source_paths", []) or [])[:8]
            if normalize_repo_path(str(item))
        ]
        intended_paths = candidate_paths if bool(patch_intent.get("preserve_candidate_source", False)) else target_paths
        selected_action = str(patch_intent.get("selected_action", "") or "")
        directives = {
            str(item).strip()
            for item in list(patch_intent.get("directives", []) or [])
            if str(item).strip()
        }
        flags: list[str] = []
        overlap = [
            path
            for path in intended_paths
            if path and (path in plan_text or path.lower() in plan_lower)
        ]
        require_target_touch = bool(patch_intent.get("require_target_touch", False))
        if require_target_touch and intended_paths and not overlap:
            flags.append("plan_missing_intended_source_target")
        if bool(patch_intent.get("require_fresh_source_diff", True)):
            source_intent_terms = {
                "source",
                "diff",
                "patch",
                "modify",
                "edit",
                "源码",
                "源文件",
                "修改",
                "补丁",
            }
            if not any(term in plan_lower or term in plan_text for term in source_intent_terms):
                flags.append("plan_missing_fresh_source_diff_intent")
        validation_terms = {
            "pytest",
            "test",
            "验证",
            "focused",
            "fail_to_pass",
            "reproduce",
            "运行",
        }
        if not any(term in plan_lower or term in plan_text for term in validation_terms):
            flags.append("plan_missing_focused_validation")
        if selected_action in {"SCOPE_EXPAND", "EXPAND_SCOPE"}:
            evidence_terms = {
                "locator",
                "localization",
                "relocalize",
                "broader",
                "evidence",
                "定位",
                "证据",
                "扩大",
                "范围",
            }
            if not any(term in plan_lower or term in plan_text for term in evidence_terms):
                flags.append("plan_missing_scope_expansion_evidence")
            patch_terms = {"patch", "modify", "edit", "source", "diff", "修改", "源码", "源文件", "补丁"}
            if not any(term in plan_lower or term in plan_text for term in patch_terms):
                flags.append("plan_missing_scope_expansion_patch_target")
        if "PRESERVE_AND_REFINE_SOURCE_CANDIDATE" in directives and candidate_paths:
            candidate_overlap = [
                path
                for path in candidate_paths
                if path and (path in plan_text or path.lower() in plan_lower)
            ]
            if not candidate_overlap:
                flags.append("plan_missing_preserved_candidate")
        return {
            "intent_present": True,
            "selected_action": selected_action,
            "target_paths": target_paths,
            "candidate_source_paths": candidate_paths,
            "satisfied": not flags,
            "flags": flags,
            "target_overlap": overlap,
        }

    def _patch_intent_plan_feedback(
        self,
        audit: dict[str, Any],
    ) -> str:
        if not audit or not audit.get("intent_present") or audit.get("satisfied"):
            return ""
        target_paths = [
            str(item)
            for item in list(audit.get("target_paths", []) or [])
            if str(item).strip()
        ]
        candidate_paths = [
            str(item)
            for item in list(audit.get("candidate_source_paths", []) or [])
            if str(item).strip()
        ]
        lines = [
            "CAR Patch Intent plan check failed.",
            "Rewrite the plan so the implementer receives an executable recovery plan, not a generic repair sketch.",
        ]
        if target_paths:
            lines.append("The plan must name at least one intended source target:")
            lines.extend(f"- {path}" for path in target_paths[:6])
        if candidate_paths:
            lines.append("If refining an existing source candidate, the plan must preserve and minimally adjust:")
            lines.extend(f"- {path}" for path in candidate_paths[:6])
        lines.append("The plan must explicitly say how it will produce a fresh source diff and how it will run focused validation.")
        if "plan_missing_scope_expansion_evidence" in set(audit.get("flags", []) or []):
            lines.append("For SCOPE_EXPAND, first state what new localization evidence justifies the new target.")
        lines.append("Do not propose test-only edits or broad repository rewrites.")
        return "\n".join(lines)

    def _prepend_patch_intent_plan_guard(
        self,
        plan: str,
        audit: dict[str, Any],
        recovery_context: str = "",
    ) -> str:
        feedback = self._patch_intent_plan_feedback(audit)
        guarded_plan = str(plan or "")
        if feedback:
            guarded_plan = f"{feedback}\n\n--- original plan ---\n{guarded_plan}"
        return self._prepend_strict_source_edit_script_guard(guarded_plan, recovery_context)

    @staticmethod
    def _is_masguard_source_edit_contract(recovery_context: str) -> bool:
        return "[MASGUARD SOURCE EDIT CONTRACT]" in str(recovery_context or "")

    @staticmethod
    def _extract_masguard_source_edit_targets(recovery_context: str) -> list[str]:
        return list(PlannerPatcherAdapter._extract_masguard_source_edit_requirements(recovery_context).get("targets", []))

    @staticmethod
    def _extract_masguard_source_edit_requirements(recovery_context: str) -> dict[str, Any]:
        text = str(recovery_context or "")
        targets: list[str] = []
        required_source_files: list[str] = []
        max_source_files = 1

        def add(value: str) -> None:
            path = normalize_repo_path(value)
            if path and path not in targets:
                targets.append(path)

        def add_required(value: str) -> None:
            path = normalize_repo_path(value)
            if path and path not in required_source_files:
                required_source_files.append(path)

        payload_match = re.search(
            r"\[MASGUARD SOURCE EDIT CONTRACT\](.*?)\[/MASGUARD SOURCE EDIT CONTRACT\]",
            text,
            flags=re.DOTALL,
        )
        payload_text = payload_match.group(1) if payload_match else text
        json_match = re.search(r"(\{.*\})", payload_text, flags=re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                payload = {}
            requirements = dict(payload.get("requirements") or {}) if isinstance(payload, dict) else {}
            add(str(requirements.get("primary_source_target", "") or ""))
            for item in list(requirements.get("allowed_source_files", []) or []):
                add(str(item))
            for item in list(requirements.get("required_source_files", []) or []):
                add_required(str(item))
            try:
                max_source_files = max(1, int(requirements.get("max_source_files", 1) or 1))
            except (TypeError, ValueError):
                max_source_files = 1
        if not targets:
            primary = re.search(r'"primary_source_target"\s*:\s*"([^"]+)"', text)
            if primary:
                add(primary.group(1))
            allowed = re.search(r'"allowed_source_files"\s*:\s*\[(.*?)\]', text)
            if allowed:
                for match in re.findall(r'"([^"]+)"', allowed.group(1)):
                    add(match)
        if not required_source_files:
            required = re.search(r'"required_source_files"\s*:\s*\[(.*?)\]', text)
            if required:
                for match in re.findall(r'"([^"]+)"', required.group(1)):
                    add_required(match)
        if max_source_files == 1:
            max_match = re.search(r'"max_source_files"\s*:\s*(\d+)', text)
            if max_match:
                try:
                    max_source_files = max(1, int(max_match.group(1)))
                except ValueError:
                    max_source_files = 1
        return {
            "targets": targets,
            "required_source_files": required_source_files,
            "max_source_files": max_source_files,
        }

    def _prepend_strict_source_edit_script_guard(
        self,
        plan: str,
        recovery_context: str,
    ) -> str:
        if not self._is_masguard_source_edit_contract(recovery_context):
            return str(plan or "")
        requirements = self._extract_masguard_source_edit_requirements(recovery_context)
        targets = list(requirements.get("targets", []) or [])
        required_source_files = list(requirements.get("required_source_files", []) or [])
        max_source_files = int(requirements.get("max_source_files", 1) or 1)
        lines = [
            "[MASGUARD STRICT SOURCE-EDIT SCRIPT REQUIRED]",
            "This replay has already localized a bounded source-edit target. The implementer must turn the plan into an executable patch command, not another read-only diagnosis.",
            "The next implementer answer should contain exactly one ```bash``` block.",
            "That bash block must either edit repository source code or explicitly abstain without touching the repository.",
            "Candidate admission rule: no fresh canonical source diff means the candidate is rejected, not completed.",
            "Diff-first interface: before focused validation or completion, print exactly one MASGUARD_DIFF_INTENT={...} JSON object naming target_file, edit_kind, expected_changed_symbol, and validation_command.",
            "Repository-root interface: before reading or writing a target path, discover the real repository root used by the command runtime. Do not assume /testbed or the current shell directory contains the source tree.",
            "Required root-discovery pattern: set root = first existing directory among Path.cwd(), Path.cwd().parents, /testbed, /workspace, and /repo where all required/allowed target files exist; then read/write root / target_path.",
            "If no such root is found, print pwd, list the immediate directory once, then abstain with MASGUARD_ABSTAIN_REASON=repository_root_not_found; do not retry the same missing relative or /testbed path.",
            "If no safe edit is possible, print MASGUARD_STRICT_ABSTAIN_NO_EDIT and MASGUARD_ABSTAIN_REASON=<one concrete reason>, then exit nonzero.",
            "If the first anchor is uncertain, run at most one bounded in-file probe of the allowed source target, then immediately choose edit or explicit abstention.",
            "Allowed edit mechanisms include a Python heredoc, git apply, sed -i, or perl -pi that changes a canonical source file.",
            "Patch scripts should use robust local anchors: parse with ast when practical, or use short regex/line-local anchors around the exact function/class/branch. Do not rely on large exact multi-line string replacements copied from memory.",
            "Recommended Python edit pattern: discover root first, set target = root / target_path, read the target file, use re.subn with a short escaped local anchor or AST-derived line span, require count == 1, write the file, then assert git -C root diff --quiet returns nonzero before printing a completion marker.",
            "If an anchor is missing, print the nearby focused context, print MASGUARD_STRICT_ABSTAIN_NO_EDIT plus MASGUARD_ABSTAIN_REASON=<one concrete reason>, and exit nonzero. Do not print COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT, IMPLEMENTATION_COMPLETE, or a success marker after a no-op/abstain path.",
            "Forbidden final action: cat/grep/rg/find/sed -n/head/tail-only browsing, test-only edits, generated-file edits, broad exception swallowing, or unrelated cleanup.",
            "After the source edit, run focused validation in the same bash block when feasible; otherwise run py_compile on the edited file.",
        ]
        if targets:
            lines.append("Allowed source targets, in priority order:")
            lines.extend(f"- {path}" for path in targets[:6])
            lines.append(f"Patch at most {max_source_files} allowed source file(s).")
        if required_source_files:
            lines.append("Required source files for this contract:")
            lines.extend(f"- {path}" for path in required_source_files[:6])
            lines.append("The final source diff must touch every required source file unless the bash block explicitly abstains.")
        lines.append(
            "If evidence is still insufficient after the bounded source-target probe, the bash block may abstain only by printing MASGUARD_STRICT_ABSTAIN_NO_EDIT and MASGUARD_ABSTAIN_REASON, exiting nonzero, and making no repository changes."
        )
        return "\n".join(lines) + "\n\n--- planner output to convert into patch script ---\n" + str(plan or "")

    def _audit_fresh_recovery_diff(
        self,
        impl_result: dict[str, Any],
        *,
        baseline: dict[str, Any],
        recovery_context: str,
    ) -> dict[str, Any]:
        if not recovery_context.strip():
            return impl_result

        updated = dict(impl_result)
        summary = dict(updated.get("patch_summary") or {})
        post = self._workspace_diff_fingerprint()
        fresh_files = self._fresh_changed_files(baseline, post)
        removed_stale_files = self._removed_stale_diff_files(baseline, post)
        fresh_classes = classify_changed_files(fresh_files)

        summary.update(
            {
                "fresh_changed_files": fresh_files,
                "fresh_changed_file_classes": fresh_classes,
                "removed_stale_diff_files": removed_stale_files,
                "has_fresh_source_diff": bool(fresh_classes.get("source_files", [])),
                "recovery_fresh_diff_audit": {
                    "pre_replay_diff": baseline,
                    "post_replay_diff": post,
                    "replay_diff_changed": post.get("diff_digest") != baseline.get("diff_digest"),
                },
            }
        )
        if not fresh_files:
            summary["failure_mode"] = "no_fresh_patch"
            summary["fresh_target_legitimacy"] = "no_diff"
            updated["success"] = False
        elif not fresh_classes.get("source_files"):
            summary["failure_mode"] = "no_fresh_source_patch"
            summary["fresh_target_legitimacy"] = self._target_legitimacy(fresh_classes, fresh_files)
            updated["success"] = False
        else:
            summary["fresh_target_legitimacy"] = self._target_legitimacy(fresh_classes, fresh_files)
        updated["patch_summary"] = summary
        if not updated.get("success") and summary.get("failure_mode") in {"no_fresh_patch", "no_fresh_source_patch"}:
            updated["error"] = str(updated.get("error", "") or summary.get("failure_mode", ""))
        return updated

    def _workspace_diff_fingerprint(self) -> dict[str, Any]:
        raw_diff = "\n".join(
            text
            for text in (
                str(self.executor.execute("git diff --binary").get("output", "") or ""),
                str(self.executor.execute("git diff --cached --binary").get("output", "") or ""),
            )
            if text.strip()
        )
        changed_files = self._workspace_changed_files()
        return {
            "changed_files": changed_files,
            "diff_digest": hashlib.sha256(raw_diff.encode("utf-8")).hexdigest()[:16] if raw_diff.strip() else "",
            "file_digests": self._diff_file_digests(raw_diff),
        }

    def _workspace_changed_files(self) -> list[str]:
        def _git_lines(command: str) -> list[str]:
            result = self.executor.execute(command)
            return [
                line.strip()
                for line in str(result.get("output", "") or "").splitlines()
                if line.strip()
            ]

        def _numstat_changed_files(command: str) -> list[str]:
            paths: list[str] = []
            for line in _git_lines(command):
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                added, deleted = parts[0].strip(), parts[1].strip()
                if (added, deleted) in {("0", "0"), ("-", "-")}:
                    continue
                normalized = normalize_repo_path(parts[-1].strip())
                if normalized:
                    paths.append(normalized)
            return paths

        untracked_files = [
            normalize_repo_path(path)
            for path in _git_lines("git ls-files --others --exclude-standard")
            if normalize_repo_path(path)
        ]
        return sorted(
            dict.fromkeys(
                _numstat_changed_files("git diff --numstat")
                + _numstat_changed_files("git diff --cached --numstat")
                + untracked_files
            )
        )

    @staticmethod
    def _fresh_changed_files(pre: dict[str, Any], post: dict[str, Any]) -> list[str]:
        pre_digests = dict(pre.get("file_digests", {}) or {})
        post_digests = dict(post.get("file_digests", {}) or {})
        if pre_digests or post_digests:
            return sorted(
                path
                for path in set(post_digests)
                if str(post_digests.get(path, "") or "") != str(pre_digests.get(path, "") or "")
            )
        pre_files = set(str(item) for item in list(pre.get("changed_files", []) or []))
        post_files = set(str(item) for item in list(post.get("changed_files", []) or []))
        return sorted(path for path in post_files if path and path not in pre_files)

    @staticmethod
    def _removed_stale_diff_files(pre: dict[str, Any], post: dict[str, Any]) -> list[str]:
        pre_digests = dict(pre.get("file_digests", {}) or {})
        post_digests = dict(post.get("file_digests", {}) or {})
        if pre_digests or post_digests:
            return sorted(path for path in set(pre_digests) if path and path not in set(post_digests))
        pre_files = set(str(item) for item in list(pre.get("changed_files", []) or []))
        post_files = set(str(item) for item in list(post.get("changed_files", []) or []))
        return sorted(path for path in pre_files if path and path not in post_files)

    @staticmethod
    def _diff_file_digests(raw_diff: str) -> dict[str, str]:
        file_chunks: dict[str, list[str]] = {}
        current_path = ""
        current_lines: list[str] = []
        for line in str(raw_diff or "").splitlines():
            if line.startswith("diff --git "):
                if current_path:
                    file_chunks[current_path] = list(current_lines)
                current_path = PlannerPatcherAdapter._path_from_diff_header(line)
                current_lines = [line]
                continue
            if current_path:
                current_lines.append(line)
        if current_path:
            file_chunks[current_path] = list(current_lines)
        return {
            path: hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]
            for path, lines in sorted(file_chunks.items())
            if path
        }

    @staticmethod
    def _path_from_diff_header(header: str) -> str:
        parts = header.split()
        if len(parts) < 4:
            return ""
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path == "/dev/null" and len(parts) >= 3:
            path = parts[2]
            if path.startswith("a/"):
                path = path[2:]
        return normalize_repo_path(path)

    @staticmethod
    def _target_legitimacy(classes: dict[str, list[str]], changed_files: list[str]) -> str:
        source_files = list(classes.get("source_files", []) or [])
        test_files = list(classes.get("test_files", []) or [])
        generated_files = list(classes.get("generated_files", []) or [])
        other_files = list(classes.get("other_files", []) or [])
        if not changed_files:
            return "no_diff"
        if source_files and not test_files and not generated_files and not other_files:
            return "source_only"
        if source_files and (test_files or other_files):
            return "source_mixed"
        if test_files and not source_files:
            return "tests_only"
        if generated_files and not source_files:
            return "generated_only"
        return "non_source_only"

    @staticmethod
    def _looks_like_infra_error(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(
            marker in lowered
            for marker in (
                "openai-compatible api request failed",
                "auth_unavailable",
                "insufficient_quota",
                "bad gateway",
                "timed out",
                "timeout",
                "rate limit",
                "too many requests",
                "permission_error",
                "server_error",
            )
        )

    def patch(
        self,
        issue: str,
        workspace: str,
        *,
        located_files: str,
        recovery_context: str = "",
        escalation_level: int = 0,
    ) -> dict[str, Any]:
        from swe_mas.agents.implementer import ImplementerAgent
        from swe_mas.agents.planner import PlannerAgent

        compact_recovery = self.recovery_execution_profile == "compact"
        plan_iteration_budget = self.max_plan_iterations
        patch_iteration_budget = self.max_patch_iterations
        if compact_recovery:
            plan_iteration_budget = min(plan_iteration_budget, 1)
            patch_iteration_budget = min(patch_iteration_budget, 4)
        strict_source_edit = self._is_masguard_source_edit_contract(recovery_context)
        if strict_source_edit:
            patch_iteration_budget = max(patch_iteration_budget, 3)

        issue_with_context = self._compose_recovery_issue(
            issue=issue,
            located_files=located_files,
            recovery_context=recovery_context,
        )
        audit_fresh_diff = bool(self.enable_recovery_fresh_diff_audit and recovery_context.strip())
        recovery_diff_baseline = self._workspace_diff_fingerprint() if audit_fresh_diff else {}

        planner = PlannerAgent(
            model=self.planner_model or self.model,
            executor=self.executor,
            recorder=self.recorder,
            session_id=self.session_id,
        )
        planner.config.max_iterations = plan_iteration_budget
        plan_result = planner.run(
            problem_analysis=issue_with_context,
            located_files=located_files,
            reproduction_info="",
        )
        plan_audit = self._patch_intent_plan_audit(
            str(plan_result.get("plan", "") or ""),
            recovery_context,
        )
        plan_retry_used = False
        if bool(recovery_context.strip()) and not bool(plan_audit.get("satisfied", True)):
            plan_retry_used = True
            plan_feedback = self._patch_intent_plan_feedback(plan_audit)
            retry_plan_result = planner.run(
                problem_analysis=issue_with_context,
                located_files=located_files,
                reproduction_info=plan_feedback,
            )
            retry_plan_audit = self._patch_intent_plan_audit(
                str(retry_plan_result.get("plan", "") or ""),
                recovery_context,
            )
            if bool(retry_plan_audit.get("satisfied", False)) or not str(plan_result.get("plan", "") or "").strip():
                plan_result = retry_plan_result
                plan_audit = retry_plan_audit
        planner_error = str(plan_result.get("error", "") or "")
        if not planner_error and not bool(plan_result.get("success")):
            planner_error = str(plan_result.get("plan", "") or "")
        if self._looks_like_infra_error(planner_error) and not str(plan_result.get("plan", "") or "").strip():
            return {
                "success": False,
                "plan": "",
                "patch": "",
                "planner_error": planner_error,
                "implementer_error": "",
                "infrastructure_error": True,
                "planner_messages": plan_result.get("messages", []),
                "messages": [],
                "commands": [],
                "patch_summary": {},
                "retry_used": False,
                "retry_reason": "planner_infrastructure_error",
            }
        if self._looks_like_infra_error(planner_error) and not bool(plan_result.get("success")):
            return {
                "success": False,
                "plan": str(plan_result.get("plan", "") or ""),
                "patch": "",
                "planner_error": planner_error,
                "implementer_error": "",
                "infrastructure_error": True,
                "planner_messages": plan_result.get("messages", []),
                "messages": [],
                "commands": [],
                "patch_summary": {},
                "retry_used": False,
                "retry_reason": "planner_infrastructure_error",
            }

        implementer = ImplementerAgent(
            model=self.implementer_model or self.model,
            executor=self.executor,
            recorder=self.recorder,
            session_id=self.session_id,
        )
        if recovery_context.strip():
            implementer._force_recovery_mode = True
        set_baseline = getattr(implementer, "set_recovery_diff_baseline", None)
        if callable(set_baseline) and recovery_diff_baseline:
            set_baseline(recovery_diff_baseline)
        implementer.config.max_iterations = max(
            patch_iteration_budget,
            patch_iteration_budget + escalation_level,
        )
        impl_result = implementer.run(
            fix_plan=self._prepend_patch_intent_plan_guard(
                str(plan_result.get("plan", "") or ""),
                plan_audit,
                recovery_context,
            ),
            cwd=workspace,
            problem_statement=issue_with_context,
        )
        if audit_fresh_diff:
            impl_result = self._audit_fresh_recovery_diff(
                impl_result,
                baseline=recovery_diff_baseline,
                recovery_context=recovery_context,
            )
        retry_used = False
        retry_reason = ""
        force_contract_retry = self._is_post_evidence_source_repair_intent(recovery_context)
        if (
            (self.enable_recovery_retry or force_contract_retry)
            and self._needs_recovery_retry(impl_result, recovery_context)
        ):
            retry_used = True
            retry_reason = str(dict(impl_result.get("patch_summary") or {}).get("failure_mode", "") or "ineffective_patch")
            logger.info(
                "[PATCHER-RETRY] recovery_context active; re-planning after implementer failure_mode=%s",
                retry_reason,
            )
            retry_feedback = self._recovery_retry_feedback(impl_result, recovery_context)
            retry_issue = self._compose_recovery_issue(
                issue=issue,
                located_files=located_files,
                recovery_context=recovery_context,
                previous_failure_mode=retry_reason,
                previous_attempt_feedback=retry_feedback,
            )
            retry_planner = PlannerAgent(
                model=self.planner_model or self.model,
                executor=self.executor,
                recorder=self.recorder,
                session_id=self.session_id,
            )
            retry_planner.config.max_iterations = plan_iteration_budget
            retry_plan_result = retry_planner.run(
                problem_analysis=retry_issue,
                located_files=located_files,
                reproduction_info=retry_feedback
                or f"Previous replay attempt failed with failure_mode={retry_reason}. Re-plan with stricter source-target focus.",
            )
            retry_plan_audit = self._patch_intent_plan_audit(
                str(retry_plan_result.get("plan", "") or ""),
                recovery_context,
            )
            retry_implementer = ImplementerAgent(
                model=self.implementer_model or self.model,
                executor=self.executor,
                recorder=self.recorder,
                session_id=self.session_id,
            )
            if recovery_context.strip():
                retry_implementer._force_recovery_mode = True
            set_retry_baseline = getattr(retry_implementer, "set_recovery_diff_baseline", None)
            if callable(set_retry_baseline) and recovery_diff_baseline:
                set_retry_baseline(recovery_diff_baseline)
            retry_implementer.config.max_iterations = max(
                patch_iteration_budget + 2,
                patch_iteration_budget + escalation_level + 2,
            )
            retry_impl_result = retry_implementer.run(
                fix_plan=self._prepend_patch_intent_plan_guard(
                    str(retry_plan_result.get("plan", "") or ""),
                    retry_plan_audit,
                    recovery_context,
                ),
                cwd=workspace,
                problem_statement=retry_issue,
            )
            if audit_fresh_diff:
                retry_impl_result = self._audit_fresh_recovery_diff(
                    retry_impl_result,
                    baseline=recovery_diff_baseline,
                    recovery_context=recovery_context,
                )
            if retry_impl_result.get("success") or not impl_result.get("success"):
                plan_result = retry_plan_result
                impl_result = retry_impl_result
        implementer_error = str(impl_result.get("error", "") or "")
        return {
            "success": bool(plan_result.get("success")) and bool(impl_result.get("success")),
            "plan": plan_result.get("plan", ""),
            "patch": impl_result.get("patch", ""),
            "planner_error": planner_error,
            "implementer_error": implementer_error,
            "infrastructure_error": self._looks_like_infra_error(planner_error)
            or self._looks_like_infra_error(implementer_error),
            "planner_messages": plan_result.get("messages", []),
            "messages": impl_result.get("messages", []),
            "commands": impl_result.get("commands", []),
            "patch_summary": impl_result.get("patch_summary", {}),
            "retry_used": retry_used,
            "retry_reason": retry_reason,
            "patch_intent_plan_audit": plan_audit,
            "patch_intent_plan_retry_used": plan_retry_used,
        }
