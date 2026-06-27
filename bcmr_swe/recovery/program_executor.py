"""Execute a RecoveryProgram step-by-step on a live Coordinator.

Each atomic operator is mapped to concrete actions on the SWE substrate:

- INSPECT  → run verifier in diagnostic mode or execute test commands
- REVOKE   → quarantine a shared fact in the provenance graph
- ROLLBACK → restore workspace from a checkpoint archive
- REPLAY   → re-run pipeline stages via Coordinator._resume_from()
- ESCALATE → increase iterations / escalation_level for a stage

The executor tracks per-step outcomes so the full program execution trace
can be saved in the counterfactual dataset.
"""

from __future__ import annotations

import logging
import hashlib
import os
import re
import subprocess
import time
from typing import Any, Protocol

from bcmr_swe.types import (
    OpType,
    ProgramOutcome,
    RecoveryProgram,
    RecoveryStep,
)
from swe_mas.utils.command_classification import (
    looks_like_readonly_probe_command,
    looks_like_validation_command,
    looks_like_write_command,
)
from swe_mas.utils.path_filters import canonical_source_paths, existing_repo_source_paths

logger = logging.getLogger(__name__)


def locator_result_is_recovery_best_effort(
    locate_result: dict[str, Any],
    *,
    recovery_context: str,
) -> bool:
    """True when a failed locator result is still usable as a recovery hint."""

    if bool(locate_result.get("success", False)):
        return False
    if not bool(locate_result.get("best_effort_only", False)):
        return False
    if not str(locate_result.get("located_files", "") or "").strip():
        return False
    if not str(recovery_context or "").strip():
        return False
    if bool(locate_result.get("provider_error", False)) or bool(locate_result.get("infrastructure_error", False)):
        return False
    return True


def locator_result_allows_recovery_replay(
    locate_result: dict[str, Any],
    *,
    recovery_context: str,
) -> bool:
    """Return whether downstream recovery stages may consume locator output."""

    return bool(locate_result.get("success", False)) or locator_result_is_recovery_best_effort(
        locate_result,
        recovery_context=recovery_context,
    )


def _workspace_diff_fingerprint(workspace: str) -> dict[str, Any]:
    """Return a compact content-diff fingerprint for replay diagnostics."""

    if not workspace or not os.path.isdir(str(workspace)):
        return {"changed_files": [], "diff_numstat": "", "diff_digest": "", "file_digests": {}}
    raw_proc = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    raw_cached_proc = subprocess.run(
        ["git", "diff", "--cached", "--binary"],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    cached_proc = subprocess.run(
        ["git", "diff", "--cached", "--numstat"],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    proc = subprocess.run(
        ["git", "diff", "--numstat"],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0 and cached_proc.returncode != 0:
        return {
            "changed_files": [],
            "diff_numstat": "",
            "diff_digest": "",
            "file_digests": {},
            "error": (proc.stderr or cached_proc.stderr or raw_proc.stderr or raw_cached_proc.stderr or "")[:400],
        }
    lines = [
        line.strip()
        for line in (str(proc.stdout or "") + "\n" + str(cached_proc.stdout or "")).splitlines()
        if line.strip()
    ]
    changed_files: list[str] = []
    content_lines: list[str] = []
    for line in lines:
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added, deleted, path = (part.strip() for part in parts)
        if (added, deleted) in {("0", "0"), ("-", "-")}:
            continue
        content_lines.append(f"{added}\t{deleted}\t{path}")
        changed_files.append(path)
    diff_numstat = "\n".join(sorted(content_lines))
    raw_diff = str(raw_proc.stdout or "") + "\n" + str(raw_cached_proc.stdout or "")
    return {
        "changed_files": sorted(dict.fromkeys(changed_files)),
        "diff_numstat": diff_numstat[:2000],
        "diff_digest": hashlib.sha256(raw_diff.encode("utf-8")).hexdigest()[:16] if raw_diff.strip() else "",
        "file_digests": _diff_file_digests(raw_diff),
    }


def _diff_file_digests(raw_diff: str) -> dict[str, str]:
    file_chunks: dict[str, list[str]] = {}
    current_path = ""
    current_lines: list[str] = []
    for line in str(raw_diff or "").splitlines():
        if line.startswith("diff --git "):
            if current_path:
                file_chunks[current_path] = list(current_lines)
            current_path = _path_from_diff_header(line)
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
    return path.replace("\\", "/")


class CoordinatorProtocol(Protocol):
    """Minimal interface the executor needs from the Coordinator."""

    recorder: Any
    shared_facts: dict[str, dict]
    stage_outputs: dict[str, dict]
    workspace: str

    def _resume_from(
        self,
        stage: str,
        *,
        recovery_context: str,
        escalation_level: int,
        deep_verify: bool,
    ) -> bool: ...

    def _clear_stage_outputs_from(self, stage: str) -> None: ...
    def _usage_snapshot(self) -> dict[str, float]: ...
    def _progress_score(self) -> float: ...


class ProgramExecutor:
    """Execute a recovery program on a Coordinator, collecting per-step traces."""

    def execute(
        self,
        coordinator: CoordinatorProtocol,
        program: RecoveryProgram,
    ) -> ProgramOutcome:
        started = time.time()
        usage_before = coordinator._usage_snapshot()
        progress_before = coordinator._progress_score()
        step_outcomes: list[dict[str, Any]] = []
        escalation_level = 0
        context_hints: list[str] = []
        success = False

        for i, step in enumerate(program.steps):
            step_start = time.time()
            step_result = self._execute_step(
                coordinator, step,
                escalation_level=escalation_level,
                context_hints=context_hints,
            )
            step_latency = time.time() - step_start
            step_outcomes.append({
                "step_index": i,
                "op": step.op.value,
                "args": step.args,
                "result": step_result,
                "latency_sec": step_latency,
            })

            if step.op == OpType.ESCALATE:
                escalation_level = max(
                    escalation_level,
                    int(step.args.get("escalation_level", 1)),
                )
            if step.op == OpType.INSPECT:
                findings = step_result.get("findings", "")
                if findings:
                    context_hints.append(f"[Step {i}] Diagnosis: {findings}")
            if step.op == OpType.REVOKE and step_result.get("revoked"):
                revoked_key = step_result.get("fact_key") or step_result.get("fact_id") or "unknown_fact"
                context_hints.append(f"[Step {i}] Revoked stale fact: {revoked_key}.")
            if step.op == OpType.ROLLBACK and step_result.get("restored"):
                checkpoint_label = step_result.get("checkpoint_label") or "unknown"
                resume_from = step_result.get("resume_from") or "patcher"
                context_hints.append(
                    f"[Step {i}] Restored checkpoint {checkpoint_label}; continue from {resume_from}."
                )
            if step.op == OpType.REPLAY:
                replay_success = step_result.get("success", False)
                success = replay_success
                if replay_success:
                    context_hints.append(f"[Step {i}] Replay succeeded.")
                else:
                    context_hints.append(f"[Step {i}] Replay did not succeed yet.")
            if step.op == OpType.SELECTIVE_REPLAY:
                replay_success = step_result.get("success", False)
                success = replay_success
                role = step_result.get("role", "")
                if replay_success:
                    context_hints.append(
                        f"[Step {i}] Selective replay of {role} succeeded."
                    )
                else:
                    context_hints.append(
                        f"[Step {i}] Selective replay of {role} did not succeed yet."
                    )

        usage_after = coordinator._usage_snapshot()
        token_cost = max(
            0.0,
            float(usage_after.get("total_tokens", 0))
            - float(usage_before.get("total_tokens", 0)),
        )
        progress_after = coordinator._progress_score()

        return ProgramOutcome(
            program_id=program.program_id,
            recover_success=success,
            official_resolved=success,
            token_cost=token_cost,
            latency_sec=max(0.001, time.time() - started),
            secondary_risk=program.estimated_risk,
            milestone_gain=max(0.0, progress_after - progress_before),
            step_outcomes=step_outcomes,
            notes=f"Executed {len(program.steps)}-step program: {program.skeleton}",
            metadata=dict(program.metadata or {}),
        )

    def _execute_step(
        self,
        coord: CoordinatorProtocol,
        step: RecoveryStep,
        *,
        escalation_level: int,
        context_hints: list[str],
    ) -> dict[str, Any]:
        op = step.op
        args = step.args

        if op == OpType.INSPECT:
            return self._exec_inspect(coord, args)
        if op == OpType.REVOKE:
            return self._exec_revoke(coord, args)
        if op == OpType.ROLLBACK:
            return self._exec_rollback(coord, args)
        if op == OpType.ESCALATE:
            return self._exec_escalate(coord, args)
        if op == OpType.REPLAY:
            return self._exec_replay(coord, args, escalation_level, context_hints)
        if op == OpType.SELECTIVE_REPLAY:
            return self._exec_selective_replay(coord, args, escalation_level, context_hints)

        return {"error": f"unknown operator {op.value}"}

    def _exec_inspect(
        self, coord: CoordinatorProtocol, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Run real diagnostics: execute test commands and/or re-invoke verifier.

        Unlike the v2 placeholder that just read cached outputs, this
        actually runs commands in the workspace to produce fresh evidence.
        """
        target = str(args.get("target", "patch"))
        depth = str(args.get("depth", "quick"))

        if target in ("patch", "test_output"):
            test_result = self._run_workspace_tests(coord)
            cached_verify = coord.stage_outputs.get("verifier", {}).get("verification", "")
            combined = test_result.get("output", "")
            if depth == "deep" and cached_verify:
                combined += f"\n--- Prior verifier output ---\n{cached_verify[:300]}"
            return {
                "target": target,
                "depth": depth,
                "findings": combined[:800],
                "test_returncode": test_result.get("returncode", -1),
                "test_passed": test_result.get("returncode", 1) == 0,
                "confidence": 0.85 if test_result.get("returncode") is not None else 0.3,
            }

        if target == "localization":
            loc_result = coord.stage_outputs.get("locator", {})
            located = loc_result.get("located_files", "")
            if depth == "deep" and hasattr(coord, '_run_locator_diagnostic'):
                diag = coord._run_locator_diagnostic()
                located = f"{located}\n--- Deep diagnostic ---\n{diag}"
            return {
                "target": target,
                "depth": depth,
                "findings": located[:800] if located else "No localization output available.",
                "confidence": 0.7 if located else 0.3,
            }

        if target.startswith("fact:"):
            fact_id, node, fact_key = self._resolve_fact_reference(coord, target)
            if node:
                return {
                    "target": target,
                    "depth": depth,
                    "findings": f"Fact {fact_key or fact_id}: status={node.status}, content={str(node.content)[:200]}",
                    "confidence": 0.8,
                }
            return {"target": target, "depth": depth, "findings": "Fact not found.", "confidence": 0.2}

        return {"target": target, "depth": depth, "findings": "Unknown target.", "confidence": 0.1}

    def _run_workspace_tests(self, coord: CoordinatorProtocol) -> dict[str, Any]:
        """Actually execute test commands in the workspace."""
        executor = getattr(coord, 'executor', None)
        if executor is None:
            for agent in ('verifier', 'patcher', 'locator'):
                agent_obj = getattr(coord, agent, None)
                executor = getattr(agent_obj, 'executor', None)
                if executor is not None:
                    break
        if executor is None:
            return {"output": "No executor available for test execution.", "returncode": -1}

        run_fn = getattr(executor, 'execute', None) or getattr(executor, 'run', None)
        if not callable(run_fn):
            return {"output": "Executor has no run method.", "returncode": -1}

        test_cmd = self._resolve_test_command(coord)
        try:
            result = run_fn(test_cmd, timeout=60)
            if isinstance(result, dict):
                return {
                    "output": str(result.get("output", result.get("stdout", "")))[:800],
                    "returncode": int(result.get("returncode", result.get("exit_code", -1))),
                }
            return {"output": str(result)[:800], "returncode": -1}
        except Exception as exc:
            return {"output": f"Test execution error: {exc}", "returncode": -1}

    def _resolve_test_command(self, coord: CoordinatorProtocol) -> str:
        """Use the harness/manifest test command if available, else default."""
        for src in (
            getattr(coord, "test_command", None),
            getattr(coord, "_test_command", None),
        ):
            if isinstance(src, str) and src.strip():
                return src.strip()
        stage_out = getattr(coord, "stage_outputs", {})
        verifier_out = stage_out.get("verifier", {})
        for cmd_field in ("test_command", "test_cmd"):
            cmd = verifier_out.get(cmd_field)
            if isinstance(cmd, str) and cmd.strip():
                return cmd.strip()
        return "python -m pytest --tb=short -q 2>&1 | tail -40"

    def _exec_revoke(
        self, coord: CoordinatorProtocol, args: dict[str, Any]
    ) -> dict[str, Any]:
        fact_ref = str(args.get("fact_id", ""))
        fact_id, node, fact_key = self._resolve_fact_reference(coord, fact_ref)
        if not fact_id:
            for fkey, fval in list(coord.shared_facts.items()):
                candidate = coord.recorder.graph.get_node(fval.get("node_id", ""))
                if candidate and candidate.status == "conflicted":
                    fact_id = candidate.node_id
                    node = candidate
                    fact_key = fkey
                    break

        if node:
            node.status = "quarantined"
            if fact_key:
                coord.shared_facts.pop(fact_key, None)
            else:
                for fkey, fval in list(coord.shared_facts.items()):
                    if fval.get("node_id") == fact_id:
                        coord.shared_facts.pop(fkey, None)
                        fact_key = fkey
                        break
            return {"revoked": True, "fact_id": fact_id, "fact_key": fact_key}
        return {"revoked": False, "reason": "fact not found"}

    def _exec_rollback(
        self, coord: CoordinatorProtocol, args: dict[str, Any]
    ) -> dict[str, Any]:
        ckpt_id = str(args.get("checkpoint_id", ""))
        checkpoint = coord.recorder.get_checkpoint(ckpt_id) or coord.recorder.latest_checkpoint()
        if checkpoint:
            coord.recorder.checkpoint_store.restore(checkpoint, coord.workspace)
            resume_from = checkpoint.metadata.get("resume_from", "patcher")
            coord._clear_stage_outputs_from(resume_from)
            return {
                "restored": True,
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_label": str(args.get("checkpoint_label", "")).strip() or getattr(checkpoint, "label", ""),
                "resume_from": resume_from,
            }
        return {"restored": False, "reason": "no checkpoint available"}

    def _exec_escalate(
        self, coord: CoordinatorProtocol, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Actually modify agent execution config for the target scope."""
        scope = str(args.get("scope", "patcher"))
        strategy = str(args.get("strategy", "more_iterations"))
        escalation_level = int(args.get("escalation_level", 1))

        scope_names = [item.strip() for item in re.split(r"[+,]", scope) if item.strip()]
        if scope == "full":
            scope_names = ["locator", "patcher", "verifier"]
        agent = getattr(coord, scope, None)
        changes: list[str] = []

        if strategy == "more_iterations" and agent is not None:
            for attr in ("max_iterations", "max_patch_iterations", "max_plan_iterations"):
                current = getattr(agent, attr, None)
                if isinstance(current, int):
                    new_val = current + 2 * escalation_level
                    setattr(agent, attr, new_val)
                    changes.append(f"{attr}: {current} → {new_val}")

        if strategy == "broader_search":
            boosted_any = False
            for scope_name in scope_names or [scope]:
                scoped_agent = getattr(coord, scope_name, None)
                if scoped_agent is None:
                    continue
                for attr in ("max_iterations", "max_patch_iterations", "max_plan_iterations"):
                    current = getattr(scoped_agent, attr, None)
                    if isinstance(current, int):
                        new_val = current + 3 * escalation_level
                        setattr(scoped_agent, attr, new_val)
                        changes.append(f"{scope_name}.{attr}: {current} → {new_val}")
                        boosted_any = True
            if not boosted_any and agent is not None:
                for attr in ("max_iterations",):
                    current = getattr(agent, attr, None)
                    if isinstance(current, int):
                        new_val = current + 3 * escalation_level
                        setattr(agent, attr, new_val)
                        changes.append(f"{attr}: {current} → {new_val}")

        if strategy == "stronger_prompt":
            changes.append(f"escalation_level set to {escalation_level}")
            changes.extend(self._route_strong_model_for_scope(coord, scope))

        capability_changed = any(
            "→" in change
            or "routed_to_strong_" in change
            for change in changes
        ) or (strategy == "stronger_prompt" and escalation_level > 0)
        return {
            "escalated": True,
            "scope": scope,
            "strategy": strategy,
            "escalation_level": escalation_level,
            "config_changes": changes,
            "capability_changed": capability_changed,
        }

    @staticmethod
    def _model_capability_identity(model: Any) -> str:
        """Return a stable identity for whether two wrappers expose the same model."""

        seen: set[int] = set()
        cursor = model
        while cursor is not None and id(cursor) not in seen:
            seen.add(id(cursor))
            config = getattr(cursor, "config", None)
            if config is not None:
                fields = []
                for attr in ("provider", "model", "model_name", "base_url", "endpoint", "api_base"):
                    value = getattr(config, attr, None)
                    if isinstance(value, str) and value.strip():
                        fields.append(f"{attr}={value.strip()}")
                if fields:
                    return f"{type(cursor).__module__}.{type(cursor).__name__}|" + "|".join(fields)
            for attr in ("base_model", "model"):
                nested = getattr(cursor, attr, None)
                if nested is not None and nested is not cursor and not isinstance(nested, str):
                    cursor = nested
                    break
            else:
                break
        return ""

    def _route_strong_model_for_scope(
        self,
        coord: CoordinatorProtocol,
        scope: str,
    ) -> list[str]:
        """Route the requested stage to the configured strong model when available.

        `CAPABILITY_BOOST` should represent a real local capability increase,
        not just a longer loop.  If the runtime did not configure a strong
        model, this intentionally becomes a no-op and the caller still gets the
        prompt/iteration boost.
        """
        strong_models = dict(getattr(coord, "_bcmr_strong_stage_models", {}) or {})
        if not strong_models:
            return ["strong_model_unavailable"]

        normalized_scope = str(scope or "").strip().lower()
        if normalized_scope in {"patcher", "patcher+verifier", "local_repair"}:
            stage_attrs = [
                ("patcher", "planner_model", "planner"),
                ("patcher", "implementer_model", "implementer"),
                ("patcher", "model", "implementer"),
            ]
        elif normalized_scope in {"locator", "locator+patcher+verifier", "full"}:
            stage_attrs = [
                ("locator", "model", "locator"),
            ]
        elif normalized_scope == "verifier":
            stage_attrs = [
                ("verifier", "model", "verifier"),
            ]
        else:
            stage_attrs = [
                ("patcher", "implementer_model", "implementer"),
            ]

        changes: list[str] = []
        for component_name, attr, stage_name in stage_attrs:
            strong_model = strong_models.get(stage_name)
            component = getattr(coord, component_name, None)
            if component is None or strong_model is None:
                continue
            current = getattr(component, attr, None)
            if current is strong_model:
                changes.append(f"{component_name}.{attr}: already_strong")
                continue
            current_identity = self._model_capability_identity(current)
            strong_identity = self._model_capability_identity(strong_model)
            if current_identity and current_identity == strong_identity:
                changes.append(f"{component_name}.{attr}: same_model_noop_{stage_name}")
                continue
            setattr(component, attr, strong_model)
            changes.append(f"{component_name}.{attr}: routed_to_strong_{stage_name}")
        if not changes:
            changes.append("strong_model_scope_unmatched")
        return changes

    def _exec_replay(
        self,
        coord: CoordinatorProtocol,
        args: dict[str, Any],
        escalation_level: int,
        context_hints: list[str],
    ) -> dict[str, Any]:
        scope = str(args.get("scope", "patcher+verifier"))
        execution_profile = str(args.get("execution_profile", "") or "").strip().lower() or "normal"
        user_hint = str(args.get("context_hint", ""))
        all_hints = list(context_hints)
        if user_hint:
            all_hints.append(user_hint)
        recovery_context = self._compose_replay_context(
            coord,
            scope=scope,
            context_hints=all_hints,
        )

        if "+" in scope:
            stage = scope.split("+")[0].strip()
        elif scope == "full":
            stage = "locator"
        else:
            stage = scope

        pre_replay_diff = _workspace_diff_fingerprint(str(getattr(coord, "workspace", "")))
        coord._clear_stage_outputs_from(stage)
        patcher = getattr(coord, "patcher", None)
        had_fresh_diff_audit_attr = hasattr(patcher, "enable_recovery_fresh_diff_audit")
        previous_fresh_diff_audit = (
            bool(getattr(patcher, "enable_recovery_fresh_diff_audit", False))
            if had_fresh_diff_audit_attr
            else False
        )
        if had_fresh_diff_audit_attr:
            setattr(patcher, "enable_recovery_fresh_diff_audit", True)
        had_profile_attr = hasattr(patcher, "recovery_execution_profile")
        previous_profile = str(getattr(patcher, "recovery_execution_profile", "normal") or "normal") if had_profile_attr else "normal"
        had_retry_attr = hasattr(patcher, "enable_recovery_retry")
        previous_retry = bool(getattr(patcher, "enable_recovery_retry", True)) if had_retry_attr else True
        if had_profile_attr:
            setattr(patcher, "recovery_execution_profile", execution_profile)
        if had_retry_attr:
            setattr(patcher, "enable_recovery_retry", execution_profile != "compact")
        if getattr(coord, "_official_only_replay_verifier", False):
            try:
                success = self._resume_without_internal_verifier(
                    coord,
                    stage,
                    recovery_context=recovery_context,
                    escalation_level=escalation_level,
                )
                patcher_trace = self._summarize_patcher_trace(coord)
                post_replay_diff = _workspace_diff_fingerprint(str(getattr(coord, "workspace", "")))
                return {
                    "success": success,
                    "scope": scope,
                    "resumed_from": stage,
                    "verifier_mode": "official_only",
                    "internal_verifier_skipped": True,
                    "patcher_trace": patcher_trace,
                    "pre_replay_diff": pre_replay_diff,
                    "post_replay_diff": post_replay_diff,
                    "replay_diff_changed": post_replay_diff.get("diff_digest") != pre_replay_diff.get("diff_digest"),
                }
            finally:
                if had_fresh_diff_audit_attr:
                    setattr(patcher, "enable_recovery_fresh_diff_audit", previous_fresh_diff_audit)
                if had_profile_attr:
                    setattr(patcher, "recovery_execution_profile", previous_profile)
                if had_retry_attr:
                    setattr(patcher, "enable_recovery_retry", previous_retry)
        try:
            success = coord._resume_from(
                stage,
                recovery_context=recovery_context,
                escalation_level=escalation_level,
                deep_verify=False,
            )
            patcher_trace = self._summarize_patcher_trace(coord)
            post_replay_diff = _workspace_diff_fingerprint(str(getattr(coord, "workspace", "")))
            return {
                "success": success,
                "scope": scope,
                "resumed_from": stage,
                "patcher_trace": patcher_trace,
                "pre_replay_diff": pre_replay_diff,
                "post_replay_diff": post_replay_diff,
                "replay_diff_changed": post_replay_diff.get("diff_digest") != pre_replay_diff.get("diff_digest"),
            }
        finally:
            if had_fresh_diff_audit_attr:
                setattr(patcher, "enable_recovery_fresh_diff_audit", previous_fresh_diff_audit)
            if had_profile_attr:
                setattr(patcher, "recovery_execution_profile", previous_profile)
            if had_retry_attr:
                setattr(patcher, "enable_recovery_retry", previous_retry)

    def _exec_selective_replay(
        self,
        coord: CoordinatorProtocol,
        args: dict[str, Any],
        escalation_level: int,
        context_hints: list[str],
    ) -> dict[str, Any]:
        """Path-B MAS-native primitive.

        Re-runs one role while honoring cached upstream outputs for every
        other role. Implementation-wise: `_clear_stage_outputs_from(role)`
        clears the named role plus downstream, so the existing
        `_resume_from(role)` path already serves the intended "cache
        upstream, invalidate downstream" semantics. What is new is the
        typed propagation-object invalidation surface that we record in
        the step result — downstream drivers read it and attach it to the
        `action_executed.state_delta.invalidated_object_stages` field.

        Args expected: `role` (required), `cache_upstream` (bool, default
        True), `context_hint` (optional string merged into the replay
        context).
        """

        role = str(args.get("role", "") or "").strip().lower()
        if not role:
            return {
                "success": False,
                "error": "selective_replay requires role",
            }
        cache_upstream = bool(args.get("cache_upstream", True))
        user_hint = str(args.get("context_hint", ""))
        replay_contract = dict(args.get("replay_contract", {}) or {})
        execution_profile = str(args.get("execution_profile", "") or "").strip().lower() or "normal"
        strong_source_replay = bool(args.get("strong_source_replay", False))
        source_repair_attempt_budget = max(1, int(args.get("source_repair_attempt_budget", 1) or 1))

        all_hints = list(context_hints)
        all_hints.append(
            f"[selective_replay] role={role} cache_upstream={cache_upstream}. "
            "Upstream stage outputs are preserved; only this role's "
            "downstream consumers are considered invalidated."
        )
        if replay_contract:
            all_hints.append(self._format_replay_contract(replay_contract))
        if strong_source_replay:
            all_hints.append(
                "[CFR strong source replay] focused_source_repair profile is active. "
                "Use the strong patcher route when available, keep the frontier unchanged, "
                "make a small source-only repair in a preferred source path when possible, "
                "and revise the source diff from focused validation evidence before broadening scope."
            )
        if user_hint:
            all_hints.append(user_hint)

        recovery_context = self._compose_replay_context(
            coord,
            scope=role,
            context_hints=all_hints,
        )

        # `_clear_stage_outputs_from(role)` drops the named role and its
        # downstream stages. When `cache_upstream=True` that is the correct
        # reset. When `cache_upstream=False` the caller wants a broader
        # invalidation: widen the clear to the first non-cached upstream role.
        patcher = getattr(coord, "patcher", None)
        had_fresh_diff_audit_attr = hasattr(patcher, "enable_recovery_fresh_diff_audit")
        previous_fresh_diff_audit = (
            bool(getattr(patcher, "enable_recovery_fresh_diff_audit", False))
            if had_fresh_diff_audit_attr
            else False
        )
        had_profile_attr = hasattr(patcher, "recovery_execution_profile")
        previous_profile = (
            str(getattr(patcher, "recovery_execution_profile", "normal") or "normal")
            if had_profile_attr
            else "normal"
        )
        had_retry_attr = hasattr(patcher, "enable_recovery_retry")
        previous_retry = bool(getattr(patcher, "enable_recovery_retry", True)) if had_retry_attr else True
        if had_fresh_diff_audit_attr:
            setattr(patcher, "enable_recovery_fresh_diff_audit", True)
        if had_profile_attr:
            setattr(patcher, "recovery_execution_profile", execution_profile)
        if had_retry_attr:
            setattr(patcher, "enable_recovery_retry", True)
        try:
            if not cache_upstream:
                coord._clear_stage_outputs_from("locator")
            else:
                coord._clear_stage_outputs_from(role)

            success = coord._resume_from(
                role,
                recovery_context=recovery_context,
                escalation_level=escalation_level,
                deep_verify=False,
            )
        finally:
            if had_fresh_diff_audit_attr:
                setattr(patcher, "enable_recovery_fresh_diff_audit", previous_fresh_diff_audit)
            if had_profile_attr:
                setattr(patcher, "recovery_execution_profile", previous_profile)
            if had_retry_attr:
                setattr(patcher, "enable_recovery_retry", previous_retry)
        pre_enforcement_patcher_trace = self._summarize_patcher_trace(coord)
        pre_enforcement_contract_audit = self._audit_replay_contract(
            replay_contract,
            pre_enforcement_patcher_trace,
        )
        contract_enforcement = self._enforce_replay_contract(
            coord,
            replay_contract,
            pre_enforcement_contract_audit,
        )
        if contract_enforcement.get("enforced"):
            patcher_trace = self._summarize_patcher_trace(coord)
            contract_audit = self._audit_replay_contract(replay_contract, patcher_trace)
        else:
            patcher_trace = pre_enforcement_patcher_trace
            contract_audit = pre_enforcement_contract_audit
        invalidated_object_stages: list[tuple[str, str]] = []
        invalidated_object_ids: list[str] = []
        chain = list(getattr(coord, "mas_object_chain", None) or [])
        for obj in chain:
            producer = str(getattr(obj, "producer_stage", "") or "").strip().lower()
            consumer = str(getattr(obj, "consumer_stage", "") or "").strip().lower()
            object_id = str(getattr(obj, "object_id", "") or "").strip()
            if producer == role and producer and consumer and object_id:
                if object_id not in invalidated_object_ids:
                    invalidated_object_ids.append(object_id)
                    invalidated_object_stages.append((producer, consumer))
        return {
            "success": success,
            "role": role,
            "cache_upstream": cache_upstream,
            "resumed_from": role,
            "execution_profile": execution_profile,
            "strong_source_replay": strong_source_replay,
            "source_repair_attempt_budget": source_repair_attempt_budget,
            "patcher_trace": patcher_trace,
            "invalidated_object_ids": invalidated_object_ids,
            "invalidated_object_stages": invalidated_object_stages,
            "replay_contract": replay_contract,
            "contract_audit": contract_audit,
            "pre_enforcement_contract_audit": pre_enforcement_contract_audit,
            "contract_enforcement": contract_enforcement,
        }

    def _enforce_replay_contract(
        self,
        coord: CoordinatorProtocol,
        contract: dict[str, Any],
        audit: dict[str, Any],
    ) -> dict[str, Any]:
        if not contract or not audit:
            return {"enforced": False, "reason": "no_contract"}
        policy = str(contract.get("patch_scope_policy", "") or "")
        if policy != "source_only_preferred":
            return {"enforced": False, "reason": "policy_not_enforced", "patch_scope_policy": policy}
        restore_paths = self._contract_list(audit.get("touched_read_only_evidence_paths"))
        restore_paths.extend(self._contract_list(audit.get("changed_test_files")))
        restore_paths.extend(self._contract_list(audit.get("changed_generated_files")))
        restore_paths.extend(self._contract_list(audit.get("changed_other_files")))
        restore_paths = sorted(dict.fromkeys(path for path in restore_paths if path))
        if not restore_paths:
            return {"enforced": False, "reason": "no_forbidden_changes"}
        workspace = str(getattr(coord, "workspace", "") or "")
        if not workspace or not os.path.isdir(workspace):
            return {"enforced": False, "reason": "workspace_unavailable", "restore_paths": restore_paths}
        restored: list[str] = []
        errors: list[dict[str, Any]] = []
        for rel_path in restore_paths:
            proc = subprocess.run(
                ["git", "checkout", "--", rel_path],
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                restored.append(rel_path)
            else:
                errors.append(
                    {
                        "path": rel_path,
                        "returncode": proc.returncode,
                        "stderr": str(proc.stderr or "")[:400],
                    }
                )
        if restored:
            self._refresh_patcher_patch_summary(coord)
        return {
            "enforced": bool(restored),
            "policy": policy,
            "restored_paths": restored,
            "errors": errors,
            "reason": "restored_forbidden_contract_paths" if restored else "restore_failed",
        }

    def _refresh_patcher_patch_summary(self, coord: CoordinatorProtocol) -> None:
        stage_outputs = getattr(coord, "stage_outputs", {}) or {}
        patcher_result = stage_outputs.get("patcher")
        if not isinstance(patcher_result, dict):
            return
        workspace = str(getattr(coord, "workspace", "") or "")
        if not workspace or not os.path.isdir(workspace):
            return
        diff_fp = _workspace_diff_fingerprint(workspace)
        changed_files = self._contract_list(diff_fp.get("changed_files"))
        source_files = [path for path in changed_files if self._is_contract_source_path(path)]
        test_files = [path for path in changed_files if self._is_contract_test_path(path)]
        generated_files = [path for path in changed_files if self._is_contract_generated_path(path)]
        other_files = [
            path
            for path in changed_files
            if path not in source_files and path not in test_files and path not in generated_files
        ]
        patch_summary = dict(patcher_result.get("patch_summary", {}) or {})
        patch_summary["changed_files"] = changed_files
        patch_summary["changed_file_classes"] = {
            "source_files": source_files,
            "test_files": test_files,
            "generated_files": generated_files,
            "other_files": other_files,
            "effective_files": changed_files,
        }
        if test_files or generated_files or other_files:
            patch_summary["target_legitimacy"] = "source_mixed" if source_files else "non_source_only"
        else:
            patch_summary["target_legitimacy"] = "source_only" if source_files else "no_diff"
        patcher_result["patch_summary"] = patch_summary

    @staticmethod
    def _is_contract_test_path(path: str) -> bool:
        text = str(path or "").replace("\\", "/")
        basename = text.rsplit("/", 1)[-1]
        return text.startswith("tests/") or "/tests/" in text or basename.startswith("test_")

    @staticmethod
    def _is_contract_generated_path(path: str) -> bool:
        text = str(path or "").replace("\\", "/")
        return text.startswith("build/") or text.startswith("dist/") or text.startswith("generated/") or "/generated/" in text

    def _is_contract_source_path(self, path: str) -> bool:
        text = str(path or "").replace("\\", "/")
        return text.endswith(".py") and not self._is_contract_test_path(text) and not self._is_contract_generated_path(text)

    def _audit_replay_contract(
        self,
        contract: dict[str, Any],
        patcher_trace: dict[str, Any],
    ) -> dict[str, Any]:
        if not contract:
            return {"contract_present": False, "violations": [], "warnings": []}
        patch_summary = dict(patcher_trace.get("patcher_patch_summary", {}) or {})
        changed_classes = dict(patch_summary.get("changed_file_classes", {}) or {})
        source_files = self._contract_list(changed_classes.get("source_files"))
        test_files = self._contract_list(changed_classes.get("test_files"))
        generated_files = self._contract_list(changed_classes.get("generated_files"))
        other_files = self._contract_list(changed_classes.get("other_files"))
        effective_files = self._contract_list(changed_classes.get("effective_files"))
        preferred = self._contract_list(contract.get("preferred_source_paths"))
        readonly = set(self._contract_list(contract.get("read_only_evidence_paths")))
        violations: list[str] = []
        warnings: list[str] = []
        touched_readonly = sorted(set(test_files + generated_files + other_files + effective_files) & readonly)
        if touched_readonly:
            violations.append("edited_read_only_evidence_path")
        if test_files:
            violations.append("edited_test_path_under_source_only_policy")
        if generated_files:
            violations.append("edited_generated_path_under_source_only_policy")
        if preferred and not (set(source_files) & set(preferred)):
            warnings.append("source_diff_missed_preferred_paths")
        if not source_files:
            violations.append("no_source_diff")
        return {
            "contract_present": True,
            "patch_scope_policy": str(contract.get("patch_scope_policy", "") or ""),
            "preferred_source_paths": preferred,
            "read_only_evidence_paths": sorted(readonly),
            "changed_source_files": source_files,
            "changed_test_files": test_files,
            "changed_generated_files": generated_files,
            "changed_other_files": other_files,
            "touched_read_only_evidence_paths": touched_readonly,
            "violations": sorted(dict.fromkeys(violations)),
            "warnings": sorted(dict.fromkeys(warnings)),
            "satisfied": not violations,
        }

    @staticmethod
    def _contract_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            raw = list(value)
        else:
            raw = [value]
        ordered: list[str] = []
        seen: set[str] = set()
        for item in raw:
            text = str(item or "").strip().replace("\\", "/")
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _format_replay_contract(self, contract: dict[str, Any]) -> str:
        preferred = self._join_contract_values(contract.get("preferred_source_paths"))
        readonly = self._join_contract_values(contract.get("read_only_evidence_paths"))
        failing_tests = self._join_contract_values(contract.get("failing_tests"))
        invalidated = self._join_contract_values(contract.get("must_not_consume_object_ids") or contract.get("invalidated_object_ids"))
        preserved = self._join_contract_values(contract.get("must_preserve_object_ids") or contract.get("clean_upstream_object_ids"))
        negatives = self._join_contract_values(contract.get("negative_facts"))
        criteria = self._join_contract_values(contract.get("success_criteria"), separator="; ")
        forbidden = self._join_contract_values(contract.get("forbidden_edit_patterns"))
        verifier_excerpt = str(contract.get("verifier_excerpt", "") or "").strip()[:1000]
        lines = [
            "[CFR replay contract]",
            f"version={contract.get('contract_version', 'unknown')}",
            f"replay_start_stage={contract.get('replay_start_stage', '')}",
            f"patch_scope_policy={contract.get('patch_scope_policy', 'source_only_preferred')}",
            f"preferred_source_paths={preferred or 'none'}",
            f"read_only_evidence_paths={readonly or 'none'}",
            f"forbidden_edit_patterns={forbidden or 'none'}",
            f"failing_tests={failing_tests or 'none'}",
            f"must_not_consume_object_ids={invalidated or 'none'}",
            f"must_preserve_object_ids={preserved or 'none'}",
            f"negative_facts={negatives or 'none'}",
            f"success_criteria={criteria or 'produce a source diff and verify it'}",
            "Treat read-only evidence/test paths as diagnostic evidence only; do not edit them unless the oracle issue explicitly requires a test fixture update.",
            "If preferred_source_paths are available, make the first concrete fix in one of those source files before considering broader search.",
        ]
        if verifier_excerpt:
            lines.append(f"verifier_excerpt={verifier_excerpt}")
        return "\n".join(lines)

    @staticmethod
    def _join_contract_values(value: Any, *, separator: str = ", ") -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return separator.join(str(item) for item in value if str(item).strip())
        return str(value).strip()

    def _summarize_patcher_trace(self, coord: CoordinatorProtocol) -> dict[str, Any]:
        stage_outputs = getattr(coord, "stage_outputs", {}) or {}
        patcher_result = dict(stage_outputs.get("patcher", {}) or {})
        commands = [dict(item) for item in patcher_result.get("commands", []) or [] if isinstance(item, dict)]
        messages = [
            dict(item)
            for item in patcher_result.get("messages", []) or []
            if isinstance(item, dict)
        ]
        planner_messages = [
            dict(item)
            for item in patcher_result.get("planner_messages", []) or []
            if isinstance(item, dict)
        ]
        command_texts = [str(item.get("command", "")).strip() for item in commands]
        write_commands = [cmd for cmd in command_texts if self._looks_like_write_command(cmd)]
        validation_commands = [cmd for cmd in command_texts if self._looks_like_validation_command(cmd)]
        readonly_commands = [cmd for cmd in command_texts if self._looks_like_readonly_probe_command(cmd)]
        return {
            "patcher_success": bool(patcher_result.get("success", False)),
            "patcher_command_count": len(commands),
            "patcher_write_command_count": len(write_commands),
            "patcher_validation_command_count": len(validation_commands),
            "patcher_readonly_command_count": len(readonly_commands),
            "patcher_patch_present": bool(str(patcher_result.get("patch", "")).strip()),
            "patcher_patch_summary": dict(patcher_result.get("patch_summary") or {}),
            "patcher_error": str(patcher_result.get("error", "") or ""),
            "patcher_planner_error": str(patcher_result.get("planner_error", "") or ""),
            "patcher_implementer_error": str(patcher_result.get("implementer_error", "") or ""),
            "patcher_infrastructure_error": bool(patcher_result.get("infrastructure_error", False)),
            "patcher_retry_used": bool(patcher_result.get("retry_used", False)),
            "patcher_retry_reason": str(patcher_result.get("retry_reason", "") or ""),
            "patcher_patch_intent_plan_retry_used": bool(
                patcher_result.get("patch_intent_plan_retry_used", False)
            ),
            "patcher_patch_intent_plan_audit": dict(
                patcher_result.get("patch_intent_plan_audit") or {}
            ),
            "patcher_message_count": len(messages),
            "patcher_planner_message_count": len(planner_messages),
            "patcher_last_messages_excerpt": [
                {
                    "role": str(item.get("role", "")),
                    "content": str(item.get("content", ""))[:500],
                }
                for item in messages[-4:]
            ],
            "patcher_last_planner_messages_excerpt": [
                {
                    "role": str(item.get("role", "")),
                    "content": str(item.get("content", ""))[:500],
                }
                for item in planner_messages[-2:]
            ],
            "patcher_commands_excerpt": [
                {
                    "command": item.get("command", ""),
                    "returncode": item.get("returncode"),
                }
                for item in commands[:8]
            ],
        }

    def _looks_like_write_command(self, command: str) -> bool:
        return looks_like_write_command(command)

    def _looks_like_validation_command(self, command: str) -> bool:
        return looks_like_validation_command(command)

    def _looks_like_readonly_probe_command(self, command: str) -> bool:
        return looks_like_readonly_probe_command(command)

    def _compose_replay_context(
        self,
        coord: CoordinatorProtocol,
        *,
        scope: str,
        context_hints: list[str],
    ) -> str:
        base_context = "\n".join(hint for hint in context_hints if hint)
        failed_state = getattr(coord, "_recovery_failed_state", None)
        diagnosis_mode = str(getattr(coord, "_recovery_diagnosis_mode", "") or "").strip()
        if diagnosis_mode != "contaminated_post_patch_v1" or failed_state is None:
            return base_context

        observation = dict(getattr(failed_state, "metadata", {}).get("failure_observation", {}) or {})
        raw_suspect_paths = [
            str(path).replace("\\", "/")
            for path in list(getattr(failed_state, "metadata", {}).get("touched_paths", []) or [])
            if str(path).strip()
        ]
        suspect_paths = existing_repo_source_paths(
            raw_suspect_paths,
            str(getattr(coord, "workspace", "") or ""),
        ) or canonical_source_paths(raw_suspect_paths)
        failing_tests = [str(item) for item in observation.get("failing_tests", []) or [] if str(item).strip()]
        test_command = str(observation.get("test_command", "") or "").strip()

        instructions = [
            "This is a contaminated recovery run, not a clean-start solve.",
            "Rebuild the patch from the restored failed state instead of re-planning the whole task.",
            "Prioritize editing the suspect source path first.",
            "Do not prioritize build/, dist/, generated/, copied build outputs, or pure test-only edits.",
            "If you cannot identify a concrete source-file fix, do not pretend the repair is complete.",
        ]
        if "locator" in scope:
            instructions.append(
                "This replay has expanded to locator+patcher scope: refresh localization from the failing tests, then edit source files selected by that refreshed localization."
            )
        if failing_tests:
            instructions.append(f"Current failing tests: {', '.join(failing_tests[:3])}")
        if test_command:
            instructions.append(f"Focused verification command: {test_command}")
            instructions.append(
                "After a source edit, run this exact focused verification command before declaring the repair complete; if it still fails, use the failure text to revise the source diff."
            )
        if suspect_paths:
            instructions.append(f"Preferred suspect source paths: {', '.join(suspect_paths[:5])}")
        if scope:
            instructions.append(f"Replay scope: {scope}")
        instructions.extend(
            [
                "A successful recovery attempt should show both a source diff and a focused validation command in the command trace.",
                "Avoid writing throwaway scripts in /tmp unless they directly support a source edit and focused validation.",
            ]
        )
        if base_context:
            instructions.append(base_context)
        return "\n".join(instructions)

    def _resume_without_internal_verifier(
        self,
        coord: CoordinatorProtocol,
        stage: str,
        *,
        recovery_context: str,
        escalation_level: int,
    ) -> bool:
        """Replay locator/patcher work, but leave pass/fail to official tests.

        The recovery benchmark already runs the official fail-to-pass and
        oracle commands after each program.  Skipping the live LLM verifier here
        keeps the online method from paying for a second, noisy verification
        loop inside every candidate replay.
        """

        stage = stage.lower().strip()
        if stage == "full":
            stage = "locator"
        if stage == "locator":
            run_locator = getattr(coord, "_run_locator", None)
            if not callable(run_locator):
                return False
            locate_result = run_locator(
                recovery_context=recovery_context,
                escalation_level=escalation_level,
            )
            if not locator_result_allows_recovery_replay(
                locate_result,
                recovery_context=recovery_context,
            ):
                return False
            if locator_result_is_recovery_best_effort(
                locate_result,
                recovery_context=recovery_context,
            ):
                accept_best_effort = getattr(coord, "_accept_locator_best_effort_for_recovery", None)
                if callable(accept_best_effort):
                    accept_best_effort(locate_result)
                else:
                    locate_result["accepted_for_recovery_replay"] = True
                    stage_outputs = getattr(coord, "stage_outputs", None)
                    if isinstance(stage_outputs, dict) and isinstance(stage_outputs.get("locator"), dict):
                        stage_outputs["locator"]["accepted_for_recovery_replay"] = True
            stage = "patcher"
        if stage == "patcher":
            run_patcher = getattr(coord, "_run_patcher", None)
            if not callable(run_patcher):
                return False
            patch_result = run_patcher(
                recovery_context=recovery_context,
                escalation_level=escalation_level,
            )
            return bool(patch_result.get("success"))
        if stage == "verifier":
            return True
        return False

    def _resolve_fact_reference(
        self,
        coord: CoordinatorProtocol,
        fact_ref: str,
    ) -> tuple[str, Any, str]:
        token = str(fact_ref or "").strip()
        if not token:
            return "", None, ""

        if token.startswith("fact:"):
            token = token.split(":", 1)[1].strip()

        shared = coord.shared_facts.get(token)
        if isinstance(shared, dict):
            node_id = str(shared.get("node_id", "")).strip()
            node = coord.recorder.graph.get_node(node_id) if node_id else None
            if node is not None:
                return node_id, node, token

        node = coord.recorder.graph.get_node(token)
        if node is not None:
            for fkey, fval in coord.shared_facts.items():
                if isinstance(fval, dict) and str(fval.get("node_id", "")).strip() == token:
                    return token, node, fkey
            return token, node, token

        for fkey, fval in coord.shared_facts.items():
            if not isinstance(fval, dict):
                continue
            node_id = str(fval.get("node_id", "")).strip()
            if token == fkey and node_id:
                node = coord.recorder.graph.get_node(node_id)
                if node is not None:
                    return node_id, node, fkey
        return "", None, token
