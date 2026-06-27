"""Recovery planning for MASGuard diagnoses."""

from __future__ import annotations

from recoveragent.models import Diagnosis, EvidenceBundle, RecoveryPlan


def plan_recovery(bundle: EvidenceBundle, diagnosis: Diagnosis) -> RecoveryPlan:
    """Build a concrete recovery-control plan without executing a repair."""

    failure_type = diagnosis.failure_type
    if failure_type == "environment_or_tool_failure":
        return RecoveryPlan(
            action="rerun-targeted-tests-after-environment-preflight",
            steps=[
                "Run a dependency and command preflight before asking the repair agent to mutate code.",
                "Re-run the smallest failing test target after the environment is valid.",
                "Resume repair only if the target failure is reproduced without infrastructure blockers.",
            ],
            expected_user="researcher_or_agent_developer",
            scope_note="Preflight and target replay are validation steps; they are not repair success.",
        )
    if failure_type == "validation_target_failure":
        return RecoveryPlan(
            action="rebuild-test-target-then-rerun",
            steps=[
                "Extract valid test targets from the repository and failing log.",
                "Replace invalid target commands in the next agent prompt.",
                "Run target collection before allowing source edits.",
            ],
            expected_user="researcher_or_agent_developer",
            scope_note="A corrected test target authorizes later recovery; it is not itself recovery credit.",
        )
    if failure_type == "fault_localization_failure":
        files = ", ".join(bundle.stack_trace_files[:3]) or "the stack-trace files"
        return RecoveryPlan(
            action="rollback-and-relocalize-from-evidence",
            steps=[
                "Rollback the invalid patch or checkpoint before the misleading edit.",
                f"Build a new evidence bundle centered on {files}.",
                "Ask the repair agent to inspect the implicated source files before patching.",
                "Run targeted tests before broad validation.",
            ],
            expected_user="LLM_repair_agent_builder",
            scope_note="MASGuard plans the recovery action; a later agent/harness run must produce and validate any patch.",
        )
    if failure_type == "no_effective_patch":
        files = ", ".join(bundle.stack_trace_files[:3] or bundle.repo_files[:3]) or "the source files implicated by validation"
        return RecoveryPlan(
            action="resume-patcher-with-source-contract-evidence",
            steps=[
                "Keep the recovered localization evidence instead of restarting the whole MAS.",
                f"Build a patcher prompt centered on {files} and the failing validation contract.",
                "Require a fresh source diff before claiming progress; abstention or a no-op command is not a repair.",
                "Run the focused fail-to-pass validation before broad validation.",
            ],
            expected_user="LLM_repair_agent_builder",
            scope_note="MASGuard diagnoses a no-diff patcher failure; the MAS must still synthesize and validate the recovery patch.",
        )
    if failure_type == "patch_generation_failure":
        return RecoveryPlan(
            action="reject-test-only-patch-and-repair-source",
            steps=[
                "Reject the candidate patch because it modifies tests without a source-level justification.",
                "Regenerate the prompt with source spans, failing assertions, and patch constraints.",
                "Require a source diff and syntax check before oracle validation.",
            ],
            expected_user="LLM_repair_agent_builder",
            scope_note="Patch rejection prevents unsafe credit; it does not count as a solved repair task.",
        )
    if failure_type == "patch_semantic_failure":
        files = ", ".join(bundle.stack_trace_files[:3]) or "the failing source and test files"
        return RecoveryPlan(
            action="resume-with-contract-branch-evidence",
            steps=[
                "Keep the source-level localization but do not trust the previous patch as complete.",
                f"Reobserve the failing contract branches named by {files}.",
                "Split the validation feedback into independent obligations, such as return value and warning/logging behavior.",
                "Ask the repair agent to extend or replace the source patch so every failing obligation is covered.",
                "Run the same targeted validation command before broad validation.",
            ],
            expected_user="LLM_repair_agent_builder",
            scope_note="MASGuard identifies an incomplete semantic patch; the MAS must produce and validate the repaired patch.",
        )
    if failure_type == "context_drift_or_repeated_mistake":
        return RecoveryPlan(
            action="resume-from-checkpoint-with-condensed-evidence",
            steps=[
                "Stop the repeated failing command loop.",
                "Resume from the last checkpoint before repeated failure.",
                "Provide a condensed evidence bundle with explicit do-not-repeat constraints.",
            ],
            expected_user="agent_runtime_developer",
            scope_note="Checkpoint resume must be validated by a separate repair execution.",
        )
    return RecoveryPlan(
        action="stop-and-report",
        steps=[
            "Do not authorize a source edit from the current evidence.",
            "Emit a failure report with logs, touched files, and missing evidence.",
            "Request additional trajectory or repository evidence before recovery.",
        ],
        expected_user="researcher_or_developer",
        scope_note="Fail-closed reporting is a safety behavior, not repair success.",
    )
