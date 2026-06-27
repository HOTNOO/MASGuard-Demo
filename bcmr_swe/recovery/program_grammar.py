"""Recovery program grammar: loose constraints for LLM-driven composition.

Design philosophy
-----------------
The LLM is the program composer — it decides what sequence of operators to
use based on the specific failure context.  The grammar is NOT a skeleton
generator; it is a **lightweight validator** that rejects clearly invalid
programs while allowing the LLM maximum creative freedom.

Constraints (deliberately loose)
--------------------------------
1. Programs consist of 1-5 steps.
2. Each step must be one of the five atomic operators.
3. Operator arguments must use valid parameter names and value types.
4. At least one observable state-changing operator must appear in the
   program: either ROLLBACK or REPLAY.  A pure ROLLBACK is valid because
   the benchmark runs the official test/oracle command after the program.
5. ESCALATE cannot be the very last step (it only changes config without
   producing an observable outcome).
6. REVOKE/INSPECT followed only by REPLAY(verifier) is invalid: verifier
   replay can observe a state but cannot repair the workspace.

There are NO fixed transition rules.  The LLM is free to compose any
sequence that satisfies the above.  For example, these are all valid:

    ROLLBACK(post_patch)
    REPLAY(patcher+verifier)
    INSPECT → REPLAY → INSPECT → REPLAY       (iterative)
    REVOKE → ROLLBACK → ESCALATE → REPLAY
    INSPECT → REVOKE → REPLAY → INSPECT → REPLAY
    ROLLBACK → REPLAY

Operators
---------
- INSPECT(target, depth)        — diagnose without changing state
- REVOKE(fact_id)               — quarantine a polluted shared fact
- ROLLBACK(checkpoint_id)       — restore workspace to checkpoint
- REPLAY(scope, context_hint)   — re-run pipeline stages
- ESCALATE(scope, strategy)     — upgrade a stage's capability
"""

from __future__ import annotations

from bcmr_swe.types import OpType, RecoveryProgram

MAX_PROGRAM_STEPS = 5

VALID_ARGS: dict[OpType, dict[str, set[str] | None]] = {
    OpType.INSPECT: {
        "target": {"patch", "localization", "test_output"},
        "depth": {"quick", "deep"},
    },
    OpType.REVOKE: {
        "fact_id": None,
    },
    OpType.ROLLBACK: {
        "checkpoint_id": None,
    },
    OpType.REPLAY: {
        "scope": {"locator", "patcher", "verifier", "patcher+verifier", "locator+patcher+verifier", "full"},
        "context_hint": None,
    },
    OpType.ESCALATE: {
        "scope": {"locator", "patcher", "verifier"},
        "strategy": {"more_iterations", "broader_search", "stronger_prompt"},
    },
}


def validate_program(program: RecoveryProgram) -> tuple[bool, str]:
    """Check whether a program satisfies the loose grammar constraints.

    Returns ``(is_valid, reason)``.  Most sequences the LLM produces will
    be valid — the validator only catches clearly broken programs.
    """
    steps = program.steps
    if not steps:
        return False, "empty program"
    if len(steps) > MAX_PROGRAM_STEPS:
        return False, f"too many steps ({len(steps)} > {MAX_PROGRAM_STEPS})"

    for i, step in enumerate(steps):
        if step.op not in OpType.__members__.values():
            return False, f"step {i}: unknown operator {step.op}"

    has_observable_effect = any(s.op in {OpType.ROLLBACK, OpType.REPLAY} for s in steps)
    if not has_observable_effect:
        return False, "program must contain at least one ROLLBACK or REPLAY step"

    if steps[-1].op == OpType.ESCALATE:
        return False, "ESCALATE cannot be the last step (it needs a subsequent REPLAY)"

    has_rollback = any(s.op == OpType.ROLLBACK for s in steps)
    replay_scopes = [
        str(s.args.get("scope", "")).strip().lower()
        for s in steps
        if s.op == OpType.REPLAY
    ]
    if replay_scopes and not has_rollback and set(replay_scopes) <= {"verifier"}:
        return False, "verifier-only replay without rollback cannot repair the workspace"

    return True, "ok"


def validate_args(op: OpType, args: dict) -> tuple[bool, str]:
    """Check whether operator arguments use valid parameter names/values.

    This is advisory — programs with unexpected arguments are still
    executed, but a warning is logged.
    """
    spec = VALID_ARGS.get(op, {})
    for key, allowed_values in spec.items():
        if allowed_values is not None and key in args:
            if args[key] not in allowed_values and not str(args[key]).startswith("fact:"):
                return False, f"{op.value}: {key}={args[key]} not in {allowed_values}"
    return True, "ok"


def format_grammar_for_prompt() -> str:
    """Produce operator documentation for inclusion in LLM prompts.

    This tells the LLM what tools are available and what arguments they
    accept, but does NOT prescribe a fixed sequence.
    """
    return """## Recovery Operators

You have five atomic operators.  Compose them into a recovery program
(1-5 steps).  You decide the sequence — there are no fixed templates.
The only hard rules are: (a) include at least one ROLLBACK or REPLAY, and
(b) don't end with ESCALATE (it needs a REPLAY after it).

### INSPECT(target, depth)
Diagnose a suspect artifact WITHOUT changing any state.
- target: "patch" | "localization" | "test_output" | "fact:<id>"
- depth: "quick" (run tests only) | "deep" (analysis + tests + cross-check)
- Cost: ~200-800 tokens
- Use when: you want to confirm the root cause before taking expensive action.

### REVOKE(fact_id)
Quarantine a polluted shared fact so it stops propagating.
- fact_id: the SharedFact node ID to isolate (or omit to auto-detect the first conflicted fact)
- Cost: ~0 tokens (instant)
- Use when: a specific intermediate conclusion is wrong and needs removal.

### ROLLBACK(checkpoint_id)
Restore the workspace (files, patches) to a prior checkpoint.
- checkpoint_id: target checkpoint (or omit for the latest available)
- Cost: ~0 tokens, but discards all progress since that checkpoint
- Use when: the workspace is too polluted to repair in-place.
- Can be a complete one-step program if it restores a known healthy anchor;
  the benchmark will run official tests after the program.

### REPLAY(scope, context_hint)
Re-run pipeline stages from the given starting point.
- scope: "locator" | "patcher" | "verifier" | "patcher+verifier" | "full"
- context_hint: free-text guidance for the replayed agents (e.g. "previous localization was wrong, search more broadly")
- Cost: ~1000-5000 tokens depending on scope
- Use when: you need to re-execute stages with corrected state/context.
- You can use REPLAY more than once (e.g. replay patcher, inspect result, replay verifier).
- REPLAY(verifier) only checks; it does not repair files.  Use it after a
  ROLLBACK to a healthy checkpoint, not after REVOKE/INSPECT alone.

### ESCALATE(scope, strategy)
Upgrade a stage's execution capability WITHOUT running it yet.
- scope: "locator" | "patcher" | "verifier"
- strategy: "more_iterations" | "broader_search" | "stronger_prompt"
- Cost: 0 tokens (config change only), but the subsequent REPLAY will be more expensive
- Use when: the failure is due to insufficient capability, not wrong information.
- MUST be followed by a REPLAY in the program.

### Example Programs

Minimal anchor restore:
  [{"op": "ROLLBACK", "args": {"checkpoint_id": "ckpt_post_patch"}}]

Simple replay:
  [{"op": "REPLAY", "args": {"scope": "patcher+verifier", "context_hint": "..."}}]

Diagnose-then-fix:
  [{"op": "INSPECT", "args": {"target": "localization", "depth": "deep"}},
   {"op": "ROLLBACK", "args": {"checkpoint_id": "ckpt_xxx"}},
   {"op": "REPLAY", "args": {"scope": "full", "context_hint": "localization was wrong"}}]

Iterative recovery:
  [{"op": "REVOKE", "args": {"fact_id": "fact_xxx"}},
   {"op": "REPLAY", "args": {"scope": "patcher+verifier", "context_hint": "removed bad fact"}},
   {"op": "INSPECT", "args": {"target": "test_output", "depth": "quick"}},
   {"op": "ESCALATE", "args": {"scope": "patcher", "strategy": "more_iterations"}},
   {"op": "REPLAY", "args": {"scope": "patcher+verifier", "context_hint": "escalated after first attempt"}}]

Clean-slate with escalation:
  [{"op": "ROLLBACK", "args": {"checkpoint_id": "ckpt_xxx"}},
   {"op": "ESCALATE", "args": {"scope": "locator", "strategy": "broader_search"}},
   {"op": "REPLAY", "args": {"scope": "full", "context_hint": "starting fresh with stronger locator"}}]
"""
