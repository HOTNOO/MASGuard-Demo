"""Recovery prompt export for plugging MASGuard into another MAS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_recovery_prompt(report: dict[str, Any]) -> str:
    diagnosis = report["diagnosis"]
    plan = report["recovery_plan"]
    bundle = report["evidence_bundle"]
    graph = report.get("evidence_graph", {})
    evidence_lines = "\n".join(f"- {item}" for item in diagnosis.get("evidence", [])) or "- no evidence cited"
    steps = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(plan.get("steps", [])))
    stack_files = ", ".join(bundle.get("stack_trace_files", [])[:8]) or "none"
    touched_files = ", ".join(bundle.get("touched_files", [])[:8]) or "none"

    return f"""# MASGuard Recovery Prompt

You are resuming a failed repository-level repair attempt. Use the evidence
below. Do not repeat the previous patch direction unless the evidence justifies
it.

## Diagnosis

- Failure type: `{diagnosis['failure_type']}`
- Responsible stage: `{diagnosis['responsible_stage']}`
- Confidence: `{diagnosis['confidence']}`
- Rationale: {diagnosis['rationale']}

## Evidence

{evidence_lines}

## Repository Signals

- Stack-trace files: {stack_files}
- Patch-touched files: {touched_files}
- Evidence graph nodes: {len(graph.get('nodes', []))}
- Evidence graph edges: {len(graph.get('edges', []))}

## Recovery Action

Action: `{plan['action']}`

{steps}

## Patch Constraints

- Roll back the previous invalid patch before generating a new patch.
- Prefer source-level changes over modifying tests.
- Inspect the files named by failing validation evidence.
- Run the targeted validation command before broad validation.

## Recovery Scope

{plan['scope_note']}
"""


def write_recovery_prompt(report_path: Path, output_path: Path) -> str:
    prompt = build_recovery_prompt(load_report(report_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    return prompt
