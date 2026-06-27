# MASGuard Integration Guide

This file is the lower-level integration reference for using MASGuard as a
post-failure recovery plugin in a multi-agent software-repair system.

MASGuard can be used as a complete command-line tool or as a plugin in an
existing software-engineering agent pipeline.

In the intended MAS setting, MASGuard is a post-failure recovery plugin. It
does not replace locator/planner/patcher/verifier agents. It is called after
the MAS has produced a failed repair attempt.

## Choosing An Integration Layer

MASGuard ships three implementation layers:

```text
src/recoveragent/  generic CLI sidecar and Python package
bcmr_swe/          real SWE/MAS backend used by the online Django demo and records
swe_mas/           reusable MAS substrate and historical agent components
```

Most external users only need the `src/recoveragent` contract: export a failed
run, call `masguard analyze-run`, and feed the generated recovery prompt back
to their own MAS patcher. Users who want to inspect or rerun the bundled SWE
case should use `examples/user_mas/swe_live_mas.py`, which calls the
`bcmr_swe` backend. Users building a deeper MAS integration can either adapt
their own controller or reuse components from `swe_mas`.

You can print this map from the installed CLI:

```bash
masguard backend-info
```

## Integration Contract

An upstream repair agent only needs to export four files after a failed attempt:

```text
run_dir/
  trajectory.json
  repo/
  logs/failing.log
  patches/failed.patch
```

Minimal `trajectory.json`:

```json
{
  "issue": "Natural language issue or task description.",
  "tool_calls": [
    {
      "role": "locator",
      "command": "python -m pytest tests/test_x.py::test_y",
      "status": "failed",
      "summary": "Short failure summary."
    }
  ]
}
```

## CLI Adapter

The most stable integration is the CLI:

```bash
masguard analyze \
  --trajectory "$RUN_DIR/trajectory.json" \
  --repo "$RUN_DIR/repo" \
  --log "$RUN_DIR/logs/failing.log" \
  --patch "$RUN_DIR/patches/failed.patch" \
  --output "$RUN_DIR/masguard_report.json" \
  --html "$RUN_DIR/masguard_report.html"
```

The upstream controller can then read:

- `diagnosis.failure_type`
- `diagnosis.confidence`
- `diagnosis.evidence`
- `recovery_plan.action`
- `recovery_plan.steps`
- `scope_note`

## Python Adapter

```python
from pathlib import Path

from recoveragent.diagnosis import diagnose
from recoveragent.evidence import extract_evidence
from recoveragent.planner import plan_recovery
from recoveragent.report import build_report

run_dir = Path("/path/to/run")
bundle = extract_evidence(
    trajectory_path=run_dir / "trajectory.json",
    repo_path=run_dir / "repo",
    log_path=run_dir / "logs/failing.log",
    patch_path=run_dir / "patches/failed.patch",
)
diagnosis = diagnose(bundle)
plan = plan_recovery(bundle, diagnosis)
report = build_report(bundle, diagnosis, plan)
```

## Where It Sits In A Repair Agent

Recommended control loop:

```text
agent attempts repair
-> tests/build fail
-> export trajectory/log/patch/repo snapshot
-> MASGuard analyzes failure
-> controller selects:
   - rollback and relocalize
   - rerun targeted tests after preflight
   - rebuild invalid test target
   - resume from checkpoint with condensed evidence
   - reject unsafe test-only patch
   - fail closed and report
-> upstream agent performs the next repair attempt
```

MASGuard does not need to replace the upstream agent. It can be inserted
between validation failure and the next repair attempt.

The small online integration example is:

```bash
PYTHONPATH=src python examples/user_mas/online_minimas.py --help
```

It demonstrates:

```text
online user MAS calls a provider and fails
-> MAS exports repo/trajectory/log/patch artifacts
-> MASGuard diagnoses the failed run
-> the same online MAS calls the provider again with recovery_prompt.md
-> recovered source patch validates
```

The bundled offline executable example is:

```bash
masguard mas-plugin-demo --output-dir demo_outputs/mas_plugin_run
```

It demonstrates:

```text
mini-MAS validation fails
-> baseline MAS applies wrong test-only patch
-> validation still fails
-> MASGuard diagnoses fault localization failure
-> recovered MAS rolls back the bad patch and applies a source patch
-> validation passes
```

The real SWE integration example is:

```bash
PYTHONPATH=.:src python examples/user_mas/swe_live_mas.py --help
```

It demonstrates:

```text
bcmr_swe online MAS runs django__django-13321 and fails validation
-> failed run is exported to the MASGuard contract
-> masguard analyze-run writes diagnosis and recovery_prompt.md
-> online MAS recovery patcher consumes the prompt
-> Django's own tests validate the recovered source patch
```

## Example System Mappings

| Upstream System | Export Step | MASGuard Use |
| --- | --- | --- |
| SWE-agent-like CLI | save run log, patch, checkout | post-failure diagnostic hook |
| OpenHands/OpenDevin-style agent | save event stream and workspace | sidecar artifact analyzer |
| AutoCodeRover-like repair pipeline | save localization, patch, test output | recovery action selector |
| CI repair service | save failed branch, log, diff | CI artifact and next-action report |
| IDE coding agent | save local failing run | side-panel failure diagnosis |

## Comparison Designs

For a Tool Demo, compare behavior rather than claiming a new benchmark result
from the illustrative suite:

| Policy | Behavior To Show |
| --- | --- |
| Observed failed run | the original agent stops after failed validation |
| Naive retry | likely repeats wrong context, invalid target, or environment blocker |
| Reflection-only retry | asks the same agent to reflect without structured repository evidence |
| MASGuard-guided recovery | chooses a concrete recovery action from evidence |

For a full experiment, use fixed failed trajectories and report:

- recovered oracle successes;
- repeated-failure rate;
- action adherence;
- protocol/runtime blockers;
- calls and tokens;
- cases where MASGuard fails closed.

## API Boundary

MASGuard's deterministic core is the integration contract. The MASGuard CLI
`--llm` flag is only an optional explanation layer:

```bash
OPENAI_API_KEY=... masguard analyze ... --llm
```

Do not make an upstream system depend on the optional text explanation. Depend
on the deterministic JSON fields.

The upstream MAS can still be fully online. The included online example uses
`examples/user_mas/online_minimas.py` to call an OpenAI-compatible provider for
both the failed baseline patcher decision and the recovered patcher decision.
