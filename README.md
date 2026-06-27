# MASGuard

MASGuard is a post-failure recovery plugin for LLM-based multi-agent software
repair systems. It is called after an existing MAS has tried a repair and the
repository validation still fails. MASGuard reads the failed trajectory,
repository snapshot, validation log, and failed patch, then produces a
structured diagnosis and a recovery prompt for the same MAS patcher to resume
from.

The command-line entry point is `masguard`. The internal Python package is
named `recoveragent` for compatibility with earlier experiments.

## What This Repository Contains

This package is intentionally split into three layers.

```text
src/recoveragent/
  Installable MASGuard CLI sidecar. This is the generic integration layer used
  by any MAS that can export a failed-run directory.

bcmr_swe/
  SWE/MAS research backend. The real online SWE demonstration and the 69-case
  experiment records are built from this backend, not from a toy script.

swe_mas/
  Multi-agent repair substrate and reusable MAS components used by the broader
  implementation history.
```

The split keeps the user-facing plugin small while still shipping the backend
needed to inspect how the real SWE case and experiment records were produced.
Run the architecture command after installation:

```bash
masguard backend-info
```

For a fuller map, see [docs/architecture.md](docs/architecture.md).

## Install

```bash
cd MASGuard-Demo
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
masguard --help
masguard backend-info
```

## Failed-Run Contract

MASGuard uses a file-based contract so it can attach to different repair
agents without depending on their internal event system.

```text
failed_run/
  repo/                 # repository snapshot after the failed MAS attempt
  trajectory.json       # MAS roles, commands, tool calls, and statuses
  logs/failing.log      # failing test/build output
  patches/failed.patch  # failed patch produced by the MAS
  recoveragent/         # MASGuard outputs
```

Create the layout:

```bash
masguard init --run-dir runs/my_failed_run
```

After the upstream MAS exports the four required inputs, analyze the run:

```bash
masguard analyze-run --run-dir runs/my_failed_run
```

MASGuard writes:

```text
runs/my_failed_run/recoveragent/report.json
runs/my_failed_run/recoveragent/report.html
runs/my_failed_run/recoveragent/recovery_prompt.md
```

The upstream MAS patcher consumes `recovery_prompt.md` and produces the next
patch. MASGuard can then apply and validate that patch for accounting:

```bash
masguard apply-patch --run-dir runs/my_failed_run --patch path/to/mas_recovered.patch
masguard validate --run-dir runs/my_failed_run --command "python tests/test_widget.py"
```

`apply-patch` and `validate` record what happened. Patch synthesis remains the
responsibility of the user's MAS/LLM patcher.

## Main Commands

```bash
masguard init
masguard analyze-run
masguard analyze
masguard prompt
masguard apply-patch
masguard validate
masguard suite
masguard backend-info
masguard mas-plugin-demo
```

Lower-level path-based analysis is also available:

```bash
masguard analyze \
  --trajectory trajectory.json \
  --repo repo \
  --log logs/failing.log \
  --patch patches/failed.patch \
  --output recoveragent/report.json \
  --html recoveragent/report.html
```

## Integration Flow

```text
existing MAS attempts repair
-> tests/build fail
-> MAS exports trajectory, failed patch, validation log, and repo snapshot
-> masguard analyze-run constructs evidence and recovery instructions
-> same MAS patcher resumes with recoveragent/recovery_prompt.md
-> masguard apply-patch and validate record the recovered outcome
```

The controller can consume these stable JSON fields:

```text
diagnosis.failure_type
diagnosis.confidence
diagnosis.evidence
recovery_plan.action
recovery_plan.steps
recovery_plan.scope_note
evidence_graph.nodes
evidence_graph.edges
```

See [docs/integration.md](docs/integration.md) for CLI and Python adapters.

## Runnable Examples

Run a local MAS-plugin workflow without an API:

```bash
masguard mas-plugin-demo --output-dir demo_outputs/mas_plugin_run
```

Run the multi-case deterministic suite:

```bash
masguard suite \
  --cases-dir examples \
  --output demo_outputs/suite_report.json \
  --markdown demo_outputs/comparison.md \
  --html demo_outputs/demo_dashboard.html
```

Inspect the real SWE/MAS driver:

```bash
PYTHONPATH=.:src python examples/user_mas/swe_live_mas.py --help
```

Run the code-level integration example:

```bash
PYTHONPATH=.:src python examples/integration_adapter/minimal_mas_with_recoveragent.py
```

## Real SWE Demonstration

The real online demonstration uses `django__django-13321`. The baseline MAS is
run by `bcmr_swe.experiments.mas_recovery_run_clean_start_baseline` through
`examples/user_mas/swe_live_mas.py`. It performs live provider calls, writes a
failed source patch, exports the failed-run directory, and then resumes the MAS
patcher with MASGuard's recovery prompt.

Inspect the command surface for the real SWE driver:

```bash
PYTHONPATH=.:src python examples/user_mas/swe_live_mas.py --help
```

The public tool repository intentionally does not include local API files,
recording shell scripts, or generated demo outputs. The recording walkthrough is
packaged separately in the showcase archive, and the reproducibility scripts are
packaged separately in the artifact archive.

## Reproducibility Package

The separate archive package includes the source snapshot, expected SWE case
files, 69-instance result records, and scripts under `artifact/scripts/`.
Start with:

```bash
./artifact/scripts/00_quick_core_check.sh
./artifact/scripts/01_reproduce_swe_cached_no_api.sh
./artifact/scripts/03_verify_all69_claims.sh
```

## Scope

MASGuard automatically performs evidence extraction, evidence graph
construction, failure diagnosis, recovery-action selection, recovery prompt
generation, and validation logging. It does not replace the MAS locator,
patcher, or verifier. Its role is to make the next MAS attempt evidence-guided
instead of a blind retry.
