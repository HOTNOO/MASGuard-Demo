# MASGuard Architecture

MASGuard is organized as a sidecar plugin plus a SWE/MAS backend. The sidecar
is the stable interface that an external repair agent uses. The backend is the
larger implementation used to run the real SWE case and build the experiment
records included in the reproducibility package.

## Layer A: CLI Sidecar

Location:

```text
src/recoveragent/
```

Installed command:

```text
masguard
```

Responsibilities:

- create and validate a standard failed-run directory;
- parse agent trajectory JSON, failed patches, repository state, and logs;
- build evidence signals and an evidence graph;
- classify the failure mode;
- select a recovery action;
- write JSON, HTML, and Markdown recovery artifacts;
- apply a MAS-produced recovery patch for accounting;
- run validation commands and record stdout/stderr, return code, and status.

The sidecar is intentionally generic. It does not assume a specific MAS
framework, planner, model, or event-stream format beyond the failed-run
contract.

## Layer B: SWE/MAS Backend

Location:

```text
bcmr_swe/
```

Used by:

```text
examples/user_mas/swe_live_mas.py
artifact/scripts/01_reproduce_swe_cached_no_api.sh
artifact/scripts/02_reproduce_swe_live_online.sh
artifact/scripts/03_verify_all69_claims.sh
```

Responsibilities:

- run the online SWE baseline MAS on real repository snapshots;
- generate failed repair trajectories and source patches;
- drive the recovery patcher with MASGuard's prompt and bounded source
  evidence;
- execute local validation commands against real project tests;
- generate and verify the 69-instance result records.

For the main SWE demonstration, `examples/user_mas/swe_live_mas.py
run-baseline` calls:

```text
python -m bcmr_swe.experiments.mas_recovery_run_clean_start_baseline
```

That backend performs live model calls, runs repository commands, writes a
baseline JSON record, and provides the failed run that MASGuard analyzes.

## Layer C: MAS Substrate

Location:

```text
swe_mas/
```

Responsibilities:

- reusable multi-agent repair components;
- historical MAS agent/control abstractions;
- support code used by the broader implementation line.

This layer is shipped so the implementation lineage is visible and inspectable.
The public CLI does not require external users to import this layer for the
basic failed-run contract.

## Data Flow

```text
existing MAS or bcmr_swe MAS
  -> failed validation
  -> export failed_run/
       repo/
       trajectory.json
       logs/failing.log
       patches/failed.patch
  -> masguard analyze-run --run-dir failed_run
  -> recoveragent/report.json
  -> recoveragent/recovery_prompt.md
  -> same MAS patcher resumes with the prompt
  -> patches/recovered.patch
  -> masguard apply-patch
  -> masguard validate
  -> validation.json and validation.log
```

## What Each Layer Does Not Do

`src/recoveragent` does not perform repository-level patch synthesis by itself.
It diagnoses failure evidence and writes recovery instructions.

`bcmr_swe` is not a hidden demo stub. It is the SWE/MAS execution backend used
for the real Django online run and the 69-instance records.

`swe_mas` is not the primary public API. It is included to preserve the MAS
substrate needed by the implementation history.

## Verification Entry Points

Generic CLI and sidecar tests:

```bash
./artifact/scripts/00_quick_core_check.sh
```

Cached real SWE case without API:

```bash
./artifact/scripts/01_reproduce_swe_cached_no_api.sh
```

Live online SWE case:

```bash
API_PATH=/path/to/api-config.md ./artifact/scripts/02_reproduce_swe_live_online.sh
```

69-instance result records:

```bash
./artifact/scripts/03_verify_all69_claims.sh
```
