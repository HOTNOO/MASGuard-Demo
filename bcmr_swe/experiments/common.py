"""Shared helpers for BCMR experiment entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
import uuid

from bcmr_swe.agent import BCMRCoordinator, BCMRCoordinatorConfig
from bcmr_swe.models import (
    AnthropicCompatibleChatConfig,
    AnthropicCompatibleChatModel,
    GeminiChatConfig,
    GeminiChatModel,
    GraphConditionedRecoveryValueModel,
    OpenAICompatibleChatConfig,
    OpenAICompatibleChatModel,
    XGBoostRecoveryRanker,
)
from bcmr_swe.recovery import RecoverySelector
from bcmr_swe.substrate import (
    HarnessRuntimeConfig,
    OfficialHarnessRuntime,
    StageBundle,
    build_swe_mas_stage_bundle,
)
from swe_mas.utils.env_executor import LocalExecutor


@dataclass
class QueryBudgetModel:
    """Wrap a chat model with stage-specific default query kwargs."""

    base_model: Any
    default_kwargs: dict[str, Any]

    def query(self, messages: list[dict[str, str]], **kwargs):
        merged = dict(self.default_kwargs)
        merged.update(kwargs)
        return self.base_model.query(messages, **merged)

    def get_template_vars(self) -> dict[str, Any]:
        return self.base_model.get_template_vars()

    def get_usage_snapshot(self) -> dict[str, float]:
        getter = getattr(self.base_model, "get_usage_snapshot", None)
        if callable(getter):
            return dict(getter())
        return {
            "n_calls": float(getattr(self.base_model, "n_calls", 0.0) or 0.0),
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
        }


def build_chat_model(api_path: str | Path, model_name: str | None = None, request_timeout: int = 60, max_retries: int = 1):
    lines = [line.strip() for line in Path(api_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    looks_anthropic_compat = any(
        line.lower().startswith("provider=anthropic")
        or "/anthropic" in line.lower()
        or line.lower().startswith("base_url=") and "anthropic" in line.lower()
        for line in lines
    )
    if looks_anthropic_compat:
        config = AnthropicCompatibleChatConfig.from_api_md(api_path)
        if model_name:
            config.model = model_name
        config.request_timeout = request_timeout
        config.max_retries = max_retries
        return AnthropicCompatibleChatModel(config)
    looks_openai_compat = any(
        line.startswith("sk-") or "chat/completions" in line or line.startswith("/v1/") or line.startswith("endpoint=")
        for line in lines
    )
    if looks_openai_compat:
        config = OpenAICompatibleChatConfig.from_api_md(api_path)
        if model_name:
            config.model = model_name
        config.request_timeout = request_timeout
        config.max_retries = max_retries
        return OpenAICompatibleChatModel(config)

    config = GeminiChatConfig.from_api_md(api_path)
    if model_name:
        config.model = model_name
    config.request_timeout = request_timeout
    config.max_retries = max_retries
    return GeminiChatModel(config)


def disable_streaming_if_supported(model: Any) -> Any:
    """Force non-streaming queries for benchmark stability when supported."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "prefer_stream"):
        setattr(config, "prefer_stream", False)
    return model


def build_gemini_model(api_path: str | Path, model_name: str | None = None, request_timeout: int = 60, max_retries: int = 1):
    return build_chat_model(api_path, model_name=model_name, request_timeout=request_timeout, max_retries=max_retries)


def resolve_model_names(
    api_path: str | Path,
    *,
    model_name: str | None = None,
    strong_model_name: str | None = None,
) -> tuple[str, str]:
    """Resolve runnable model names from api.md with safe fallbacks."""
    base_model = build_chat_model(api_path, model_name=model_name, request_timeout=10, max_retries=0)
    resolved_model = str(getattr(getattr(base_model, "config", None), "model", "") or model_name or "").strip()
    if not resolved_model:
        raise ValueError("Failed to resolve a usable base model from api.md.")
    if strong_model_name:
        strong_model = build_chat_model(api_path, model_name=strong_model_name, request_timeout=10, max_retries=0)
        resolved_strong = str(getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name).strip()
    else:
        resolved_strong = resolved_model
    return resolved_model, resolved_strong


def run_model_preflight(
    api_path: str | Path,
    *,
    model_name: str,
    label: str,
    request_timeout: int = 20,
) -> dict[str, Any]:
    """Fail fast if the configured model cannot answer a trivial probe."""
    model = build_chat_model(
        api_path,
        model_name=model_name,
        request_timeout=request_timeout,
        max_retries=2,
    )
    result = model.query(
        [
            {"role": "system", "content": "You are a health-check assistant. Reply with OK only."},
            {"role": "user", "content": "Probe model availability."},
        ],
        max_tokens=8,
        request_timeout=request_timeout,
        temperature=0.0,
    )
    content = str(result.get("content", "")).strip()
    if not content:
        raise RuntimeError(f"{label} preflight returned empty content for model={model_name}")
    lowered = content.lower()
    if lowered.startswith("thinking about your request"):
        raise RuntimeError(
            f"{label} preflight returned placeholder content for model={model_name}: {content!r}"
        )
    return {"label": label, "model": model_name, "content": content[:80]}


def build_coordinator(
    *,
    workspace: str,
    model,
    strong_model=None,
    strong_stages: tuple[str, ...] = ("planner", "implementer"),
    executor: Any | None = None,
    output_root: str | Path | None = None,
    selector_policy: str = "heuristic",
    selector_model_path: str | Path | None = None,
    recovery_mode: str = "v3_program",
    synthesizer_model=None,
    capture_full_counterfactual_group: bool = False,
    capture_counterfactual_outcomes: bool = True,
    execute_live_after_full_capture: bool = True,
    continue_after_full_capture: bool = True,
    max_captured_failed_state_groups: int | None = 1,
    counterfactual_followup_recovery_calls: int = 0,
    locator_max_iterations: int = 8,
    planner_max_iterations: int = 4,
    patcher_max_iterations: int = 8,
    verifier_max_iterations: int = 6,
    stage_backend: str = "swe_mas_legacy",
    stage_bundle: StageBundle | None = None,
) -> BCMRCoordinator:
    executor = executor or LocalExecutor(cwd=workspace)
    strong_stage_set = {stage.strip().lower() for stage in strong_stages if stage and stage.strip()}

    def stage_base(name: str):
        if strong_model is not None and name.lower() in strong_stage_set:
            return strong_model
        return model

    stage_token_budgets = {
        "locator": {"max_tokens": 128},
        "planner": {"max_tokens": 512},
        "implementer": {"max_tokens": 1024},
        "verifier": {"max_tokens": 192},
    }
    locator_model = QueryBudgetModel(stage_base("locator"), stage_token_budgets["locator"])
    planner_model = QueryBudgetModel(stage_base("planner"), stage_token_budgets["planner"])
    implementer_model = QueryBudgetModel(stage_base("implementer"), stage_token_budgets["implementer"])
    verifier_model = QueryBudgetModel(stage_base("verifier"), stage_token_budgets["verifier"])
    strong_stage_models = {}
    if strong_model is not None:
        strong_stage_models = {
            stage: QueryBudgetModel(strong_model, defaults)
            for stage, defaults in stage_token_budgets.items()
        }
    if stage_bundle is None:
        if stage_backend != "swe_mas_legacy":
            raise ValueError(f"Unsupported stage backend: {stage_backend}")
        stage_bundle = build_swe_mas_stage_bundle(
            model=model,
            executor=executor,
            planner_model=planner_model,
            implementer_model=implementer_model,
            verifier_model=verifier_model,
            locator_model=locator_model,
            locator_max_iterations=locator_max_iterations,
            planner_max_iterations=planner_max_iterations,
            patcher_max_iterations=patcher_max_iterations,
            verifier_max_iterations=verifier_max_iterations,
        )
    selector = build_selector(policy=selector_policy, model_path=selector_model_path)

    synthesizer = None
    if recovery_mode == "v3_program":
        from bcmr_swe.recovery.program_synthesizer import ProgramSynthesizer
        syn_model = synthesizer_model or strong_model or model
        synthesizer = ProgramSynthesizer(syn_model)

    config = BCMRCoordinatorConfig(
        output_root=Path(output_root) if output_root else BCMRCoordinatorConfig.output_root,
        capture_full_counterfactual_group=capture_full_counterfactual_group,
        capture_counterfactual_outcomes=capture_counterfactual_outcomes,
        execute_live_after_full_capture=execute_live_after_full_capture,
        continue_after_full_capture=continue_after_full_capture,
        max_captured_failed_state_groups=max_captured_failed_state_groups,
        counterfactual_followup_recovery_calls=counterfactual_followup_recovery_calls,
    )
    runtime = BCMRCoordinator(
        locator=stage_bundle.locator,
        patcher=stage_bundle.patcher,
        verifier=stage_bundle.verifier,
        selector=selector,
        synthesizer=synthesizer,
        config=config,
    )
    runtime._bcmr_strong_stage_models = strong_stage_models
    runtime._bcmr_strong_stage_set = set(strong_stage_set)
    return runtime


def materialize_workspace(
    source_snapshot: str | Path,
    workspace_root: str | Path,
    instance_id: str,
    *,
    strategy: str = "copy",
) -> Path:
    source_snapshot = Path(source_snapshot).resolve()
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    unique_suffix = f"{int(time.time() * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    workspace = workspace_root / f"{instance_id}_{unique_suffix}"
    normalized = strategy.lower().strip()
    if normalized in {"", "copy"}:
        shutil.copytree(source_snapshot, workspace)
        return workspace
    if normalized == "git_clone":
        completed = subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(source_snapshot), str(workspace)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout or f"failed to clone snapshot workspace from {source_snapshot}")
        _restore_snapshot_submodule_worktrees(source_snapshot, workspace)
        return workspace
    raise ValueError(f"Unsupported workspace materialization strategy: {strategy}")


def _restore_snapshot_submodule_worktrees(source_snapshot: Path, workspace: Path) -> None:
    """Preserve locally preseeded submodule trees after a snapshot git clone.

    Some legacy SWE-bench repositories bootstrap build helpers from git
    submodules during editable install. Our source snapshots may already contain
    those helper trees for offline harness execution, but ``git clone`` only
    materializes the gitlink and not the local submodule working tree.
    """
    gitmodules = source_snapshot / ".gitmodules"
    try:
        text = gitmodules.read_text(encoding="utf-8")
    except OSError:
        return
    for match in re.finditer(r"(?m)^\s*path\s*=\s*(.+?)\s*$", text):
        relative = match.group(1).strip()
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            continue
        source_path = source_snapshot / relative
        target_path = workspace / relative
        if not source_path.is_dir():
            continue
        if target_path.exists() and any(target_path.iterdir()):
            continue
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)


def resolve_runtime(runtime: str, manifest: dict[str, Any] | None = None) -> str:
    normalized = runtime.lower().strip()
    if normalized in {"", "local"}:
        return "local"
    if normalized == "auto":
        return "harness" if manifest and manifest.get("dataset_name") else "local"
    if normalized == "harness":
        return "harness"
    raise ValueError(f"Unsupported runtime: {runtime}")


def _candidate_hf_dataset_cache_roots(manifest: dict[str, Any] | None = None) -> list[Path]:
    roots: list[Path] = []
    manifest_root = str(dict(manifest or {}).get("hf_cache_root", "") or "").strip()
    if manifest_root:
        roots.append(Path(manifest_root))
    for env_key in ("BCMR_HF_CACHE_ROOT", "HF_DATASETS_CACHE"):
        env_value = str(os.environ.get(env_key, "") or "").strip()
        if env_value:
            roots.append(Path(env_value))
    hf_home = str(os.environ.get("HF_HOME", "") or "").strip()
    if hf_home:
        roots.append(Path(hf_home) / "datasets")
    roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(root.expanduser())
    return out


def _maybe_enable_hf_offline_for_cached_dataset(dataset_name: str, manifest: dict[str, Any] | None = None) -> None:
    normalized = dataset_name.strip().lower().replace("/", "___")
    matched_root: Path | None = None
    for cache_root in _candidate_hf_dataset_cache_roots(manifest):
        if not cache_root.exists():
            continue
        matches = [path for path in cache_root.glob(f"{normalized}*") if path.is_dir()]
        if matches:
            matched_root = cache_root
            break
    if matched_root is None:
        return
    os.environ["HF_DATASETS_CACHE"] = str(matched_root)
    os.environ["HF_HOME"] = str(matched_root.parent)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        import datasets  # type: ignore
    except ModuleNotFoundError:
        return
    config = getattr(datasets, "config", None)
    if config is None:
        return
    if hasattr(config, "HF_DATASETS_CACHE"):
        setattr(config, "HF_DATASETS_CACHE", str(matched_root))
    if hasattr(config, "HF_CACHE_HOME"):
        setattr(config, "HF_CACHE_HOME", str(matched_root.parent))
    if hasattr(config, "HF_DATASETS_OFFLINE"):
        setattr(config, "HF_DATASETS_OFFLINE", True)


def workspace_strategy_for_runtime(runtime: str, manifest: dict[str, Any] | None = None) -> str:
    return "git_clone" if resolve_runtime(runtime, manifest) == "harness" else "copy"


def swebench_instance_from_manifest(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build the official SWE-bench row needed for TestSpec from a local manifest.

    The harness only needs the canonical SWE-bench fields to select the env
    image and repo setup script.  Providing them avoids a startup-time
    HuggingFace lookup when the local manifest already pins the instance.
    """
    if not manifest:
        return None
    required = ("instance_id", "repo", "version", "base_commit")
    if any(not str(manifest.get(key, "") or "").strip() for key in required):
        return None
    fail_to_pass = manifest.get("FAIL_TO_PASS", manifest.get("dataset_fail_to_pass", []))
    pass_to_pass = manifest.get("PASS_TO_PASS", manifest.get("dataset_pass_to_pass", []))
    row = {
        "instance_id": str(manifest["instance_id"]),
        "repo": str(manifest["repo"]),
        "version": str(manifest["version"]),
        "base_commit": str(manifest["base_commit"]),
        "problem_statement": str(manifest.get("problem_statement", "") or ""),
        "hints_text": str(manifest.get("hints_text", "") or ""),
        "test_patch": str(manifest.get("test_patch", "") or ""),
        "FAIL_TO_PASS": fail_to_pass if isinstance(fail_to_pass, list) else [],
        "PASS_TO_PASS": pass_to_pass if isinstance(pass_to_pass, list) else [],
    }
    source_snapshot = str(manifest.get("source_snapshot", "") or "").strip()
    if source_snapshot:
        row["source_snapshot"] = source_snapshot
    environment_setup_commit = str(
        manifest.get("environment_setup_commit")
        or manifest.get("dataset_environment_setup_commit")
        or ""
    ).strip()
    if environment_setup_commit:
        row["environment_setup_commit"] = environment_setup_commit
        row["dataset_environment_setup_commit"] = environment_setup_commit
    return row


def build_selector(*, policy: str = "heuristic", model_path: str | Path | None = None) -> RecoverySelector:
    normalized = policy.lower().strip()
    if normalized in {"", "heuristic"}:
        return RecoverySelector()
    if model_path is None:
        raise ValueError(f"selector policy '{policy}' requires --selector-model-path")
    model_path = Path(model_path)
    if normalized == "gcrv":
        return RecoverySelector(model=GraphConditionedRecoveryValueModel.load(model_path))
    if normalized == "xgb":
        return RecoverySelector(model=XGBoostRecoveryRanker.load(model_path))
    raise ValueError(f"Unsupported selector policy: {policy}")


def build_executor(
    *,
    workspace: str,
    runtime: str = "local",
    manifest: dict[str, Any] | None = None,
    force_rebuild_harness: bool = False,
    harness_setup_timeout: int | None = None,
    harness_container_start_timeout: int | None = None,
    harness_container_cleanup_timeout: int | None = None,
    harness_env_image_key: str = "",
):
    normalized = resolve_runtime(runtime, manifest)
    if normalized == "local":
        return LocalExecutor(cwd=workspace), None
    if manifest is None:
        raise ValueError("Harness runtime requires a manifest payload.")
    dataset_name = str(manifest.get("dataset_name") or "princeton-nlp/SWE-bench_Verified")
    dataset_instance = swebench_instance_from_manifest(manifest)
    if dataset_instance is None:
        _maybe_enable_hf_offline_for_cached_dataset(dataset_name, manifest)
    session = OfficialHarnessRuntime(
        instance_id=str(manifest["instance_id"]),
        workspace=workspace,
        config=HarnessRuntimeConfig(
            dataset_name=dataset_name,
            split=str(manifest.get("dataset_split") or "test"),
            dataset_instance=dataset_instance,
            force_rebuild=force_rebuild_harness,
            run_id=f"bcmr-{manifest['instance_id']}",
            setup_timeout=int(harness_setup_timeout) if harness_setup_timeout is not None else HarnessRuntimeConfig.setup_timeout,
            container_start_timeout=(
                int(harness_container_start_timeout)
                if harness_container_start_timeout is not None
                else HarnessRuntimeConfig.container_start_timeout
            ),
            container_cleanup_timeout=(
                int(harness_container_cleanup_timeout)
                if harness_container_cleanup_timeout is not None
                else HarnessRuntimeConfig.container_cleanup_timeout
            ),
            env_image_key_override=str(harness_env_image_key or ""),
            skip_recursive_chmod=bool(str(harness_env_image_key or "").strip()),
        ),
    )
    session.start()
    return session, session
