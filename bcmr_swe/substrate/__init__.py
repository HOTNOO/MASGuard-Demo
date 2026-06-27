"""Substrate helpers for BCMR-SWE."""

from bcmr_swe.substrate.harness_bridge import DockerHarnessBridge, DockerHarnessConfig
from bcmr_swe.substrate.harness_runtime import HarnessRuntimeConfig, OfficialHarnessRuntime
from bcmr_swe.substrate.manifests import load_manifest_catalog, run_oracle_preflight, select_manifest_entry
from bcmr_swe.substrate.stage_bundle import LocatorStage, PatcherStage, StageBundle, VerifierStage
from bcmr_swe.substrate.swe_mas_legacy import build_swe_mas_stage_bundle
from bcmr_swe.substrate.swe_env import SWEEnvironment

__all__ = [
    "DockerHarnessBridge",
    "DockerHarnessConfig",
    "HarnessRuntimeConfig",
    "LocatorStage",
    "OfficialHarnessRuntime",
    "PatcherStage",
    "SWEEnvironment",
    "StageBundle",
    "VerifierStage",
    "build_swe_mas_stage_bundle",
    "load_manifest_catalog",
    "run_oracle_preflight",
    "select_manifest_entry",
]
