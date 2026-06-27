"""Multi-Agent System for SWE-bench.

基于 mini-swe-agent 的多智能体协作系统。
"""

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

legacy_config_dir = PROJECT_ROOT / "src" / "swe_mas" / "config"
local_config_dir = PACKAGE_ROOT / "config"
CONFIG_DIR = local_config_dir if local_config_dir.exists() else legacy_config_dir
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["__version__", "PACKAGE_ROOT", "PROJECT_ROOT", "CONFIG_DIR", "OUTPUT_DIR"]
