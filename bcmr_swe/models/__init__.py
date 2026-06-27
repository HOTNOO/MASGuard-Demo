"""Model interfaces for BCMR-SWE selectors."""

try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional runtime dependency
    load_dotenv = None
else:
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path or None, override=False)

from bcmr_swe.models.gemini_chat import GeminiChatConfig, GeminiChatModel
from bcmr_swe.models.anthropic_compatible_chat import AnthropicCompatibleChatConfig, AnthropicCompatibleChatModel
from bcmr_swe.models.openai_compatible_chat import OpenAICompatibleChatConfig, OpenAICompatibleChatModel
from bcmr_swe.models.gcrv import GraphConditionedRecoveryValueModel
from bcmr_swe.models.student_reranker import StudentReranker
from bcmr_swe.models.xgb_baseline import XGBoostRecoveryRanker

__all__ = [
    "GeminiChatConfig",
    "GeminiChatModel",
    "GraphConditionedRecoveryValueModel",
    "AnthropicCompatibleChatConfig",
    "AnthropicCompatibleChatModel",
    "OpenAICompatibleChatConfig",
    "OpenAICompatibleChatModel",
    "StudentReranker",
    "XGBoostRecoveryRanker",
]
