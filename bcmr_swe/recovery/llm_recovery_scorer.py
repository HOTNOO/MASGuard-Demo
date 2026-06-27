"""LLM-based recovery value scorer using structured provenance context.

This module is the core of the BCMR-Hybrid Phase 1 (BCMR-LLM).  It sends
a structured provenance summary to an LLM and asks it to score every
candidate recovery action in a single call.  The key innovation is that the
LLM receives *structured provenance context* (causal dependency chain,
conflicting facts, suspect region scope) rather than raw execution logs,
which dramatically improves recovery reasoning quality.

The scorer output includes per-action estimates and a brief diagnosis that
later serves as a *rationale feature* for the Phase 2 student reranker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bcmr_swe.provenance.graph_store import ExecutionProvenanceGraph
from bcmr_swe.recovery.provenance_summarizer import ProvenanceSummarizer
from bcmr_swe.types import (
    ActionScore,
    CandidateAction,
    FailedState,
    RecoveryBudget,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a recovery decision engine for a multi-stage SWE agent system.

The agent follows a pipeline: locator → patcher → verifier.  It has entered
a *failed state* where the current context, shared facts, or workspace may
be polluted.  Your task is to evaluate each candidate recovery action and
estimate its value under budget constraints.

Guidelines:
- Prefer actions that address the root cause shown in the dependency chain.
- Prefer minimal-cost actions when recovery probabilities are similar.
- A QUARANTINE_FACT is cheap but only effective when a specific fact is wrong.
- A ROLLBACK_TO_CHECKPOINT is safe but discards progress.
- A REPLAY_SUBGRAPH reruns only the suspect chain – good when the error is
  localised but the rest of the pipeline is healthy.
- INSERT_VERIFIER adds a deeper check – useful when the verdict was wrong
  but the patch might actually be correct.
- ESCALATE_NODE upgrades the strategy – useful when the stage simply lacks
  capability.

Respond with valid JSON only (no markdown fences).  Use this schema:
{
  "diagnosis": "<one-paragraph root-cause diagnosis>",
  "action_scores": [
    {
      "action_index": 1,
      "estimated_recover_prob": <float 0-1>,
      "estimated_token_cost": <int>,
      "estimated_latency_sec": <float>,
      "estimated_risk": <float 0-1>,
      "rationale": "<one sentence>"
    }
  ],
  "recommended_index": <int, 1-based>
}
"""

USER_TEMPLATE = """\
Evaluate the following failed state and its candidate recovery actions.

{provenance_context}

For each action [A1] through [A{n_actions}], provide your estimates.
Remember: focus on budget efficiency – pick the cheapest action that has
a reasonable chance of recovering the task.
"""


class LLMBackendProtocol(Protocol):
    """Any object with a ``query(messages, **kwargs) -> dict`` method."""

    def query(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]: ...


@dataclass
class LLMRecoveryScorerConfig:
    temperature: float = 0.0
    max_output_tokens: int = 1024
    request_timeout: int = 90
    cache_dir: Path | None = None
    enable_cache: bool = True


class LLMRecoveryScorer:
    """Score candidate recovery actions via an LLM using provenance context.

    The scorer converts the execution provenance graph into a structured
    natural-language summary and asks the LLM to evaluate all actions at once.
    Results include estimated recovery probability, cost, risk, and a brief
    rationale for each action.

    Attributes
    ----------
    model : LLMBackendProtocol
        An LLM chat adapter (OpenAI-compatible or Gemini).
    config : LLMRecoveryScorerConfig
        Temperature, timeout, and caching settings.
    summarizer : ProvenanceSummarizer
        Converts the provenance graph into structured text.
    """

    def __init__(
        self,
        model: LLMBackendProtocol,
        *,
        config: LLMRecoveryScorerConfig | None = None,
    ):
        self.model = model
        self.config = config or LLMRecoveryScorerConfig()
        self.summarizer = ProvenanceSummarizer()
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self.last_raw_response: str = ""
        self.last_diagnosis: str = ""
        self.total_scorer_calls: int = 0
        self.total_scorer_tokens: float = 0.0

    def score(
        self,
        graph: ExecutionProvenanceGraph,
        failed_state: FailedState,
        actions: list[CandidateAction],
        budget: RecoveryBudget,
        *,
        used_recovery_calls: int = 0,
        used_tokens: float = 0.0,
    ) -> list[ActionScore]:
        """Return scored actions sorted by budget-aware utility (descending)."""
        if not actions:
            return []

        provenance_text = self.summarizer.summarize(
            graph,
            failed_state,
            actions,
            budget,
            used_recovery_calls=used_recovery_calls,
            used_tokens=used_tokens,
        )

        cache_key = self._cache_key(provenance_text)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return self._to_action_scores(cached, actions, budget)

        raw_scores = self._call_llm(provenance_text, len(actions))
        if raw_scores is None:
            logger.warning("LLM scorer failed; falling back to prior estimates.")
            return self._fallback_scores(actions, budget)

        self._save_cache(cache_key, raw_scores)
        return self._to_action_scores(raw_scores, actions, budget)

    def score_to_dict(
        self,
        graph: ExecutionProvenanceGraph,
        failed_state: FailedState,
        actions: list[CandidateAction],
        budget: RecoveryBudget,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Score and return full metadata (for dataset persistence)."""
        scores = self.score(graph, failed_state, actions, budget, **kwargs)
        return {
            "diagnosis": self.last_diagnosis,
            "action_scores": [s.to_dict() for s in scores],
            "scorer_model": getattr(
                getattr(self.model, "config", None), "model", "unknown"
            ),
            "raw_response_excerpt": self.last_raw_response[:2000],
        }

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(
        self, provenance_text: str, n_actions: int
    ) -> list[dict[str, Any]] | None:
        user_msg = USER_TEMPLATE.format(
            provenance_context=provenance_text,
            n_actions=n_actions,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            result = self.model.query(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                request_timeout=self.config.request_timeout,
            )
            raw_text = result.get("content", "")
            self.last_raw_response = raw_text
            self.total_scorer_calls += 1

            usage = result.get("extra", {}).get("usage", {})
            self.total_scorer_tokens += float(
                usage.get("total_tokens", 0)
                or usage.get("totalTokenCount", 0)
                or 0
            )

            return self._parse_response(raw_text, n_actions)
        except Exception:
            logger.exception("LLM recovery scorer call failed")
            return None

    def _parse_response(
        self, raw: str, n_actions: int
    ) -> list[dict[str, Any]] | None:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse LLM JSON response")
                    return None
            else:
                logger.warning("No JSON found in LLM response")
                return None

        self.last_diagnosis = str(data.get("diagnosis", ""))
        raw_scores = data.get("action_scores", [])
        if not isinstance(raw_scores, list) or len(raw_scores) == 0:
            logger.warning("LLM returned empty action_scores")
            return None

        parsed: list[dict[str, Any]] = []
        for item in raw_scores[:n_actions]:
            if not isinstance(item, dict):
                continue
            parsed.append({
                "action_index": int(item.get("action_index", len(parsed) + 1)),
                "estimated_recover_prob": _clamp(
                    float(item.get("estimated_recover_prob", 0.5)), 0.0, 1.0
                ),
                "estimated_token_cost": max(
                    0.0, float(item.get("estimated_token_cost", 1000))
                ),
                "estimated_latency_sec": max(
                    0.0, float(item.get("estimated_latency_sec", 30))
                ),
                "estimated_risk": _clamp(
                    float(item.get("estimated_risk", 0.1)), 0.0, 1.0
                ),
                "rationale": str(item.get("rationale", "")),
            })

        if not parsed:
            return None
        return parsed

    # ------------------------------------------------------------------
    # Score conversion
    # ------------------------------------------------------------------

    def _to_action_scores(
        self,
        raw_scores: list[dict[str, Any]],
        actions: list[CandidateAction],
        budget: RecoveryBudget,
    ) -> list[ActionScore]:
        score_by_index = {
            int(s.get("action_index", i + 1)): s
            for i, s in enumerate(raw_scores)
        }
        results: list[ActionScore] = []
        for idx, action in enumerate(actions):
            llm_s = score_by_index.get(idx + 1, {})
            p_recover = float(llm_s.get("estimated_recover_prob", action.estimated_recover_prob))
            c_token = float(llm_s.get("estimated_token_cost", action.estimated_token_cost))
            c_latency = float(llm_s.get("estimated_latency_sec", action.estimated_latency_sec))
            risk = float(llm_s.get("estimated_risk", action.estimated_risk))
            rationale = str(llm_s.get("rationale", ""))

            utility = (
                p_recover
                - budget.lambda_token * c_token
                - budget.lambda_latency * c_latency
                - budget.lambda_risk * risk
            )
            results.append(
                ActionScore(
                    action_id=action.action_id,
                    utility=utility,
                    estimated_recover_prob=p_recover,
                    estimated_token_cost=c_token,
                    estimated_latency_sec=c_latency,
                    estimated_risk=risk,
                    explanation=f"llm_teacher | {rationale}",
                )
            )
        results.sort(key=lambda s: s.utility, reverse=True)
        return results

    def _fallback_scores(
        self,
        actions: list[CandidateAction],
        budget: RecoveryBudget,
    ) -> list[ActionScore]:
        results: list[ActionScore] = []
        for action in actions:
            utility = (
                action.estimated_recover_prob
                - budget.lambda_token * action.estimated_token_cost
                - budget.lambda_latency * action.estimated_latency_sec
                - budget.lambda_risk * action.estimated_risk
            )
            results.append(
                ActionScore(
                    action_id=action.action_id,
                    utility=utility,
                    estimated_recover_prob=action.estimated_recover_prob,
                    estimated_token_cost=action.estimated_token_cost,
                    estimated_latency_sec=action.estimated_latency_sec,
                    estimated_risk=action.estimated_risk,
                    explanation="fallback_prior_estimates",
                )
            )
        results.sort(key=lambda s: s.utility, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]

    def _load_cache(self, key: str) -> list[dict[str, Any]] | None:
        if not self.config.enable_cache:
            return None
        if key in self._cache:
            return self._cache[key]
        if self.config.cache_dir:
            path = self.config.cache_dir / f"{key}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self.last_diagnosis = str(data.get("diagnosis", ""))
                    scores = data.get("action_scores", [])
                    self._cache[key] = scores
                    return scores
                except Exception:
                    pass
        return None

    def _save_cache(self, key: str, scores: list[dict[str, Any]]) -> None:
        self._cache[key] = scores
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.config.cache_dir / f"{key}.json"
            payload = {
                "diagnosis": self.last_diagnosis,
                "action_scores": scores,
            }
            try:
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                logger.warning("Failed to write scorer cache to %s", path)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
