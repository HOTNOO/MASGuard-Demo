"""恢复管理：接收故障信号，标记DAG，生成摘要并返回分支锚点"""

from __future__ import annotations

from typing import Any
import os
from dataclasses import asdict
import json

from swe_mas.memory import MemoryRecorder
from swe_mas.perception.models import FaultSignal
from swe_mas.recovery.summary_engine import SummaryEngine
from swe_mas.recovery.models import RecoveryRequest, ResourceCondition, ResourceLevel
from swe_mas.recovery.interfaces import LLMClientProtocol


class RecoveryManager:
    def __init__(
        self,
        recorder: MemoryRecorder,
        llm: LLMClientProtocol | None = None,
        role_focus: dict[str, list[str]] | None = None,
        summary_policy=None,
    ):
        self.recorder = recorder
        self.summary_engine = SummaryEngine(recorder=recorder, llm=llm, role_focus=role_focus, summary_policy=summary_policy)

    def _resolve_resource(self, resource: ResourceCondition, session_id: str) -> ResourceCondition:
        resolved = ResourceCondition(bandwidth=resource.bandwidth, token_left=resource.token_left)
        env_bw = os.getenv("SWE_MAS_BANDWIDTH", "").strip().lower()
        if env_bw in {"low", "medium", "high"}:
            resolved.bandwidth = ResourceLevel(env_bw)

        env_token_left = os.getenv("SWE_MAS_TOKEN_LEFT")
        if resolved.token_left is None and env_token_left:
            try:
                resolved.token_left = max(0, int(env_token_left))
            except Exception:
                pass

        if resolved.token_left is None:
            env_budget = os.getenv("SWE_MAS_TOKEN_BUDGET")
            if env_budget:
                try:
                    budget = int(env_budget)
                    used = 0
                    try:
                        used = int(self.recorder.get_token_usage_sum(session_id))
                    except Exception:
                        used = 0
                    resolved.token_left = max(0, budget - used)
                except Exception:
                    pass
        return resolved

    def handle_fault(self, *, fault: FaultSignal, session_id: str, target_role: str, resource: ResourceCondition, fault_index: int | None = None) -> dict[str, Any]:
        # 标记故障节点和故障Phase
        if fault.node_ids:
            self.recorder.update_validity(fault.node_ids, validity="INVALID", status="failed")
        
        # 标记当前Phase为失败（用于后续快速查询健康Phase）
        self.recorder.mark_phase_invalid(session_id, target_role)

        resolved_resource = self._resolve_resource(resource, session_id)
        request = RecoveryRequest(
            session_id=session_id,
            target_role=target_role,
            fault_type=fault.fault_type,
            resource=resolved_resource,
            details={"message": fault.message, "extra": fault.extra, "fault_node_ids": fault.node_ids, "fault_index": fault_index},
        )
        summary = self.summary_engine.summarize(request)
        # 记录一次策略反馈（reward 后续可回填）
        try:
            strategy = summary.get("strategy")
            bucket_id = request.details.get("summary_bucket_id") if request.details else None
            strategy_id = request.details.get("summary_strategy_id") if request.details else None
            fault_index = request.details.get("fault_index") if request.details else None
            fallback_node_id = ""
            try:
                last = self.recorder.get_nodes(session_id, limit=1)
                fallback_node_id = last[-1]["node_id"] if last else ""
            except Exception:
                pass
            action_dict: dict[str, Any] = {}
            if strategy:
                action_dict = {
                    "granularity": getattr(strategy.granularity, "value", strategy.granularity),
                    "focus_areas": getattr(strategy, "focus_areas", []),
                    "discard_recent_branch": getattr(strategy, "discard_recent_branch", False),
                    "reuse_healthy_intent": getattr(strategy, "reuse_healthy_intent", True),
                    "max_history": getattr(strategy, "max_history", None),
                }
            payload = {
                "kind": "summary",
                "bucket_id": bucket_id,
                "strategy_id": strategy_id,
                "fault_index": fault_index,
                "context": {
                    "fault_type": str(fault.fault_type),
                    "target_role": target_role,
                    "resource": {
                        "bandwidth": resource.bandwidth.value,
                        "token_left": resource.token_left,
                    },
                },
                "action": action_dict,
            }
            self.recorder.log_feedback(
                node_id=fault.node_ids[0] if fault.node_ids else fallback_node_id,
                reward=0.0,
                strategy=json.dumps(payload, ensure_ascii=False),
                policy_name="summary_policy_v1",
                strategy_json=payload,
            )
        except Exception:
            pass
        try:
            strategy = summary.get("strategy")
            if strategy and getattr(strategy, "discard_recent_branch", False) and fault.node_ids:
                try:
                    all_nodes = self.recorder.get_nodes(request.session_id)
                    fault_set = set(fault.node_ids)
                    first_idx = None
                    for i, n in enumerate(all_nodes):
                        if n.get("node_id") in fault_set:
                            first_idx = i
                            break
                    if first_idx is not None:
                        tail_ids = [n["node_id"] for n in all_nodes[first_idx:]]
                        self.recorder.update_validity(tail_ids, validity="INVALID", status="failed")
                except Exception:
                    self.recorder.update_validity(fault.node_ids, validity="INVALID", status="failed")
        except Exception:
            pass
        return summary
