"""感知层管理器：汇集心跳、图异常、故障注入"""

from __future__ import annotations

import time
from typing import Any

from swe_mas.perception.heartbeat import HeartbeatMonitor
from swe_mas.perception.graph_analyzer import GraphAnomalyDetector
from swe_mas.perception.fault_injector import FaultInjector
from swe_mas.perception.models import FaultSignal, FaultType


class PerceptionManager:
    """统一调度感知模块，可扩展。"""

    def __init__(
        self,
        heartbeat: HeartbeatMonitor | None = None,
        graph_detector: GraphAnomalyDetector | None = None,
        injector: FaultInjector | None = None,
    ):
        self.heartbeat = heartbeat
        self.graph_detector = graph_detector
        self.injector = injector or FaultInjector()

    def _aggregate_signals(self, signals: list[FaultSignal]) -> list[FaultSignal]:
        """简单聚合：优先外部硬故障，否则合并认知类故障"""
        if not signals:
            return []
        # 优先返回高优先级外部故障
        for sig in signals:
            if sig.severity == "error" and sig.fault_type in {FaultType.NETWORK_ERROR, FaultType.HEARTBEAT_LOSS}:
                return [sig]
        # 合并内部认知类
        cognitive_types = {FaultType.MISALIGNMENT, FaultType.HALLUCINATION, FaultType.LOOP}
        cognitive = [s for s in signals if s.fault_type in cognitive_types]
        if not cognitive:
            return signals
        merged_ids: list[str] = []
        extras: list[dict] = []
        agg_scores = {"loop_score": [], "test_score": [], "churn_score": [], "total_score": []}
        for s in cognitive:
            if s.node_ids:
                merged_ids.extend(s.node_ids)
            if s.extra:
                extras.append(s.extra)
                for k in agg_scores.keys():
                    if isinstance(s.extra, dict) and k in s.extra and isinstance(s.extra[k], (int, float)):
                        agg_scores[k].append(s.extra[k])
        score_summary = {}
        for k, vals in agg_scores.items():
            if vals:
                score_summary[k] = sum(vals) / len(vals)
        extra_payload = {"source": "graph"}
        if extras:
            extra_payload["details"] = extras
        if score_summary:
            extra_payload["scores"] = score_summary
        if getattr(self.graph_detector, "policy", None):
            extra_payload["policy"] = {
                "w_loop": self.graph_detector.policy.w_loop,
                "w_test": self.graph_detector.policy.w_test,
                "w_churn": self.graph_detector.policy.w_churn,
                "threshold": self.graph_detector.policy.threshold,
            }
        merged = FaultSignal(
            fault_type=cognitive[0].fault_type,
            severity=cognitive[0].severity,
            agent_id=cognitive[0].agent_id,
            message=cognitive[0].message,
            suggested_action=cognitive[0].suggested_action,
            extra=extra_payload,
            node_ids=list(dict.fromkeys(merged_ids)) if merged_ids else None,
        )
        return [merged]

    def run(self, *, session_id: str, agents_state: list[dict[str, Any]] | None = None, inject: bool = False) -> list[FaultSignal]:
        signals: list[FaultSignal] = []

        # 1) 心跳检测
        if self.heartbeat and agents_state:
            for st in agents_state:
                last_seen = st.get("last_seen_ts", time.time())
                token_left = st.get("token_left")
                signals.extend(self.heartbeat.check(agent_id=st.get("agent_id", "unknown"), last_seen_ts=last_seen, token_left=token_left))

        # 2) 图异常检测
        if self.graph_detector:
            signals.extend(self.graph_detector.run(session_id=session_id))

        # 3) 故障注入（用于实验）
        if inject and self.injector:
            signals.append(self.injector.inject(agent_id="synthetic"))

        return self._aggregate_signals(signals)
