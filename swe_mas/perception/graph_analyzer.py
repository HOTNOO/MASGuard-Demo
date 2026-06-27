"""基于记忆DAG的异常检测（循环/幻觉/意图-动作不一致）"""

from __future__ import annotations

from typing import Iterable, Tuple
from dataclasses import dataclass
import json
import re

import numpy as np

from swe_mas.memory import MemoryRecorder
from swe_mas.perception.models import FaultSignal, FaultType
@dataclass
class PerceptionPolicyConfig:
    w_loop: float = 0.4
    w_test: float = 0.3
    w_churn: float = 0.3
    threshold: float = 0.5


class GraphAnomalyDetector:
    """图异常检测，包含：
    - 相似度循环检测（基于 embedding）
    - 大模型检测幻觉/循环
    - 大模型检测意图与动作/观察不一致
    """

    def __init__(
        self,
        recorder: MemoryRecorder,
        model: object | None = None,
        similarity_threshold: float = 0.99,
        window: int = 5,
        min_hits: int = 5,
        loop_weight: float = 0.4,
        test_weight: float = 0.3,
        churn_weight: float = 0.3,
        policy: PerceptionPolicyConfig | None = None,
        perception_policy=None,
    ):
        self.recorder = recorder
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.window = window
        self.min_hits = min_hits
        self.loop_weight = loop_weight
        self.test_weight = test_weight
        self.churn_weight = churn_weight
        self.policy = policy or PerceptionPolicyConfig(
            w_loop=loop_weight,
            w_test=test_weight,
            w_churn=churn_weight,
            threshold=0.5,
        )
        self.perception_policy = perception_policy

    def _cos_sim(self, a: Iterable[float], b: Iterable[float]) -> float:
        va = np.array(a)
        vb = np.array(b)
        if va.size == 0 or vb.size == 0:
            return 0.0
        denom = (np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _norm(self, s: str) -> str:
        s = (s or "").strip().lower()
        return re.sub(r"\s+", " ", s)

    def _compute_similarity_loop_score(self, nodes: list[dict], role: str | None = None) -> Tuple[float, list[str]]:
        if len(nodes) < 2:
            return 0.0, []
        recent = nodes[-self.window :]
        sim_pairs = []
        has_any_vec = any((n.get("embedding") or []) for n in recent)
        if has_any_vec:
            for i in range(len(recent) - 1):
                for j in range(i + 1, len(recent)):
                    sim = self._cos_sim(recent[i].get("embedding", []), recent[j].get("embedding", []))
                    if sim >= self.similarity_threshold:
                        sim_pairs.append((sim, recent[i], recent[j]))
            if sim_pairs:
                score = min(1.0, len(sim_pairs) / max(1, self.min_hits))
                node_ids = []
                for _, a, b in sim_pairs:
                    node_ids.extend([a["node_id"], b["node_id"]])
                unique_ids = list(dict.fromkeys(node_ids))
                return score, unique_ids

        texts = [self._norm(n.get("content", "")) for n in recent]
        dup = 0
        ids = []
        for i in range(len(texts) - 1):
            for j in range(i + 1, len(texts)):
                if texts[i] and texts[i] == texts[j]:
                    dup += 1
                    ids.extend([recent[i]["node_id"], recent[j]["node_id"]])
        if dup == 0:
            return 0.0, []
        score = min(1.0, dup / max(1, self.min_hits))
        return score, list(dict.fromkeys(ids))

    def _compute_test_stagnation_score(self, nodes: list[dict]) -> Tuple[float, list[str]]:
        test_nodes = [n for n in nodes if n.get("step_type") == "test_result" and n.get("test_status") in {"pass", "fail"}]
        if not test_nodes:
            return 0.0, []
        recent = test_nodes[-self.window :]
        fail_runs = [n for n in recent if n.get("test_status") == "fail"]
        total = len(recent)
        score = len(fail_runs) / total if total > 0 else 0.0
        return min(1.0, score), [n["node_id"] for n in recent]

    def _compute_code_churn_score(self, nodes: list[dict]) -> Tuple[float, list[str]]:
        if not nodes:
            return 0.0, []
        snapshots = [n.get("env_snapshot_hash") for n in nodes if n.get("env_snapshot_hash")]
        if not snapshots:
            return 0.0, []
        unique_snaps = list(dict.fromkeys(snapshots))
        churn_ratio = len(unique_snaps) / len(snapshots) if snapshots else 0.0
        # 如果快照变化很少但节点很多，认为陷入无效改动；反之快照变化很多但结果无改善也给中等分
        score = 1.0 - churn_ratio if len(snapshots) > self.window else max(0.0, 0.5 * (1.0 - churn_ratio))
        return max(0.0, min(1.0, score)), [n["node_id"] for n in nodes[-self.window :]]

    def run(self, session_id: str, role: str | None = None) -> list[FaultSignal]:
        nodes = self.recorder.get_nodes(session_id, role=role, limit=80)
        # 只对“当前有效轨迹”做检测：历史 INVALID/failed 分支会被恢复链路保留用于摘要，
        # 但不应继续影响后续分支的循环/停滞判断。
        nodes = [
            n
            for n in nodes
            if (n.get("validity", "VALID") or "").upper() != "INVALID"
            and (n.get("status", "active") or "").lower() != "failed"
        ]
        if not nodes:
            return []

        loop_score, loop_ids = self._compute_similarity_loop_score(nodes, role)
        test_score, test_ids = self._compute_test_stagnation_score(nodes)
        churn_score, churn_ids = self._compute_code_churn_score(nodes)

        selected_pid = None
        bucket_id = None
        if self.perception_policy:
            bucket_id = self.perception_policy.bucket_id(loop_score, test_score, churn_score)
            selected_pid, cfg = self.perception_policy.select_config(loop_score, test_score, churn_score)
            self.policy = cfg

        total_score = (
            self.policy.w_loop * loop_score +
            self.policy.w_test * test_score +
            self.policy.w_churn * churn_score
        )

        if total_score < self.policy.threshold:
            return []

        all_ids = list(dict.fromkeys(loop_ids + test_ids + churn_ids))

        signal = FaultSignal(
            fault_type=FaultType.LOOP,
            severity="warn" if total_score < 0.8 else "error",
            message=f"loop_score={loop_score:.2f}, test_stagnation={test_score:.2f}, code_churn={churn_score:.2f}",
            suggested_action="rollback_to_healthy_anchor",
            extra={
                "source": "graph_detector",
                "loop_score": loop_score,
                "test_score": test_score,
                "churn_score": churn_score,
                "total_score": total_score,
                "policy": {
                    "w_loop": self.policy.w_loop,
                    "w_test": self.policy.w_test,
                    "w_churn": self.policy.w_churn,
                    "threshold": self.policy.threshold,
                    "bucket_id": bucket_id,
                    "policy_id": selected_pid,
                },
            },
            node_ids=all_ids,
        )

        # 记录一次感知策略反馈，供 bandit 进化使用
        try:
            if self.perception_policy and self.recorder and nodes:
                payload = {
                    "kind": "perception",
                    "bucket_id": bucket_id,
                    "policy_id": selected_pid,
                    "scores": {
                        "loop": loop_score,
                        "test": test_score,
                        "churn": churn_score,
                        "total": total_score,
                    },
                }
                self.recorder.log_feedback(
                    node_id=nodes[-1].get("node_id", ""),
                    reward=0.0,
                    strategy=json.dumps(payload, ensure_ascii=False),
                    policy_name="perception_policy_v1",
                    strategy_json=payload,
                )
        except Exception:
            pass

        return [signal]
