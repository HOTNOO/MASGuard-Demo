"""Dependency-light GCRV with heterogeneous graph message passing.

This keeps the BCMR training and serving path self-contained while moving the
selector beyond tabular heuristics. The model consumes three signals:

- failed-state scalar/categorical features
- suspect-subgraph structure and typed nodes/edges
- candidate-action descriptors

The graph encoder is implemented with deterministic heterogeneous message
passing over hashed node vectors so it works in constrained environments
without PyTorch/PyG. The learned part remains a multi-head value model that
predicts recovery success, token cost, latency, and risk.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from bcmr_swe.types import ActionScore, CandidateAction, FailedState, RecoveryBudget, ReplayOutcome


class GraphConditionedRecoveryValueModel:
    """Graph-conditioned recovery selector with typed message-passing features."""

    def __init__(
        self,
        *,
        embedding_dim: int = 24,
        message_passing_steps: int = 2,
        max_content_tokens: int = 6,
    ):
        self.embedding_dim = max(8, int(embedding_dim))
        self.message_passing_steps = max(1, int(message_passing_steps))
        self.max_content_tokens = max(2, int(max_content_tokens))
        self.success_weights: dict[str, float] = defaultdict(float)
        self.token_weights: dict[str, float] = defaultdict(float)
        self.latency_weights: dict[str, float] = defaultdict(float)
        self.risk_weights: dict[str, float] = defaultdict(float)
        self.utility_weights: dict[str, float] = defaultdict(float)
        self.bias = {"success": 0.0, "token": 0.0, "latency": 0.0, "risk": 0.0, "utility": 0.0}

    def fit(
        self,
        groups: list[tuple[FailedState, list[CandidateAction], list[ReplayOutcome]]],
        *,
        epochs: int = 24,
        lr: float = 0.02,
    ) -> None:
        for _epoch in range(epochs):
            for failed_state, actions, outcomes in groups:
                outcome_by_action = {outcome.action_id: outcome for outcome in outcomes}
                ranked_pairs: list[tuple[dict[str, float], float]] = []
                for action in actions:
                    outcome = outcome_by_action.get(action.action_id)
                    if outcome is None:
                        continue
                    feats = self._features(failed_state, action)
                    target_success = 1.0 if outcome.recover_success else 0.0
                    target_token = float(outcome.token_cost)
                    target_latency = float(outcome.latency_sec)
                    target_risk = float(outcome.secondary_risk)
                    target_utility = self._target_utility(outcome)

                    self._sgd_head(self.success_weights, feats, target_success, lr, "success")
                    self._sgd_head(self.token_weights, feats, target_token, lr * 0.001, "token")
                    self._sgd_head(self.latency_weights, feats, target_latency, lr * 0.001, "latency")
                    self._sgd_head(self.risk_weights, feats, target_risk, lr * 0.01, "risk")
                    self._sgd_head(self.utility_weights, feats, target_utility, lr * 0.2, "utility")
                    ranked_pairs.append((feats, target_utility))

                for better, worse in self._iter_rank_pairs(ranked_pairs):
                    self._pairwise_update(better, worse, lr=lr * 0.15)

    def rank(self, failed_state: FailedState, actions: list[CandidateAction], budget: RecoveryBudget) -> list[ActionScore]:
        ranked: list[ActionScore] = []
        for action in actions:
            features = self._features(failed_state, action)
            recover = self._sigmoid(self._dot(self.success_weights, features) + self.bias["success"])
            token_cost = max(
                0.0,
                self._dot(self.token_weights, features) + self.bias["token"] + float(action.estimated_token_cost),
            )
            latency = max(
                0.0,
                self._dot(self.latency_weights, features) + self.bias["latency"] + float(action.estimated_latency_sec),
            )
            risk = max(
                0.0,
                min(1.0, self._dot(self.risk_weights, features) + self.bias["risk"] + float(action.estimated_risk)),
            )
            utility = (
                recover
                - budget.lambda_token * token_cost
                - budget.lambda_latency * latency
                - budget.lambda_risk * risk
                + 0.10 * (self._dot(self.utility_weights, features) + self.bias["utility"])
            )
            ranked.append(
                ActionScore(
                    action_id=action.action_id,
                    utility=utility,
                    estimated_recover_prob=recover,
                    estimated_token_cost=token_cost,
                    estimated_latency_sec=latency,
                    estimated_risk=risk,
                    explanation="gcrv_hetero_message_passing",
                )
            )
        ranked.sort(key=lambda item: item.utility, reverse=True)
        return ranked

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "config": {
                "embedding_dim": self.embedding_dim,
                "message_passing_steps": self.message_passing_steps,
                "max_content_tokens": self.max_content_tokens,
            },
            "success_weights": dict(self.success_weights),
            "token_weights": dict(self.token_weights),
            "latency_weights": dict(self.latency_weights),
            "risk_weights": dict(self.risk_weights),
            "utility_weights": dict(self.utility_weights),
            "bias": dict(self.bias),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GraphConditionedRecoveryValueModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = payload.get("config", {})
        model = cls(
            embedding_dim=int(config.get("embedding_dim", 24)),
            message_passing_steps=int(config.get("message_passing_steps", 2)),
            max_content_tokens=int(config.get("max_content_tokens", 6)),
        )
        model.success_weights.update(payload.get("success_weights", {}))
        model.token_weights.update(payload.get("token_weights", {}))
        model.latency_weights.update(payload.get("latency_weights", {}))
        model.risk_weights.update(payload.get("risk_weights", {}))
        model.utility_weights.update(payload.get("utility_weights", {}))
        model.bias.update(payload.get("bias", {}))
        return model

    def _features(self, failed_state: FailedState, action: CandidateAction) -> dict[str, float]:
        features: dict[str, float] = {}

        for key, value in failed_state.state_features.numeric.items():
            features[f"state::{key}"] = float(value)
        for key, value in failed_state.state_features.categorical.items():
            features[f"cat::{key}::{value}"] = 1.0

        for key, value in failed_state.suspect_region.summary.items():
            if isinstance(value, bool):
                features[f"region::{key}"] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                features[f"region::{key}"] = float(value)
            elif isinstance(value, str):
                features[f"region::{key}::{value}"] = 1.0
            elif isinstance(value, list):
                features[f"region::{key}::len"] = float(len(value))
                for item in value[:4]:
                    features[f"region::{key}::{item}"] = 1.0

        features[f"action::{action.action_type.value}"] = 1.0
        features["action::estimated_recover_prob"] = float(action.estimated_recover_prob)
        features["action::estimated_token_cost"] = float(action.estimated_token_cost) / 1000.0
        features["action::estimated_latency_sec"] = float(action.estimated_latency_sec) / 100.0
        features["action::estimated_risk"] = float(action.estimated_risk)
        features.update(self._action_payload_features(action))
        features.update(self._graph_features(failed_state))
        return features

    def _action_payload_features(self, action: CandidateAction) -> dict[str, float]:
        payload = dict(action.payload)
        features: dict[str, float] = {
            "action_payload::has_checkpoint": 1.0 if payload.get("checkpoint_id") else 0.0,
            "action_payload::has_fact_node": 1.0 if payload.get("fact_node_id") else 0.0,
        }
        resume_from = payload.get("resume_from")
        if resume_from:
            features[f"action_payload::resume_from::{resume_from}"] = 1.0
        target_role = payload.get("target_role")
        if target_role:
            features[f"action_payload::target_role::{target_role}"] = 1.0
        deep_verify = payload.get("deep_verify")
        if isinstance(deep_verify, bool):
            features["action_payload::deep_verify"] = 1.0 if deep_verify else 0.0
        escalation_level = payload.get("escalation_level")
        if isinstance(escalation_level, (int, float)):
            features["action_payload::escalation_level"] = float(escalation_level)
        suspect_node_ids = payload.get("suspect_node_ids")
        if isinstance(suspect_node_ids, list):
            features["action_payload::suspect_node_count"] = float(len(suspect_node_ids))
        return features

    def _graph_features(self, failed_state: FailedState) -> dict[str, float]:
        features: dict[str, float] = {}
        graph_payload = failed_state.metadata.get("suspect_graph", {})
        if not isinstance(graph_payload, dict):
            features["graph::missing"] = 1.0
            return features

        raw_nodes = graph_payload.get("nodes", [])
        raw_edges = graph_payload.get("edges", [])
        if not isinstance(raw_nodes, list) or not raw_nodes:
            features["graph::missing"] = 1.0
            return features

        nodes = [node for node in raw_nodes if isinstance(node, dict) and node.get("node_id")]
        node_ids = {str(node["node_id"]) for node in nodes}
        edges = [
            edge
            for edge in raw_edges
            if isinstance(edge, dict) and edge.get("source_id") in node_ids and edge.get("target_id") in node_ids
        ]
        if not nodes:
            features["graph::missing"] = 1.0
            return features

        features["graph::num_nodes"] = float(len(nodes))
        features["graph::num_edges"] = float(len(edges))
        features["graph::density"] = float(len(edges)) / max(1.0, float(len(nodes) * max(1, len(nodes) - 1)))
        features["graph::trigger_in_region"] = 1.0 if failed_state.trigger.trigger_node_id in node_ids else 0.0

        kind_counts = Counter(str(node.get("kind", "unknown")) for node in nodes)
        role_counts = Counter(str(node.get("role", "unknown")) for node in nodes)
        phase_counts = Counter(str(node.get("phase", "unknown")) for node in nodes)
        status_counts = Counter(str(node.get("status", "unknown")) for node in nodes)
        for key, value in kind_counts.items():
            features[f"graph::node_kind::{key}"] = float(value) / len(nodes)
        for key, value in role_counts.items():
            features[f"graph::node_role::{key}"] = float(value) / len(nodes)
        for key, value in phase_counts.items():
            features[f"graph::node_phase::{key}"] = float(value) / len(nodes)
        for key, value in status_counts.items():
            features[f"graph::node_status::{key}"] = float(value) / len(nodes)

        indegree = Counter()
        outdegree = Counter()
        motif_counts = Counter()
        edge_kind_counts = Counter()
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in edges:
            source_id = str(edge.get("source_id", ""))
            target_id = str(edge.get("target_id", ""))
            kind = str(edge.get("kind", "unknown"))
            indegree[target_id] += 1
            outdegree[source_id] += 1
            edge_kind_counts[kind] += 1
            adjacency[source_id].append((target_id, kind))
            source_kind = str(self._node_value(nodes, source_id, "kind", default="unknown"))
            target_kind = str(self._node_value(nodes, target_id, "kind", default="unknown"))
            motif_counts[f"{source_kind}|{kind}|{target_kind}"] += 1
        for key, value in edge_kind_counts.items():
            features[f"graph::edge_kind::{key}"] = float(value) / max(1, len(edges))
        for key, value in motif_counts.items():
            features[f"graph::motif::{key}"] = float(value) / max(1, len(edges))

        degree_values = [float(indegree[node_id] + outdegree[node_id]) for node_id in node_ids]
        if degree_values:
            features["graph::degree_mean"] = sum(degree_values) / len(degree_values)
            features["graph::degree_max"] = max(degree_values)
            features["graph::isolated_ratio"] = sum(1.0 for value in degree_values if value == 0.0) / len(degree_values)

        two_hop_counts = Counter()
        for source_id, neighbors in adjacency.items():
            source_kind = str(self._node_value(nodes, source_id, "kind", default="unknown"))
            for mid_id, first_kind in neighbors:
                mid_kind = str(self._node_value(nodes, mid_id, "kind", default="unknown"))
                for dst_id, second_kind in adjacency.get(mid_id, []):
                    dst_kind = str(self._node_value(nodes, dst_id, "kind", default="unknown"))
                    token = f"{source_kind}|{first_kind}|{mid_kind}|{second_kind}|{dst_kind}"
                    two_hop_counts[token] += 1
        for key, value in two_hop_counts.items():
            features[f"graph::path2::{key}"] = float(value) / max(1, sum(two_hop_counts.values()))

        pooled_vectors = self._encode_graph(nodes, edges)
        for layer_index, vector in enumerate(pooled_vectors, start=1):
            for dim_index, value in enumerate(vector):
                features[f"graph_emb::layer{layer_index}::{dim_index}"] = value
            features[f"graph_emb::layer{layer_index}::l2_norm"] = math.sqrt(sum(item * item for item in vector))
        return features

    def _encode_graph(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[list[float]]:
        node_vectors = {
            str(node["node_id"]): self._hashed_vector(self._node_tokens(node), self.embedding_dim)
            for node in nodes
        }
        indegree = Counter(str(edge.get("target_id", "")) for edge in edges)
        outdegree = Counter(str(edge.get("source_id", "")) for edge in edges)
        pooled_vectors: list[list[float]] = []

        for step in range(self.message_passing_steps):
            incoming_messages = {node_id: [0.0] * self.embedding_dim for node_id in node_vectors}
            outgoing_messages = {node_id: [0.0] * self.embedding_dim for node_id in node_vectors}
            for edge in edges:
                source_id = str(edge.get("source_id", ""))
                target_id = str(edge.get("target_id", ""))
                kind = str(edge.get("kind", "unknown"))
                if source_id not in node_vectors or target_id not in node_vectors:
                    continue
                inbound = self._relation_transform(node_vectors[source_id], kind, direction="forward", step=step)
                outbound = self._relation_transform(node_vectors[target_id], kind, direction="reverse", step=step)
                self._accumulate(
                    incoming_messages[target_id],
                    inbound,
                    scale=1.0 / max(1.0, float(indegree[target_id])),
                )
                self._accumulate(
                    outgoing_messages[source_id],
                    outbound,
                    scale=1.0 / max(1.0, float(outdegree[source_id])),
                )

            next_vectors: dict[str, list[float]] = {}
            for node_id, vector in node_vectors.items():
                combined = [0.0] * self.embedding_dim
                for index in range(self.embedding_dim):
                    combined[index] = (
                        0.55 * vector[index]
                        + 0.35 * incoming_messages[node_id][index]
                        + 0.10 * outgoing_messages[node_id][index]
                    )
                next_vectors[node_id] = self._normalize_vector(combined)
            node_vectors = next_vectors
            pooled_vectors.append(self._pool_vectors(node_vectors.values()))
        return pooled_vectors

    def _node_tokens(self, node: dict[str, Any]) -> list[str]:
        payload = node.get("payload") if isinstance(node.get("payload"), dict) else {}
        tokens = [
            f"kind={node.get('kind', 'unknown')}",
            f"role={node.get('role', 'unknown')}",
            f"phase={node.get('phase', 'unknown')}",
            f"status={node.get('status', 'active')}",
            f"validity={node.get('validity', 'valid')}",
        ]
        for key in ("fact_key", "verdict", "test_status", "failure_signature", "returncode"):
            value = payload.get(key)
            if value not in (None, "", []):
                tokens.append(f"payload::{key}={value}")
        for key in ("failing_tests", "files_touched", "contradicted_fact_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                tokens.append(f"payload::{key}_count={min(len(value), 8)}")
        content = str(node.get("content", ""))
        for token in self._content_tokens(content)[: self.max_content_tokens]:
            tokens.append(f"content={token}")
        return tokens

    def _content_tokens(self, text: str) -> list[str]:
        cleaned = re.findall(r"[A-Za-z0-9_./:-]+", text.lower())
        return [token for token in cleaned if len(token) >= 3][: self.max_content_tokens]

    def _hashed_vector(self, tokens: Iterable[str], dim: int) -> list[float]:
        vector = [0.0] * dim
        for token in tokens:
            digest = self._stable_digest(token)
            index = digest % dim
            sign = -1.0 if ((digest >> 8) & 1) else 1.0
            scale = 1.0 + float((digest >> 16) % 5) * 0.1
            vector[index] += sign * scale
        return self._normalize_vector(vector)

    def _relation_transform(self, vector: list[float], relation: str, *, direction: str, step: int) -> list[float]:
        digest = self._stable_digest(f"{relation}:{direction}:{step}")
        shift = 1 + digest % max(1, self.embedding_dim - 1)
        sign = -1.0 if ((digest >> 12) & 1) else 1.0
        gain = 0.75 + float((digest >> 20) % 6) * 0.05
        transformed = [0.0] * self.embedding_dim
        for index, value in enumerate(vector):
            transformed[(index + shift) % self.embedding_dim] += sign * gain * value
        return transformed

    def _pool_vectors(self, vectors: Iterable[list[float]]) -> list[float]:
        vector_list = list(vectors)
        if not vector_list:
            return [0.0] * self.embedding_dim
        pooled = [0.0] * self.embedding_dim
        for vector in vector_list:
            for index, value in enumerate(vector):
                pooled[index] += value
        count = float(len(vector_list))
        return [value / count for value in pooled]

    def _accumulate(self, base: list[float], update: list[float], *, scale: float) -> None:
        for index, value in enumerate(update):
            base[index] += value * scale

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-9:
            return vector
        return [value / norm for value in vector]

    def _iter_rank_pairs(self, ranked_pairs: list[tuple[dict[str, float], float]]) -> Iterable[tuple[tuple[dict[str, float], float], tuple[dict[str, float], float]]]:
        ordered = sorted(ranked_pairs, key=lambda item: item[1], reverse=True)
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                if abs(ordered[left_index][1] - ordered[right_index][1]) < 1e-9:
                    continue
                yield ordered[left_index], ordered[right_index]

    def _pairwise_update(
        self,
        better: tuple[dict[str, float], float],
        worse: tuple[dict[str, float], float],
        *,
        lr: float,
        margin: float = 0.25,
    ) -> None:
        better_feats, _better_utility = better
        worse_feats, _worse_utility = worse
        better_score = self._dot(self.utility_weights, better_feats) + self.bias["utility"]
        worse_score = self._dot(self.utility_weights, worse_feats) + self.bias["utility"]
        if better_score - worse_score >= margin:
            return
        diff = {
            key: better_feats.get(key, 0.0) - worse_feats.get(key, 0.0)
            for key in set(better_feats) | set(worse_feats)
        }
        for key, value in diff.items():
            self.utility_weights[key] += lr * value
            self.success_weights[key] += 0.2 * lr * value
        self.bias["utility"] += lr
        self.bias["success"] += 0.1 * lr

    def _sgd_head(self, weights: dict[str, float], feats: dict[str, float], target: float, lr: float, head: str) -> None:
        pred = self._dot(weights, feats) + self.bias[head]
        error = target - pred
        for key, value in feats.items():
            weights[key] += lr * error * value
        self.bias[head] += lr * error

    def _target_utility(self, outcome: ReplayOutcome) -> float:
        return (
            (1.0 if outcome.recover_success else 0.0)
            - 1e-4 * outcome.token_cost
            - 5e-4 * outcome.latency_sec
            - 0.2 * outcome.secondary_risk
            + 0.1 * outcome.milestone_gain
        )

    def _dot(self, weights: dict[str, float], feats: dict[str, float]) -> float:
        return sum(weights.get(key, 0.0) * value for key, value in feats.items())

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            z = 1.0 / (1.0 + math.exp(-value))
            return z
        z = math.exp(value)
        return z / (1.0 + z)

    def _node_value(self, nodes: list[dict[str, Any]], node_id: str, key: str, *, default: Any) -> Any:
        for node in nodes:
            if str(node.get("node_id", "")) == node_id:
                return node.get(key, default)
        return default

    def _stable_digest(self, text: str) -> int:
        return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)
