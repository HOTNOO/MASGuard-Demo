"""节点Schema定义"""

from __future__ import annotations

import enum
from typing import TypedDict, Any


class NodeType(str, enum.Enum):
    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"
    MESSAGE = "MESSAGE"
    PLAN = "PLAN"
    SUMMARY = "SUMMARY"
    CHECKPOINT = "CHECKPOINT"


class MemoryNode(TypedDict, total=False):
    node_id: str
    session_id: str
    agent_id: str
    role: str
    node_type: str
    content: str
    payload: dict[str, Any]
    embedding: list[float]
    timestamp: float
    step_id: str
    iteration: int
    validity: str
    token_usage: float
    confidence: float
    causal_parents: list[str]
    temporal_prev: str | None
    env_snapshot_hash: str | None
    resource_stat: dict[str, Any]
    status: str
    # 细粒度阶段与步骤信息
    phase: str  # e.g., analyze / locate / plan / implement / verify
    step_type: str  # e.g., llm_thought / run_tests / edit_code
    # 观测与摘要辅助字段
    test_status: str  # pass / fail / not_run (only meaningful for test-related nodes)
    files_touched: list[str]
    summary_hint: str  # optional hint for summarization, e.g., suspect_path / healthy_anchor
