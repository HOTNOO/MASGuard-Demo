"""记忆DAG记录器 - SQLite后端，支持可插拔嵌入和反馈埋点"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable
import threading

from swe_mas.memory.schemas import MemoryNode
from swe_mas.memory.embedder import load_embedder


def _default_embedder(text: str) -> list[float]:
    """简单占位嵌入：将文本哈希为固定长度向量."""
    if not text:
        return []
    vec = []
    for i, b in enumerate(text.encode("utf-8")[:64]):
        vec.append(((b + i * 17) % 101) / 100.0)
    return vec


class MemoryRecorder:
    """记忆节点持久化与查询"""

    def __init__(self, db_path: str | Path, embedder: Callable[[str], Iterable[float]] | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.embedder = embedder or load_embedder() or _default_embedder
        self._init_db()
        self._subscribers: list[Callable[[str], None]] = []
        self._event = threading.Event()
        # 追踪当前活跃Phase的节点ID
        self._active_phase_nodes: dict[int, list[str]] = {}  # {snapshot_id: [node_ids]}

    def subscribe(self, callback: Callable[[str], None]):
        self._subscribers.append(callback)

    def wait_for_event(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout=timeout)

    def clear_event(self):
        self._event.clear()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                session_id TEXT,
                agent_id TEXT,
                role TEXT,
                node_type TEXT,
                content TEXT,
                payload TEXT,
                embedding TEXT,
                timestamp REAL,
                step_id TEXT,
                iteration INTEGER,
                validity TEXT,
                token_usage REAL,
                confidence REAL,
                causal_parents TEXT,
                temporal_prev TEXT,
                env_snapshot_hash TEXT,
                resource_stat TEXT,
                status TEXT,
                phase TEXT,
                step_type TEXT,
                test_status TEXT,
                files_touched TEXT,
                summary_hint TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_role ON nodes(role);")
        existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(nodes);").fetchall()}
        new_cols = [
            ("phase", "TEXT"),
            ("step_type", "TEXT"),
            ("test_status", "TEXT"),
            ("files_touched", "TEXT"),
            ("summary_hint", "TEXT"),
        ]
        for col, col_type in new_cols:
            if col not in existing_cols:
                cur.execute(f"ALTER TABLE nodes ADD COLUMN {col} {col_type};")
        
        # 创建Phase快照表（用于快速恢复）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS phase_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                role TEXT NOT NULL,
                start_node_id TEXT,
                end_node_id TEXT,
                node_ids TEXT,
                inputs TEXT,
                outputs TEXT,
                env_snapshot_start TEXT,
                env_snapshot_end TEXT,
                validity TEXT DEFAULT 'VALID',
                success INTEGER,
                timestamp_start REAL,
                timestamp_end REAL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_phase_session ON phase_snapshots(session_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_phase_validity ON phase_snapshots(session_id, validity);")
        
        # 迁移：为已存在的表添加 node_ids 列
        existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(phase_snapshots);").fetchall()}
        if "node_ids" not in existing_cols:
            cur.execute("ALTER TABLE phase_snapshots ADD COLUMN node_ids TEXT;")
        
        self.conn.commit()

    def append(self, node: MemoryNode) -> str:
        node_id = node.get("node_id") or str(uuid.uuid4())
        text_for_embed = node.get("content", "")
        embedding = node.get("embedding")
        if embedding is None and self.embedder:
            try:
                embedding = list(self.embedder(text_for_embed))
            except Exception:
                embedding = []

        payload = node.get("payload") or {}
        causal_parents = node.get("causal_parents") or []
        files_touched = json.dumps(node.get("files_touched") or [], ensure_ascii=False)

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO nodes (
                node_id, session_id, agent_id, role, node_type, content, payload, embedding,
                timestamp, step_id, iteration, validity, token_usage, confidence,
                causal_parents, temporal_prev, env_snapshot_hash, resource_stat, status,
                phase, step_type, test_status, files_touched, summary_hint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                node.get("session_id", ""),
                node.get("agent_id", ""),
                node.get("role", ""),
                node.get("node_type", ""),
                node.get("content", ""),
                json.dumps(payload, ensure_ascii=False),
                json.dumps(embedding, ensure_ascii=False),
                node.get("timestamp", time.time()),
                node.get("step_id"),
                node.get("iteration"),
                node.get("validity", "VALID"),
                node.get("token_usage"),
                node.get("confidence"),
                json.dumps(causal_parents, ensure_ascii=False),
                node.get("temporal_prev"),
                node.get("env_snapshot_hash"),
                json.dumps(node.get("resource_stat") or {}, ensure_ascii=False),
                node.get("status", "active"),
                node.get("phase"),
                node.get("step_type"),
                node.get("test_status"),
                files_touched,
                node.get("summary_hint"),
            ),
        )
        self.conn.commit()
        
        # 追踪节点到活跃Phase（如果有）
        phase = node.get("phase")
        if phase:
            # 查找当前session和phase对应的最新活跃快照
            for snapshot_id, node_list in self._active_phase_nodes.items():
                # 简单关联：只要phase匹配就添加
                # 更精确的方式是检查session_id和phase是否匹配
                node_list.append(node_id)
        
        # 通知订阅者
        for cb in self._subscribers:
            try:
                cb(node_id)
            except Exception:
                pass
        self._event.set()
        return node_id

    def get_nodes(self, session_id: str, role: str | None = None, limit: int | None = None, phase: str | None = None) -> list[dict]:
        """按 session/role/phase 查询节点；limit 时取“最新 N 条”"""
        cur = self.conn.cursor()
        sql = "SELECT * FROM nodes WHERE session_id = ?"
        params: list[Any] = [session_id]
        if role:
            sql += " AND role = ?"
            params.append(role)
        if phase:
            sql += " AND phase = ?"
            params.append(phase)

        if limit is not None:
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(int(limit))
            rows = cur.execute(sql, params).fetchall()
            rows = list(reversed(rows))  # 恢复为时间正序
        else:
            sql += " ORDER BY timestamp ASC"
            rows = cur.execute(sql, params).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def get_nodes_filtered(
        self,
        session_id: str,
        *,
        role: str | None = None,
        node_types: list[str] | None = None,
        step_types: list[str] | None = None,
        test_status: list[str] | None = None,
        phase: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """按 node_type / step_type / test_status / phase 过滤的通用查询"""
        cur = self.conn.cursor()
        sql = "SELECT * FROM nodes WHERE session_id = ?"
        params: list[Any] = [session_id]
        if role:
            sql += " AND role = ?"
            params.append(role)
        if node_types:
            placeholders = ",".join("?" for _ in node_types)
            sql += f" AND node_type IN ({placeholders})"
            params.extend(node_types)
        if step_types:
            placeholders = ",".join("?" for _ in step_types)
            sql += f" AND step_type IN ({placeholders})"
            params.extend(step_types)
        if test_status:
            placeholders = ",".join("?" for _ in test_status)
            sql += f" AND test_status IN ({placeholders})"
            params.extend(test_status)
        if phase:
            sql += " AND phase = ?"
            params.append(phase)

        if limit is not None:
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(int(limit))
            rows = cur.execute(sql, params).fetchall()
            rows = list(reversed(rows))
        else:
            sql += " ORDER BY timestamp ASC"
            rows = cur.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_subgraph(self, session_id: str, root_id: str | None = None, depth: int = 2, role: str | None = None) -> dict:
        nodes = self.get_nodes(session_id, role=role)
        node_map = {n["node_id"]: n for n in nodes}
        if not nodes:
            return {"nodes": [], "edges": []}

        if root_id is None:
            root_id = nodes[-1]["node_id"]

        visited = set()
        queue = [(root_id, 0)]
        result_nodes = {}
        edges = []

        while queue:
            nid, d = queue.pop(0)
            if nid in visited or d > depth:
                continue
            visited.add(nid)
            node = node_map.get(nid)
            if not node:
                continue
            result_nodes[nid] = node
            for parent in node.get("causal_parents", []) or []:
                edges.append({"from": parent, "to": nid, "type": "CAUSED_BY"})
                queue.append((parent, d + 1))
            if node.get("temporal_prev"):
                prev_id = node["temporal_prev"]
                edges.append({"from": prev_id, "to": nid, "type": "FOLLOWS"})
                queue.append((prev_id, d + 1))

        return {"nodes": list(result_nodes.values()), "edges": edges}

    def update_validity(self, node_ids: list[str], validity: str = "INVALID", status: str = "failed") -> None:
        if not node_ids:
            return
        placeholders = ",".join("?" for _ in node_ids)
        cur = self.conn.cursor()
        cur.execute(
            f"UPDATE nodes SET validity=?, status=? WHERE node_id IN ({placeholders})",
            [validity, status, *node_ids],
        )
        self.conn.commit()

    def get_snapshots(self, session_id: str) -> list[tuple[str, str]]:
        """按时间顺序返回 (node_id, env_snapshot_hash) 列表"""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT node_id, env_snapshot_hash FROM nodes WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        )
        rows = cur.fetchall()
        return [(r["node_id"], r["env_snapshot_hash"]) for r in rows]

    def get_token_usage_sum(self, session_id: str) -> float:
        """统计指定 session 的 token_usage 总和（缺失则为 0）。"""
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT SUM(token_usage) AS total FROM nodes WHERE session_id = ? AND token_usage IS NOT NULL",
            (session_id,),
        ).fetchone()
        if not row or row["total"] is None:
            return 0.0
        try:
            return float(row["total"])
        except Exception:
            return 0.0

    def latest_session(self) -> str | None:
        cur = self.conn.cursor()
        row = cur.execute("SELECT session_id FROM nodes ORDER BY timestamp DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def log_feedback(self, node_id: str, reward: float, strategy: str | None = None, policy_name: str | None = None, strategy_json: dict | None = None):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT,
                reward REAL,
                strategy TEXT,
                policy_name TEXT,
                strategy_json TEXT,
                timestamp REAL
            );
            """
        )
        existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(feedback);").fetchall()}
        for col, col_type in [("policy_name", "TEXT"), ("strategy_json", "TEXT")]:
            if col not in existing_cols:
                cur.execute(f"ALTER TABLE feedback ADD COLUMN {col} {col_type};")
        cur.execute(
            "INSERT INTO feedback (node_id, reward, strategy, policy_name, strategy_json, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                node_id,
                reward,
                strategy,
                policy_name,
                json.dumps(strategy_json, ensure_ascii=False) if strategy_json is not None else None,
                time.time(),
            ),
        )
        self.conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        return {
            "node_id": row["node_id"],
            "session_id": row["session_id"],
            "agent_id": row["agent_id"],
            "role": row["role"],
            "node_type": row["node_type"],
            "content": row["content"],
            "payload": json.loads(row["payload"] or "{}"),
            "embedding": json.loads(row["embedding"] or "[]"),
            "timestamp": row["timestamp"],
            "step_id": row["step_id"],
            "iteration": row["iteration"],
            "validity": row["validity"],
            "token_usage": row["token_usage"],
            "confidence": row["confidence"],
            "causal_parents": json.loads(row["causal_parents"] or "[]"),
            "temporal_prev": row["temporal_prev"],
            "env_snapshot_hash": row["env_snapshot_hash"],
            "resource_stat": json.loads(row["resource_stat"] or "{}"),
            "status": row["status"],
            "phase": row["phase"],
            "step_type": row["step_type"],
            "test_status": row["test_status"],
            "files_touched": json.loads(row["files_touched"] or "[]"),
            "summary_hint": row["summary_hint"],
        }

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def apply_episode_reward(self, session_id: str, episode_reward: float) -> None:
        """将本 session 中 reward=0 的反馈统一更新为 episode_reward."""
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE feedback
            SET reward = ?
            WHERE node_id IN (
                SELECT node_id FROM nodes WHERE session_id = ?
            ) AND reward = 0.0
            """,
            (episode_reward, session_id),
        )
        self.conn.commit()

    def start_phase(self, session_id: str, phase: str, role: str, inputs: dict, start_node_id: str | None = None) -> int:
        """记录Phase开始（用于恢复）"""
        from swe_mas.utils.git_utils import get_env_snapshot
        from swe_mas.utils.logger import get_logger
        logger = get_logger(__name__)
        
        cur = self.conn.cursor()
        env_snapshot = get_env_snapshot(inputs.get("cwd", "."))
        cur.execute(
            """
            INSERT INTO phase_snapshots 
            (session_id, phase, role, start_node_id, inputs, env_snapshot_start, validity, timestamp_start)
            VALUES (?, ?, ?, ?, ?, ?, 'VALID', ?)
            """,
            (session_id, phase, role, start_node_id, json.dumps(inputs, ensure_ascii=False), env_snapshot, time.time()),
        )
        self.conn.commit()
        snapshot_id = cur.lastrowid
        
        # 初始化该Phase的节点追踪列表
        self._active_phase_nodes[snapshot_id] = []
        
        logger.debug(f"[CVC] Phase快照开始 - ID={snapshot_id}, phase={phase}, role={role}, snapshot={env_snapshot[:12] if env_snapshot else 'none'}")
        return snapshot_id

    def end_phase(self, snapshot_id: int, outputs: dict, success: bool, end_node_id: str | None = None, cwd: str = ".") -> None:
        """记录Phase结束"""
        from swe_mas.utils.git_utils import get_env_snapshot
        from swe_mas.utils.logger import get_logger
        logger = get_logger(__name__)
        
        cur = self.conn.cursor()
        env_snapshot = get_env_snapshot(cwd)
        
        # 获取该Phase产生的所有节点ID
        node_ids = self._active_phase_nodes.get(snapshot_id, [])
        node_ids_json = json.dumps(node_ids, ensure_ascii=False)
        
        cur.execute(
            """
            UPDATE phase_snapshots
            SET end_node_id = ?, outputs = ?, env_snapshot_end = ?, success = ?, timestamp_end = ?, node_ids = ?
            WHERE id = ?
            """,
            (end_node_id, json.dumps(outputs, ensure_ascii=False), env_snapshot, 1 if success else 0, time.time(), node_ids_json, snapshot_id),
        )
        self.conn.commit()
        
        # 清理追踪记录
        if snapshot_id in self._active_phase_nodes:
            del self._active_phase_nodes[snapshot_id]
        
        logger.debug(f"[CVC] Phase快照结束 - ID={snapshot_id}, success={success}, node_count={len(node_ids)}, snapshot={env_snapshot[:12] if env_snapshot else 'none'}")

    def get_phase_snapshot(self, session_id: str, phase: str | None = None, validity: str = "VALID") -> dict | None:
        """获取最后一个有效的Phase快照"""
        cur = self.conn.cursor()
        sql = "SELECT * FROM phase_snapshots WHERE session_id = ? AND validity = ?"
        params = [session_id, validity]
        if phase:
            sql += " AND phase = ?"
            params.append(phase)
        sql += " ORDER BY timestamp_start DESC LIMIT 1"
        row = cur.execute(sql, params).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "phase": row["phase"],
            "role": row["role"],
            "start_node_id": row["start_node_id"],
            "end_node_id": row["end_node_id"],
            "node_ids": json.loads(row["node_ids"] or "[]"),
            "inputs": json.loads(row["inputs"] or "{}"),
            "outputs": json.loads(row["outputs"] or "{}") if row["outputs"] else None,
            "env_snapshot_start": row["env_snapshot_start"],
            "env_snapshot_end": row["env_snapshot_end"],
            "validity": row["validity"],
            "success": bool(row["success"]),
            "timestamp_start": row["timestamp_start"],
            "timestamp_end": row["timestamp_end"],
        }

    def mark_phase_invalid(self, session_id: str, phase: str) -> None:
        """标记Phase为无效（发生故障时）"""
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE phase_snapshots SET validity = 'INVALID' WHERE session_id = ? AND phase = ? AND validity = 'VALID'",
            (session_id, phase),
        )
        self.conn.commit()
