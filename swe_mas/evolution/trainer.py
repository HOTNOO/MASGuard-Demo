from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List

from .bandit import ArmStats, BetaBandit
from .policy_store import PolicyStore


def _iter_feedback_rows(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT f.node_id, f.reward, f.strategy_json, n.session_id
        FROM feedback f
        LEFT JOIN nodes n ON f.node_id = n.node_id
        """
    )
    for row in cursor.fetchall():
        node_id, reward, strategy_json, session_id = row
        try:
            data = json.loads(strategy_json) if strategy_json else {}
        except Exception:
            data = {}
        yield {
            "node_id": node_id,
            "reward": float(reward),
            "session_id": session_id,
            "data": data,
        }


def train_policy(db_path: Path, policy_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    store = PolicyStore.load(policy_path)
    bandit = BetaBandit()

    summary_updates: Dict[tuple[str, str], float] = {}
    perception_updates: Dict[tuple[str, str], float] = {}

    rows: List[dict] = list(_iter_feedback_rows(conn))
    session_fault_max: Dict[str, float] = {}
    for row in rows:
        data = row["data"]
        session_id = row.get("session_id")
        fault_idx = data.get("fault_index") or data.get("context", {}).get("fault_index")
        if session_id and fault_idx is not None:
            current = session_fault_max.get(session_id, 0.0)
            session_fault_max[session_id] = max(current, float(fault_idx))

    for row in rows:
        reward = row["reward"]
        data = row["data"]
        session_id = row.get("session_id")

        if not data:
            continue

        kind = data.get("kind") or data.get("type")
        fault_idx = data.get("fault_index") or data.get("context", {}).get("fault_index")
        fault_count = session_fault_max.get(session_id)
        weight = 1.0
        if fault_idx is not None and fault_count:
            fault_idx = float(fault_idx)
            if reward > 0:
                weight = max(0.1, (fault_count - fault_idx + 1) / fault_count)
            else:
                weight = max(0.1, fault_idx / fault_count)

        if kind == "summary":
            bucket = data.get("bucket_id")
            strategy_id = data.get("strategy_id")
            if not bucket or not strategy_id:
                continue
            summary_updates[(bucket, strategy_id)] = summary_updates.get((bucket, strategy_id), 0.0) + reward * weight

        elif kind == "perception":
            bucket = data.get("bucket_id")
            policy_id = data.get("policy_id")
            if not bucket or not policy_id:
                continue
            perception_updates[(bucket, policy_id)] = perception_updates.get((bucket, policy_id), 0.0) + reward * weight

    for (bucket, sid), r in summary_updates.items():
        if bucket not in store.summary.buckets:
            store.summary.buckets[bucket] = {}
        arms = store.summary.buckets[bucket]
        if sid not in arms:
            arms[sid] = ArmStats()
        bandit.update(arms, sid, r, weight=abs(r))

    for (bucket, pid), r in perception_updates.items():
        if bucket not in store.perception.buckets:
            store.perception.buckets[bucket] = {}
        arms = store.perception.buckets[bucket]
        if pid not in arms:
            arms[pid] = ArmStats()
        bandit.update(arms, pid, r, weight=abs(r))

    store.save()
    print(f"Updated policy stored at: {policy_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True, help="Path to memory SQLite db")
    parser.add_argument("--policy", type=Path, required=True, help="Path to evolution_policy.json")
    args = parser.parse_args()

    train_policy(args.db, args.policy)


if __name__ == "__main__":
    main()
