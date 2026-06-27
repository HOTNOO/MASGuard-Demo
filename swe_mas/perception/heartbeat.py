"""心跳与资源监控"""

from __future__ import annotations

import time
from typing import Callable

from swe_mas.perception.models import FaultSignal, FaultType


class HeartbeatMonitor:
    """检测 Agent 存活和资源状态。"""

    def __init__(
        self,
        timeout_sec: int = 120,
        min_token_left: int | None = None,
        health_check: Callable[[], bool] | None = None,
    ):
        self.timeout_sec = timeout_sec
        self.min_token_left = min_token_left
        self.health_check = health_check

    def check(self, *, agent_id: str, last_seen_ts: float, token_left: int | None = None) -> list[FaultSignal]:
        now = time.time()
        signals: list[FaultSignal] = []

        if now - last_seen_ts > self.timeout_sec:
            signals.append(FaultSignal(
                fault_type=FaultType.HEARTBEAT_LOSS,
                severity="error",
                agent_id=agent_id,
                message="Agent heartbeat timeout",
                suggested_action="restart_agent",
                extra={
                    "source": "heartbeat",
                    "timeout_sec": self.timeout_sec,
                    "last_seen_ts": last_seen_ts,
                },
            ))

        if self.min_token_left is not None and token_left is not None and token_left < self.min_token_left:
            signals.append(FaultSignal(
                fault_type=FaultType.TOKEN_EXHAUSTED,
                severity="warn",
                agent_id=agent_id,
                message=f"Token left low: {token_left}",
                suggested_action="use_backup_model",
                extra={
                    "source": "heartbeat",
                    "min_token_left": self.min_token_left,
                    "token_left": token_left,
                },
            ))

        if self.health_check:
            try:
                ok = self.health_check()
            except Exception:
                ok = False
            if not ok:
                signals.append(FaultSignal(
                    fault_type=FaultType.EXTERNAL,
                    severity="error",
                    agent_id=agent_id,
                    message="Custom health check failed",
                    suggested_action="switch_to_backup",
                    extra={"source": "heartbeat", "health_check": "custom"},
                ))

        return signals
