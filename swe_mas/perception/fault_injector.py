"""故障注入，用于实验场景模拟"""

from __future__ import annotations

from swe_mas.perception.models import FaultSignal, FaultType


class FaultInjector:
    """可插拔的故障注入器，用于模拟外部/内部故障"""

    def inject(self, *, agent_id: str | None = None, fault: FaultType | None = None, message: str | None = None) -> FaultSignal:
        ft = fault or FaultType.INJECTED
        return FaultSignal(
            fault_type=ft,
            severity="warn" if ft == FaultType.INJECTED else "error",
            agent_id=agent_id,
            message=message or f"Injected fault: {ft}",
            suggested_action="trigger_recovery",
            extra={"injected": True},
        )
