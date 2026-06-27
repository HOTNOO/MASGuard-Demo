"""后台感知线程：定期扫描感知层，填充故障信号供编排器使用"""

from __future__ import annotations

import threading
import time
from typing import Optional

from swe_mas.perception import PerceptionManager
from swe_mas.perception.models import FaultSignal
from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


class PerceptionWorker:
    def __init__(self, perception: PerceptionManager, session_id: str, recorder_event, interval: float = 2.0):
        self.perception = perception
        self.session_id = session_id
        self.interval = interval
        self._stop = threading.Event()
        self.fault_signal: Optional[FaultSignal] = None
        self.thread: Optional[threading.Thread] = None
        self.recorder_event = recorder_event

    def start(self):
        def _loop():
            while not self._stop.is_set():
                try:
                    # 等待事件或超时
                    self.recorder_event.wait(self.interval)
                    self.recorder_event.clear()
                    signals = self.perception.run(session_id=self.session_id)
                    if signals:
                        self.fault_signal = signals[0]
                        logger.warning(f"[PerceptionWorker] Fault detected: {self.fault_signal}")
                        break
                except Exception as e:
                    logger.warning(f"[PerceptionWorker] perceive error: {e}")

        self.thread = threading.Thread(target=_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def consume_fault(self) -> Optional[FaultSignal]:
        sig = self.fault_signal
        self.fault_signal = None
        return sig

    def peek_fault(self) -> Optional[FaultSignal]:
        return self.fault_signal
