"""回滚接口与实现：Git 回滚为主，无法回滚时使用 No-op"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol

from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


class Rollbacker(Protocol):
    def rollback(self, snapshot: str | None, workdir: str) -> bool:
        ...


class GitRollbacker:
    def rollback(self, snapshot: str | None, workdir: str) -> bool:
        if not snapshot:
            return False
        try:
            # env_snapshot_hash 可能包含附加信息（如 dirty/diff_hash），仅使用首段 git 提交哈希
            if ":" in snapshot:
                snapshot = snapshot.split(":")[0]
            result = subprocess.run(
                ["git", "reset", "--hard", snapshot],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(f"回滚到快照 {snapshot} 成功")
                return True
            logger.warning(f"回滚失败: {result.stderr}")
            return False
        except Exception as e:
            logger.warning(f"回滚异常: {e}")
            return False


class NoopRollbacker:
    def rollback(self, snapshot: str | None, workdir: str) -> bool:       
        logger.info("无可用回滚策略，跳过回滚")
        return False
