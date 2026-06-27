"""日志工具模块"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logger(
    name: str = "swe_mas",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径，None则不写文件
        
    Returns:
        配置好的Logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的handler
    logger.handlers.clear()
    
    # 控制台handler（使用Rich美化输出）
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的日志
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "swe_mas") -> logging.Logger:
    """获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        Logger对象
    """
    return logging.getLogger(name)


# 默认日志记录器
default_logger = setup_logger()

