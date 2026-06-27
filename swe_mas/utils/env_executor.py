"""环境执行器 - 参考mini-swe-agent的LocalEnvironment实现"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LocalExecutorConfig:
    """本地执行器配置"""
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 60
    encoding: str = "utf-8"


class LocalExecutor:
    """本地命令执行器
    
    参考mini-swe-agent的Environment Protocol实现:
    - config: 配置对象
    - execute(command) -> dict
    - get_template_vars() -> dict
    """
    
    def __init__(self, config: LocalExecutorConfig | None = None, **kwargs):
        """初始化执行器
        
        Args:
            config: 配置对象
            **kwargs: 配置参数
        """
        if config is None:
            config = LocalExecutorConfig(**kwargs)
        self.config = config
        
    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> dict[str, Any]:
        """执行shell命令
        
        Args:
            command: 要执行的命令
            cwd: 工作目录，默认使用config中的cwd
            timeout: 超时时间（秒），默认使用config中的timeout
            
        Returns:
            {"output": "命令输出", "returncode": 返回码}
        """
        work_dir = cwd or self.config.cwd or os.getcwd()
        timeout_val = timeout or self.config.timeout
        
        logger.debug(f"执行命令: {command[:100]}...")
        logger.debug(f"工作目录: {work_dir}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                cwd=work_dir,
                env=os.environ | self.config.env,
                timeout=timeout_val,
                encoding=self.config.encoding,
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            
            output = {
                "output": result.stdout or "",
                "returncode": result.returncode,
            }
            
            if result.returncode != 0:
                logger.debug(f"命令执行失败 (返回码={result.returncode})")
            
            return output
            
        except subprocess.TimeoutExpired as e:
            logger.warning(f"命令执行超时 ({timeout_val}秒)")
            timeout_output = ""
            if e.output:
                if isinstance(e.output, bytes):
                    timeout_output = e.output.decode(self.config.encoding, errors="replace")
                else:
                    timeout_output = str(e.output)
            return {
                "output": f"命令执行超时 ({timeout_val}秒)\n" + timeout_output,
                "returncode": -1,
            }
        except Exception as e:
            logger.error(f"命令执行异常: {str(e)}")
            return {
                "output": f"命令执行异常: {str(e)}",
                "returncode": -2,
            }
    
    def get_template_vars(self) -> dict[str, Any]:
        """获取模板变量
        
        Returns:
            包含环境信息的字典
        """
        import platform
        return {
            "cwd": self.config.cwd or os.getcwd(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        }
