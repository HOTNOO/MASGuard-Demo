"""动态复现 Agent：尝试最小代价复现问题并收集错误日志"""

import re
from typing import Any

from swe_mas.agents.base import AgentConfig, BaseAgent
from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


class ReproducerAgent(BaseAgent):
    """动态复现场景专家

    职责：在最小代价下复现问题，收集失败日志/堆栈，为后续规划提供证据。
    """

    def __init__(self, *args, **kwargs):
        config = kwargs.pop("config", AgentConfig())
        if not config.phase:
            config.phase = "locate"
        super().__init__(*args, agent_type="reproducer", config=config, **kwargs)

    def run(self, problem_analysis: str, cwd: str = "") -> dict[str, Any]:
        """尝试复现问题

        Args:
            problem_analysis: 经过分析的结构化描述
            cwd: 工作目录

        Returns:
            {"reproduction": str, "success": bool, "commands": [...], "messages": [...]} 
        """
        logger.info("[Reproducer] 开始动态复现...")

        if cwd:
            self.executor.config.cwd = cwd

        # 记录Phase开始
        self._start_phase({"problem_analysis": problem_analysis, "cwd": cwd or self.executor.config.cwd})
        prompt_cwd = self._prompt_cwd()

        system_prompt = self.prompts.get("system", "")
        user_prompt = self.render_template(
            self.prompts.get("user", ""),
            problem_analysis=problem_analysis,
            cwd=prompt_cwd,
        )

        self.messages = []
        self.history = []
        self.add_message("system", system_prompt)
        self.add_message("user", user_prompt)

        for iteration in range(self.config.max_iterations):
            try:
                response = self.query_model()

                # 完成信号
                if "```finish" in response:
                    match = re.search(r"```finish\s*\n(.*?)\n```", response, re.DOTALL)
                    if match:
                        reproduction = match.group(1).strip()
                        logger.info("[Reproducer] 复现流程完成")
                        result = {
                            "reproduction": reproduction,
                            "success": True,
                            "commands": self.history.copy(),
                            "messages": self.messages.copy(),
                        }
                        self._end_phase(result, success=True, cwd=self.executor.config.cwd)
                        return result

                # 执行命令
                command = self.parse_bash_command(response)
                if command:
                    if self._looks_like_write_command(command):
                        self.add_message(
                            "user",
                            "错误：动态复现阶段禁止修改仓库文件（例如 > / >> 重定向、sed -i、tee、rm/mv/cp 等）。"
                            "请改用只读命令（pytest/python/ls/rg/cat 等）来复现并收集日志。",
                        )
                        continue
                    result = self.execute_command(command)
                    clean, dirty = self._ensure_repo_clean()
                    if not clean:
                        self.add_message(
                            "user",
                            "警告：检测到上一个命令修改了工作区（只读阶段不允许），已自动回滚清理。\n"
                            f"git status --porcelain:\n{dirty[:800]}",
                        )
                    observation = self.render_template(
                        "{{output}}",
                        output=result.get("output", "")[:2000],
                        returncode=result.get("returncode", 0),
                    )
                    if result.get("returncode", 0) != 0:
                        observation += f"\nreturncode: {result.get('returncode', 0)}"
                    self.add_message("user", f"Command output:\n{observation}")
                else:
                    self.add_message("user", "Please send one bash command wrapped in ```bash``` or provide final summary in ```finish``` block.")

            except Exception as e:
                logger.error(f"[Reproducer] 迭代{iteration}出错: {str(e)}")
                result = {
                    "reproduction": f"复现失败: {str(e)}",
                    "success": False,
                    "commands": self.history.copy(),
                    "messages": self.messages.copy(),
                }
                self._end_phase(result, success=False, cwd=self.executor.config.cwd)
                return result

        logger.warning("[Reproducer] 达到最大迭代次数")
        result = {
            "reproduction": "复现未完成（达到最大迭代次数）",
            "success": False,
            "commands": self.history.copy(),
            "messages": self.messages.copy(),
        }
        self._end_phase(result, success=False, cwd=self.executor.config.cwd)
        return result
