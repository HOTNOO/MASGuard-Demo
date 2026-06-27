"""验证测试Agent"""

import re
from typing import Any

from swe_mas.agents.base import AgentConfig, BaseAgent
from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


class VerifierAgent(BaseAgent):
    """验证测试专家
    
    职责：验证代码修改是否解决问题，运行测试检查
    """
    
    def __init__(self, *args, **kwargs):
        config = kwargs.pop("config", AgentConfig())
        if not config.phase:
            config.phase = "verify"
        super().__init__(*args, agent_type="verifier", config=config, **kwargs)

    def _validation_required_message(self) -> str:
        return (
            "你还没有完成有效验证：你现在不是普通聊天助手；你输出的单个 "
            "```bash``` 代码块会由框架在当前仓库/容器环境中执行。"
            "不要回答“无法运行 pytest”或“当前对话环境不支持执行”。"
            "请先给出一个非交互式测试命令（pytest/python -m pytest/unittest 等），"
            "等待命令输出后，再输出最终评估 YAML；status 必须是明确单值"
            "（通过/需要改进/失败 或 PASS/NEEDS_IMPROVEMENT/FAIL），不要输出模板占位符。"
        )
    
    def run(self, problem_statement: str, implementation: str, cwd: str = "") -> dict[str, Any]:
        """验证修复效果
        
        Args:
            problem_statement: 原始问题描述
            implementation: 实现的修改（git diff）
            cwd: 工作目录
            
        Returns:
            {"verification": "验证结果", "success": bool, "status": "通过/需要改进/失败"}
        """
        logger.info(f"[Verifier] 开始验证修复...")
        
        # 设置工作目录
        if cwd:
            self.executor.config.cwd = cwd
        
        # 记录Phase开始
        self._start_phase({
            "problem_statement": problem_statement,
            "implementation": implementation,
            "cwd": cwd or self.executor.config.cwd
        })
        
        # 构建prompt
        system_prompt = self.prompts.get("system", "")
        user_prompt = self.render_template(
            self.prompts.get("user", ""),
            problem_statement=problem_statement,
            implementation=implementation
        )
        
        # 初始化对话
        self.messages = []
        self.history = []
        self.add_message("system", system_prompt)
        self.add_message("user", user_prompt)
        
        # 迭代执行验证
        ran_tests = False
        for iteration in range(self.config.max_iterations):
            try:
                # 查询模型
                response = self.query_model()
                
                # If the model emits a validation command and a premature YAML verdict in
                # the same response, run the command first. Otherwise a malformed final
                # answer can suppress the bounded validation probe entirely.
                command = self.parse_bash_command(response)
                yaml_match = re.search(r"```yaml\s*\n(.*?)\n```", response, re.DOTALL)
                if command and not ran_tests:
                    result = self.execute_command(command)
                    lowered = command.lower()
                    if "pytest" in lowered or "nosetests" in lowered or "unittest" in lowered:
                        ran_tests = True

                    observation = f"命令输出:\n{result['output'][:2000]}"
                    if result["returncode"] != 0:
                        observation += f"\n返回码: {result['returncode']}"
                    self.add_message("user", observation)
                    continue

                # 检查是否给出了最终评估
                if yaml_match:
                    verification = yaml_match.group(1).strip()
                    # 解析状态
                    status = "未知"
                    if "status:" in verification:
                        status_match = re.search(r"status:\s*[\"']?(.*?)[\"']?\s*\n", verification)
                        if status_match:
                            status = status_match.group(1).strip()

                    status_norm = status.strip().lower()

                    # 1) 防止模型直接返回模板占位符（未跑测试）
                    placeholder_statuses = {
                        "通过/需要改进/失败",
                        "pass | needs_improvement | fail",
                        "pass|needs_improvement|fail",
                        "pass/needs_improvement/fail",
                    }
                    is_placeholder = (
                        status in placeholder_statuses
                        or status_norm in placeholder_statuses
                        or ("通过" in status and "需要改进" in status and "失败" in status)
                        or ("pass" in status_norm and "needs" in status_norm and "fail" in status_norm)
                    )
                    if is_placeholder or not ran_tests:
                        self.add_message("user", self._validation_required_message())
                        continue

                    logger.info(f"[Verifier] 验证完成")
                    
                    # 约定：只有明确"通过/pass"才算验证成功；"需要改进/失败"一律视为未通过
                    failed_markers = ("失败",)
                    needs_markers = ("需要改进",)
                    fail_words = ("fail", "failed", "error")
                    needs_words = ("needs", "improve", "improvement")
                    pass_markers = ("通过",)
                    pass_words = ("pass", "passed", "success", "ok")

                    if any(m in status for m in failed_markers + needs_markers) or any(w in status_norm for w in fail_words + needs_words):
                        passed = False
                    elif any(m in status for m in pass_markers) or any(w in status_norm for w in pass_words):
                        passed = True
                    else:
                        passed = False
                    
                    result = {
                        "verification": verification,
                        "success": passed,
                        "status": status,
                        "commands": self.history.copy(),
                        "messages": self.messages.copy(),
                        "stop_reason": "submitted_verdict",
                        "ran_tests": ran_tests,
                        "test_command_count": sum(
                            1
                            for item in self.history
                            if any(
                                token in str(item.get("command", "")).lower()
                                for token in ("pytest", "nosetests", "unittest")
                            )
                        ),
                    }
                    self._end_phase(result, success=passed, cwd=self.executor.config.cwd)
                    return result
                
                # 提取并执行命令
                if command:
                    result = self.execute_command(command)
                    lowered = command.lower()
                    if "pytest" in lowered or "nosetests" in lowered or "unittest" in lowered:
                        ran_tests = True
                    
                    # 添加观察结果
                    observation = f"命令输出:\n{result['output'][:2000]}"
                    if result["returncode"] != 0:
                        observation += f"\n返回码: {result['returncode']}"
                    self.add_message("user", observation)
                else:
                    # 没有命令也没有评估结果
                    self.add_message("user", self._validation_required_message())
                    
            except Exception as e:
                logger.error(f"[Verifier] 迭代{iteration}出错: {str(e)}")
                is_infra_error = self._should_retry_model_error(e) or "MODEL_ERROR" in str(e)
                result = {
                    "verification": (
                        f"验证中断（基础设施异常）: {str(e)}"
                        if is_infra_error
                        else f"验证失败: {str(e)}"
                    ),
                    "success": False,
                    "status": "基础设施异常" if is_infra_error else "失败",
                    "error": f"MODEL_ERROR: {str(e)}" if is_infra_error else f"验证失败: {str(e)}",
                    "infrastructure_error": is_infra_error,
                    "commands": self.history.copy(),
                    "messages": self.messages.copy(),
                    "stop_reason": "runtime_error",
                    "ran_tests": ran_tests,
                }
                self._end_phase(result, success=False, cwd=self.executor.config.cwd)
                return result
        
        # 达到最大迭代次数
        logger.warning(f"[Verifier] 达到最大迭代次数")
        result = {
            "verification": "验证未完成（达到最大迭代次数）",
            "success": False,
            "status": "未知",
            "commands": self.history.copy(),
            "messages": self.messages.copy(),
            "stop_reason": "budget_exhausted_with_partial_evidence" if ran_tests else "true_no_progress",
            "ran_tests": ran_tests,
            "test_command_count": sum(
                1
                for item in self.history
                if any(
                    token in str(item.get("command", "")).lower()
                    for token in ("pytest", "nosetests", "unittest")
                )
            ),
        }
        self._end_phase(result, success=False, cwd=self.executor.config.cwd)
        return result
