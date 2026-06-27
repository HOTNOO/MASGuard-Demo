"""Agent基类 - 参考mini-swe-agent的DefaultAgent设计"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import yaml
from jinja2 import Template

from swe_mas.config import CONFIG_DIR
from swe_mas.memory import MemoryRecorder, NodeType
from swe_mas.utils.command_classification import has_file_redirection, looks_like_write_command
from swe_mas.utils.env_executor import LocalExecutor
from swe_mas.utils.git_utils import get_env_snapshot
from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


class ChatModelProtocol(Protocol):
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        ...

    def get_template_vars(self) -> dict[str, Any]:
        ...


@dataclass
class AgentConfig:
    """Agent配置"""
    max_iterations: int = 10
    phase: str = ""
    timeout: int = 60


class BaseAgent:
    """Agent基类
    
    参考mini-swe-agent的设计理念：
    - 使用异常控制流
    - 模板化prompt管理
    - 无状态命令执行
    """
    
    def __init__(
        self,
        model: ChatModelProtocol,
        executor: LocalExecutor,
        agent_type: str,
        config: AgentConfig | None = None,
        recorder: MemoryRecorder | None = None,
        session_id: str | None = None,
    ):
        """初始化Agent
        
        Args:
            model: 语言模型
            executor: 命令执行器
            agent_type: Agent类型（analyzer/locator/planner/implementer/verifier）
            config: Agent配置
        """
        self.model = model
        self.executor = executor
        self.agent_type = agent_type
        self.config = config or AgentConfig()
        self.phase = self.config.phase or agent_type  # 用于节点阶段标记
        self.iteration = 0
        self.recorder = recorder
        self.session_id = session_id or "default"
        self._last_node_id: str | None = None
        self._phase_snapshot_id: int | None = None  # Phase快照ID
        self._recovery_state: dict | None = None  # 恢复状态（用于resume）
        self._stop_checker: Callable[[], bool] | None = None  # 停止检查器（用于prefault等中断）
        
        # 加载prompt模板
        self.prompts = self._load_prompts()
        
        # 对话历史
        self.messages: list[dict[str, str]] = []
        
        # 执行历史
        self.history: list[dict[str, Any]] = []
        if recorder and session_id:
            # 允许外部指定锚点，用于恢复分支
            self._last_node_id = None
        
    def _load_prompts(self) -> dict:
        """加载prompt模板"""
        prompts_file = CONFIG_DIR / "prompts.yaml"
        with open(prompts_file, "r", encoding="utf-8") as f:
            all_prompts = yaml.safe_load(f)
        prompt_block = all_prompts.get(self.agent_type, {})

        # 兼容多语言结构：{agent_type: {zh: {...}, en: {...}}}
        lang = os.getenv("PROMPT_LANG", "zh").lower()
        if isinstance(prompt_block, dict) and ("zh" in prompt_block or "en" in prompt_block):
            return prompt_block.get(lang, prompt_block.get("zh", {}))

        # 兼容旧格式：直接是system/user
        return prompt_block
    
    def render_template(self, template_str: str, **kwargs) -> str:
        """渲染Jinja2模板
        
        Args:
            template_str: 模板字符串
            **kwargs: 模板变量
            
        Returns:
            渲染后的字符串
        """
        # 合并模板变量：模型变量 + 执行器变量 + 自定义变量
        template_vars = (
            self.model.get_template_vars() |
            self.executor.get_template_vars() |
            kwargs
        )
        
        template = Template(template_str)
        return template.render(**template_vars)
    
    def add_message(self, role: str, content: str):
        """添加消息到历史
        
        Args:
            role: 角色（system/user/assistant）
            content: 消息内容
        """
        self.messages.append({"role": role, "content": content})
        logger.debug(f"[{self.agent_type}] {role}: {content[:100]}...")

    def _prompt_cwd(self) -> str:
        """返回应该暴露给模型的可见工作目录。

        本地执行器直接返回本地 cwd；official harness 运行时返回容器内路径
        （如 /testbed），避免把宿主机路径泄露给模型，导致其在容器里反复
        cd 到不存在的目录。
        """
        try:
            template_vars = self.executor.get_template_vars()
        except Exception:
            template_vars = {}
        visible = str(template_vars.get("cwd", "") or "").strip()
        if visible:
            return visible
        return str(getattr(getattr(self.executor, "config", None), "cwd", "") or os.getcwd())

    def _approx_prompt_chars(self) -> int:
        return sum(len(m.get("content") or "") for m in self.messages)

    def _trim_messages_for_model(
        self,
        *,
        max_chars: int = 28000,
        keep_last: int = 12,
        max_single_message_chars: int = 20000,
    ) -> None:
        """尽量避免对话历史过长导致模型侧报错（如 Qwen 30720 输入限制）。

        策略：
        - 保留开头 system（若存在）+ 第一条 user（若存在）
        - 仅保留最后 keep_last 条交互
        - 若仍超限，截断超长消息；再按需继续丢弃最旧 tail
        """
        # 可通过环境变量覆盖（方便实验调参/回归）
        try:
            env_max = os.getenv("SWE_MAS_MAX_PROMPT_CHARS")
            if env_max:
                max_chars = int(env_max)
        except Exception:
            pass
        try:
            env_single = os.getenv("SWE_MAS_MAX_SINGLE_MESSAGE_CHARS")
            if env_single:
                max_single_message_chars = int(env_single)
        except Exception:
            pass
        if max_chars > 0:
            max_single_message_chars = min(max_single_message_chars, max_chars)

        if max_chars <= 0:
            return
        if self._approx_prompt_chars() <= max_chars:
            return

        preserved: list[dict[str, str]] = []
        idx = 0
        if self.messages and self.messages[0].get("role") == "system":
            preserved.append(self.messages[0])
            idx = 1
        if idx < len(self.messages) and self.messages[idx].get("role") == "user":
            preserved.append(self.messages[idx])
            idx += 1

        tail = self.messages[idx:]
        if keep_last > 0 and len(tail) > keep_last:
            tail = tail[-keep_last:]
        self.messages = preserved + tail

        # 1) 若仍超限：优先丢弃最旧 tail，直到满足阈值或只剩 preserved + 2 条
        min_len = len(preserved) + 2
        while self._approx_prompt_chars() > max_chars and len(self.messages) > min_len:
            self.messages.pop(len(preserved))

        # 2) 若仍超限：再截断单条过长消息（包括首次 system+user prompt）。
        if self._approx_prompt_chars() > max_chars:
            for m in self.messages:
                text = m.get("content") or ""
                if len(text) <= max_single_message_chars:
                    continue
                head = text[: int(max_single_message_chars * 0.75)]
                tail_keep = max_single_message_chars - len(head)
                m["content"] = head + "\n...[truncated]...\n" + text[-tail_keep:]

        # 3) 如果多条 preserved 消息相加仍超限，继续按消息长度削减到总预算内。
        while self._approx_prompt_chars() > max_chars:
            longest = max(range(len(self.messages)), key=lambda i: len(self.messages[i].get("content") or ""))
            text = self.messages[longest].get("content") or ""
            if len(text) <= 256:
                break
            overflow = self._approx_prompt_chars() - max_chars
            new_len = max(256, len(text) - overflow - len("\n...[truncated]...\n"))
            head_len = max(128, int(new_len * 0.7))
            tail_len = max(64, new_len - head_len)
            self.messages[longest]["content"] = (
                text[:head_len] + "\n...[truncated]...\n" + text[-tail_len:]
            )

    def set_anchor_node(self, node_id: str | None):
        """外部恢复时设置锚点，使新分支从健康节点继续"""
        self._last_node_id = node_id
    
    def set_recovery_state(self, recovery_state: dict | None):
        """设置恢复状态（用于resume模式）
        
        recovery_state应包含：
        - messages: 故障前的对话历史
        - history: 故障前的执行历史
        - last_iteration: 最后完成的迭代次数
        """
        self._recovery_state = recovery_state

    def set_stop_checker(self, checker: Callable[[], bool] | None):
        """设置停止检查器（用于外部触发的中断/停止）"""
        self._stop_checker = checker

    def _stop_requested(self) -> bool:
        if not self._stop_checker:
            return False
        try:
            return bool(self._stop_checker())
        except Exception:
            return False

    def _start_phase(self, inputs: dict) -> None:
        """记录Phase开始（用于快速恢复）"""
        if self.recorder:
            self._phase_snapshot_id = self.recorder.start_phase(
                session_id=self.session_id,
                phase=self.phase,
                role=self.agent_type,
                inputs=inputs,
                start_node_id=self._last_node_id,
            )

    def _end_phase(self, outputs: dict, success: bool, cwd: str = ".") -> None:
        """记录Phase结束"""
        if self.recorder and self._phase_snapshot_id:
            self.recorder.end_phase(
                snapshot_id=self._phase_snapshot_id,
                outputs=outputs,
                success=success,
                end_node_id=self._last_node_id,
                cwd=cwd,
            )

    def _should_retry_model_error(self, error: Exception) -> bool:
        """仅对明显的瞬时模型/网络错误做有限重试。"""
        text = str(error).lower()
        transient_markers = (
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "temporary failure",
            "temporarily unavailable",
            "service unavailable",
            "rate limit",
            "too many requests",
            "upstream_error",
            "transport request failed",
            "master_data_plane_disabled",
            "master data plane is disabled",
            "failed to validate api key",
            "endpoint returned no choices",
            "empty streaming response",
            "non-json upstream response",
            "heartbeat stream connected",
            "tls connect error",
            "curl: (28)",
            "curl: (35)",
            "429",
            "403",
            "502",
            "503",
            "504",
        )
        return any(marker in text for marker in transient_markers)
    
    def query_model(self) -> str:
        """查询模型获取回复
        
        Returns:
            模型回复内容
        """
        max_attempts = 1
        try:
            max_attempts = max(1, int(os.getenv("SWE_MAS_MODEL_RETRY_ATTEMPTS", "3")))
        except Exception:
            max_attempts = 3
        base_sleep_sec = 2.0
        try:
            base_sleep_sec = max(0.0, float(os.getenv("SWE_MAS_MODEL_RETRY_BASE_SLEEP_SEC", "2")))
        except Exception:
            base_sleep_sec = 2.0

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                self._trim_messages_for_model()
                response = self.model.query(self.messages)
                content = response.get("content", "")
                usage = {}
                try:
                    usage = (response.get("extra") or {}).get("usage") or {}
                except Exception:
                    usage = {}
                token_usage = None
                try:
                    token_usage = usage.get("total_tokens") or usage.get("total") or None
                except Exception:
                    token_usage = None
                self.add_message("assistant", content)
                self._record_node(
                    node_type=NodeType.THOUGHT,
                    content=content,
                    payload={"messages": self.messages[-2:], "step_type": "llm_thought"},
                    token_usage=token_usage,
                    resource_stat={"usage": usage},
                )
                parsed = self._maybe_parse_structured_response(content)
                if parsed:
                    self._record_structured_thought(parsed)
                return content
            except Exception as e:
                last_error = e
                if attempt < max_attempts and self._should_retry_model_error(e):
                    sleep_sec = base_sleep_sec * (2 ** (attempt - 1))
                    logger.warning(
                        f"[{self.agent_type}] 模型请求失败，准备重试 "
                        f"({attempt}/{max_attempts}): {e}"
                    )
                    time.sleep(sleep_sec)
                    continue
                # 将模型错误写入记忆图，便于后续恢复/摘要使用
                err_text = f"MODEL_ERROR: {e}"
                self._record_node(
                    node_type=NodeType.MESSAGE,
                    content=err_text,
                    payload={"step_type": "model_error"},
                    validity="INVALID",
                )
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("unexpected query_model fallthrough")
    
    def _extract_bash_commands(self, text: str) -> list[str]:
        """从文本中提取所有非空 bash 命令块。"""
        pattern = r"```bash[ \t]*\r?\n(.*?)\r?\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

    def _select_bash_command_from_blocks(self, commands: list[str]) -> str | None:
        """从命令块列表中选择实际执行的命令。"""
        return commands[0] if commands else None

    def parse_bash_command(self, text: str) -> str | None:
        """从文本中提取bash命令
        
        Args:
            text: 包含命令的文本
            
        Returns:
            提取的命令，如果没有则返回None
        """
        commands = self._extract_bash_commands(text)
        if commands:
            command = self._select_bash_command_from_blocks(commands)
            if command and len(commands) > 1:
                logger.warning(
                    f"[{self.agent_type}] 模型回复包含多个 bash 代码块，选择执行: {command[:80]}"
                )
            return command
        return None

    def _has_file_redirection(self, command: str) -> bool:
        """检测 shell 文件重定向（> / >>），忽略 2>&1 这类 FD 重定向。"""
        return has_file_redirection(command)

    def _looks_like_write_command(self, command: str) -> bool:
        """粗略判断命令是否可能写入仓库文件（用于只读阶段的护栏）。"""
        return looks_like_write_command(command)

    def _is_interactive_command(self, command: str) -> bool:
        """检测交互式命令（在非TTY环境下容易卡住或无效）。"""
        cmd = (command or "").strip()
        if not cmd:
            return False
        lowered = cmd.lower()
        interactive_prefixes = (
            "less ",
            "more ",
            "man ",
            "vim ",
            "vi ",
            "nano ",
            "emacs ",
            "top",
            "htop",
            "watch ",
        )
        if lowered.startswith(interactive_prefixes):
            return True
        if " tail -f " in f" {lowered} " or lowered.endswith(" tail -f"):
            return True
        return False

    def _ensure_repo_clean(self) -> tuple[bool, str]:
        """检测工作区是否被修改；若被修改则尽力回滚（仅对只读 Agent 调用）。"""
        try:
            status = self.executor.execute("git status --porcelain")
            out = (status.get("output") or "").strip()
            if not out:
                return True, ""
            # 回滚 tracked + 清理 untracked（不包含 ignored）
            self.executor.execute("git reset --hard")
            self.executor.execute("git clean -fd")
            return False, out
        except Exception as e:
            return False, f"git clean failed: {e}"
    
    def execute_command(self, command: str) -> dict[str, Any]:
        """执行bash命令
        
        Args:
            command: 要执行的命令
            
        Returns:
            执行结果 {"output": ..., "returncode": ...}
        """
        logger.info(f"[{self.agent_type}] 执行命令: {command[:80]}...")
        if self._is_interactive_command(command):
            result = {
                "output": "交互式命令不被允许，请使用非交互式命令（如 cat/grep/sed）。",
                "returncode": 126,
            }
        else:
            result = self.executor.execute(command, timeout=self.config.timeout)

        self._record_node(
            node_type=NodeType.ACTION,
            content=command,
            payload={
                "command": command,
                "step_type": "run_tests" if "pytest" in command or "nosetests" in command else "shell_command",
            },
        )

        env_hash = get_env_snapshot(self.executor.config.cwd)
        test_status = None
        if "pytest" in command or "nosetests" in command:
            test_status = "fail" if result.get("returncode", 0) != 0 else "pass"
        files_touched = self._maybe_extract_files_from_output(result.get("output", ""))
        self._record_node(
            node_type=NodeType.OBSERVATION,
            content=result.get("output", "")[:2000],
            payload={
                "returncode": result.get("returncode", 0),
                "command": command,
                "step_type": "test_result" if test_status else "command_output",
                "test_status": test_status,
                "files_touched": files_touched,
            },
            env_snapshot_hash=env_hash,
        )

        # 记录到历史
        self.history.append({
            "command": command,
            "output": result["output"],
            "returncode": result["returncode"],
        })

        return result
    
    def run(self, **kwargs) -> dict[str, Any]:
        """运行Agent（子类需要实现）
        
        Args:
            **kwargs: 输入参数
            
        Returns:
            执行结果字典
        """
        raise NotImplementedError("子类需要实现run方法")

    def _maybe_parse_structured_response(self, text: str) -> dict | None:
        """尝试从模型回复中解析结构化JSON，失败返回None"""
        if not text:
            return None
        candidates: list[str] = []
        block = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if block:
            candidates.append(block.group(1).strip())
        if not candidates:
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            if brace:
                candidates.append(brace.group(0).strip())
        for raw in candidates:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return None

    def _record_structured_thought(self, data: dict):
        """根据结构化回复拆解多条节点，便于感知与摘要"""
        thought_text = data.get("thought") or data.get("summary") or data.get("intent")
        if isinstance(thought_text, str) and thought_text.strip():
            self._record_node(
                node_type=NodeType.THOUGHT,
                content=thought_text,
                payload={
                    "structured": True,
                    "step_type": "llm_structured",
                    "summary_hint": data.get("summary_hint"),
                },
            )
        plan_block = data.get("plan") or data.get("steps")
        plan_text = ""
        if isinstance(plan_block, list):
            plan_text = "\n".join(f"- {p}" for p in plan_block)
        elif isinstance(plan_block, str):
            plan_text = plan_block
        if plan_text:
            self._record_node(
                node_type=NodeType.PLAN,
                content=plan_text,
                payload={"structured": True, "step_type": "llm_structured"},
            )
        remaining = {k: v for k, v in data.items() if k not in {"thought", "summary", "intent", "plan", "steps", "summary_hint"}}
        if remaining:
            try:
                extra_text = json.dumps(remaining, ensure_ascii=False)
            except Exception:
                extra_text = str(remaining)
            extra_payload = {"structured": True, "step_type": "llm_structured"}
            if isinstance(remaining.get("files_intended"), list):
                extra_payload["files_intended"] = remaining.get("files_intended")
            if isinstance(remaining.get("test_plan"), list):
                extra_payload["test_plan"] = remaining.get("test_plan")
            if isinstance(remaining.get("patch_strategy"), str):
                extra_payload["patch_strategy"] = remaining.get("patch_strategy")
            self._record_node(
                node_type=NodeType.MESSAGE,
                content=extra_text,
                payload=extra_payload,
            )

    def _maybe_extract_files_from_output(self, output: str) -> list[str]:
        """从命令输出中粗略提取文件路径，用于后续摘要/感知"""
        if not output:
            return []
        matches = re.findall(r"[A-Za-z0-9_./\\-]+\\.py", output)
        seen = set()
        files: list[str] = []
        for m in matches:
            norm = m.strip()
            if norm and norm not in seen:
                seen.add(norm)
                files.append(norm)
        return files

    def _record_node(
        self,
        node_type: NodeType,
        content: str,
        payload: dict[str, Any] | None = None,
        env_snapshot_hash: str | None = None,
        resource_stat: dict[str, Any] | None = None,
        validity: str = "VALID",
        token_usage: float | int | None = None,
    ):
        if not self.recorder:
            return
        payload = payload or {}
        node = {
            "session_id": self.session_id,
            "agent_id": self.agent_type,
            "role": self.agent_type,
            "node_type": node_type.value if isinstance(node_type, NodeType) else str(node_type),
            "content": content,
            "payload": payload,
            "timestamp": time.time(),
            "temporal_prev": self._last_node_id,
            "env_snapshot_hash": env_snapshot_hash,
            "resource_stat": resource_stat or {},
            "phase": self.phase,
            "step_type": payload.get("step_type"),
            "test_status": payload.get("test_status"),
            "files_touched": payload.get("files_touched"),
            "summary_hint": payload.get("summary_hint"),
            "iteration": self.iteration,
            "validity": validity,
            "token_usage": token_usage,
        }
        if self._last_node_id:
            node["causal_parents"] = [self._last_node_id]
        node_id = self.recorder.append(node)
        self._last_node_id = node_id
