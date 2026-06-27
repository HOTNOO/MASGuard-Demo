"""代码定位Agent"""

import json
import os
from pathlib import Path
import re
from typing import Any

from swe_mas.agents.base import AgentConfig, BaseAgent
from swe_mas.utils.command_classification import looks_like_readonly_probe_command
from swe_mas.utils.logger import get_logger
from swe_mas.utils.path_filters import normalize_repo_path, repo_path_variants
from swe_mas.utils.probe_analysis import (
    classify_probe_delta,
    no_progress_probe_streak,
    summarize_probe_paths,
    summarize_regions_or_symbols,
)

logger = get_logger(__name__)


class LocatorAgent(BaseAgent):
    """代码定位专家
    
    职责：根据问题分析定位相关代码文件和函数
    """

    MAX_READONLY_STREAK = 6
    ALLOW_BEST_EFFORT_FINISH = False
    
    def __init__(self, *args, **kwargs):
        config = kwargs.pop("config", AgentConfig())
        if not config.phase:
            config.phase = "locate"
        super().__init__(*args, agent_type="locator", config=config, **kwargs)
    
    def run(self, problem_analysis: str, cwd: str = "") -> dict[str, Any]:
        """定位相关代码
        
        Args:
            problem_analysis: 问题分析结果
            cwd: 工作目录
            
        Returns:
            {"located_files": "定位结果", "success": bool, "commands": [...]}
        """
        logger.info(f"[Locator] 开始定位代码...")
        
        # 设置工作目录
        if cwd:
            self.executor.config.cwd = cwd
        
        # 记录Phase开始
        self._start_phase({"problem_analysis": problem_analysis, "cwd": cwd or self.executor.config.cwd})
        prompt_cwd = self._prompt_cwd()
        
        # 构建prompt
        system_prompt = self.prompts.get("system", "")
        user_prompt = self.render_template(
            self.prompts.get("user", ""),
            problem_analysis=problem_analysis,
            cwd=prompt_cwd,
        )
        
        # 初始化对话
        self.messages = []
        self.history = []
        self.add_message("system", system_prompt)
        self.add_message("user", user_prompt)
        
        # 迭代执行
        for iteration in range(self.config.max_iterations):
            try:
                # 查询模型
                response = self.query_model()
                
                # 检查是否完成。优先接受规范的 ```finish```，同时容忍模型
                # 已给出结构化定位结果但漏写 finish fence 的情况。
                finish_payload = self._extract_finish_payload(response)
                if finish_payload is not None:
                    located_files = self._normalize_located_files_output(finish_payload)
                    logger.info(f"[Locator] 定位完成")
                    result = {
                        "located_files": located_files,
                        "success": True,
                        "commands": self.history.copy(),
                        "messages": self.messages.copy(),
                        **self._result_metadata(
                            located_files=located_files,
                            stop_reason="submitted_finish",
                        ),
                    }
                    self._end_phase(result, success=True, cwd=self.executor.config.cwd)
                    return result
                
                # 提取并执行命令
                command = self.parse_bash_command(response)
                if command:
                    force_finish = self._force_finish_analysis(command)
                    if force_finish is not None:
                        best_effort = self._best_effort_located_files()
                        if best_effort:
                            metadata = self._result_metadata(
                                located_files=best_effort,
                                stop_reason="true_no_progress",
                                analysis=force_finish,
                            )
                            if self._allow_best_effort_finish():
                                logger.info("[Locator] 使用best-effort定位结果收束")
                                result = {
                                    "located_files": best_effort,
                                    "success": True,
                                    "best_effort_only": True,
                                    "commands": self.history.copy(),
                                    "messages": self.messages.copy(),
                                    **metadata,
                                }
                                self._end_phase(result, success=True, cwd=self.executor.config.cwd)
                                return result
                            logger.info("[Locator] 返回best-effort定位候选，但不标记为成功")
                            result = {
                                "located_files": best_effort,
                                "success": False,
                                "best_effort_only": True,
                                "failure_mode": "locator_best_effort_only",
                                "commands": self.history.copy(),
                                "messages": self.messages.copy(),
                                **metadata,
                            }
                            self._end_phase(result, success=False, cwd=self.executor.config.cwd)
                            return result
                        self.add_message(
                            "user",
                            "提醒：你已经连续多次执行只读探索命令，定位证据已经足够。"
                            "下一步不要继续浏览目录或重复查看同一文件，而是直接输出```finish```："
                            "给出最可能相关的 1-3 个文件、可疑函数/类，以及你的置信度。",
                        )
                        continue
                    if self._looks_like_write_command(command):
                        self.add_message(
                            "user",
                            "错误：定位阶段禁止修改仓库文件（例如 > / >> 重定向、sed -i、tee、rm/mv/cp 等）。"
                            "请改用只读命令（find/rg/grep/ls/cat/head 等）继续定位。",
                        )
                        continue
                    result = self.execute_command(command)
                    self._record_probe_history(command)
                    clean, dirty = self._ensure_repo_clean()
                    if not clean:
                        self.add_message(
                            "user",
                            "警告：检测到上一个命令修改了工作区（定位阶段不允许），已自动回滚清理。\n"
                            f"git status --porcelain:\n{dirty[:800]}",
                        )
                    
                    # 添加观察结果
                    observation = self.render_template(
                        "{{output}}",
                        output=result["output"][:2000],  # 限制长度
                        returncode=result["returncode"]
                    )
                    self.add_message("user", f"命令输出:\n{observation}")
                else:
                    self.add_message("user", "错误：未找到bash命令，请用```bash```包裹命令")
                    
            except Exception as e:
                logger.error(f"[Locator] 迭代{iteration}出错: {str(e)}")
                result = {
                    "located_files": f"定位失败: {str(e)}",
                    "success": False,
                    "commands": self.history.copy(),
                    "messages": self.messages.copy(),
                    **self._result_metadata(
                        located_files="",
                        stop_reason="runtime_error",
                    ),
                }
                self._end_phase(result, success=False, cwd=self.executor.config.cwd)
                return result
        
        # 达到最大迭代次数
        logger.warning(f"[Locator] 达到最大迭代次数")
        best_effort = self._best_effort_located_files()
        if best_effort:
            metadata = self._result_metadata(
                located_files=best_effort,
                stop_reason=self._terminal_stop_reason(),
            )
            if self._allow_best_effort_finish():
                logger.info("[Locator] 达到最大迭代后返回best-effort定位结果")
                result = {
                    "located_files": best_effort,
                    "success": True,
                    "best_effort_only": True,
                    "commands": self.history.copy(),
                    "messages": self.messages.copy(),
                    **metadata,
                }
                self._end_phase(result, success=True, cwd=self.executor.config.cwd)
                return result
            logger.info("[Locator] 达到最大迭代后仅返回best-effort候选，不标记为成功")
            result = {
                "located_files": best_effort,
                "success": False,
                "best_effort_only": True,
                "failure_mode": "locator_best_effort_only",
                "commands": self.history.copy(),
                "messages": self.messages.copy(),
                **metadata,
            }
            self._end_phase(result, success=False, cwd=self.executor.config.cwd)
            return result
        result = {
            "located_files": "定位未完成（达到最大迭代次数）",
            "success": False,
            "commands": self.history.copy(),
            "messages": self.messages.copy(),
            **self._result_metadata(
                located_files="",
                stop_reason=self._terminal_stop_reason(),
            ),
        }
        self._end_phase(result, success=False, cwd=self.executor.config.cwd)
        return result

    def _force_finish_analysis(self, command: str) -> dict[str, Any] | None:
        if not self._is_read_only_probe_command(command):
            return None
        analysis = classify_probe_delta(command, self.history)
        if analysis["evidence_incremental"]:
            return None
        if no_progress_probe_streak(self.history) >= self.MAX_READONLY_STREAK - 1:
            return analysis
        return None

    def _should_force_finish(self, command: str) -> bool:
        if self._force_finish_analysis(command) is not None:
            return True
        legacy_readonly = [
            item
            for item in getattr(self, "history", [])
            if isinstance(item, dict)
            and "read_only_probe" not in item
            and self._is_read_only_probe_command(str(item.get("command", "")))
        ]
        return (
            self._is_read_only_probe_command(command)
            and len(legacy_readonly) >= self.MAX_READONLY_STREAK
        )

    def _is_read_only_probe_command(self, command: str) -> bool:
        return looks_like_readonly_probe_command(command)

    def _extract_finish_payload(self, response: str) -> str | None:
        text = str(response or "").strip()
        if not text:
            return None
        match = re.search(r"```finish\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        if self.parse_bash_command(text):
            return None
        if self._looks_like_unfenced_final_location(text):
            return text
        return None

    def _looks_like_unfenced_final_location(self, text: str) -> bool:
        """True when a locator response is clearly a final answer, not a probe."""
        if re.fullmatch(r"finish", text.strip(), flags=re.IGNORECASE):
            return False
        json_match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except Exception:
                parsed = None
            files = parsed.get("files", []) if isinstance(parsed, dict) else []
            if isinstance(files, list) and any(isinstance(item, dict) and str(item.get("path", "")).strip() for item in files):
                return True

        path_count = len(set(re.findall(r"[A-Za-z0-9_./-]+\.py", text)))
        if path_count == 0:
            return False
        final_markers = (
            "相关文件列表",
            "简要说明",
            "confidence",
            "entry_points",
            "已定位",
            "定位完成",
            "most likely",
            "relevant files",
        )
        return any(marker.lower() in text.lower() for marker in final_markers)

    def _best_effort_located_files(self) -> str:
        counts: dict[str, int] = {}
        for item in self.history:
            for source in (str(item.get("command", "")), str(item.get("output", ""))):
                for path in re.findall(r"[A-Za-z0-9_./-]+\.py", source):
                    normalized = self._normalize_candidate_path(path)
                    counts[normalized] = counts.get(normalized, 0) + 1
        if not counts:
            return ""
        ranked_paths = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        files = [{"path": path, "reason": f"定位阶段重复出现 {count} 次"} for path, count in ranked_paths]
        human_lines = [
            "best-effort 定位结果：以下文件在定位阶段被反复命中，优先作为修改候选。",
        ]
        for entry in files:
            human_lines.append(f"- {entry['path']}: {entry['reason']}")
        payload = {
            "files": files,
            "entry_points": [],
            "confidence": "medium",
        }
        return "\n".join(
            [
                *human_lines,
                "",
                "```json",
                json.dumps(payload, ensure_ascii=False, indent=2),
                "```",
            ]
        )

    def _workspace_root(self) -> Path:
        executor = getattr(self, "executor", None)
        cwd = str(getattr(getattr(executor, "config", None), "cwd", "") or os.getcwd())
        return Path(cwd)

    def _normalize_candidate_path(self, path: str) -> str:
        normalized = normalize_repo_path(str(path or ""))
        if not normalized:
            return ""
        workspace = self._workspace_root()
        variants = repo_path_variants(normalized)
        for prefix in ("src/", "artifacts/", "repo/", "source/"):
            if normalized.startswith(prefix):
                candidate = normalized[len(prefix):]
                if candidate and candidate not in variants:
                    variants.append(candidate)
        for candidate in variants:
            if (workspace / candidate).exists():
                return candidate
        if "/" not in normalized:
            matches = sorted(
                {
                    str(path.relative_to(workspace)).replace("\\", "/")
                    for path in workspace.rglob(normalized)
                    if path.is_file()
                }
            )
            if len(matches) == 1:
                return matches[0]
        return normalized

    def _normalize_located_files_output(self, located_files: str) -> str:
        text = str(located_files or "").strip()
        if not text:
            return ""
        replacements: dict[str, str] = {}
        match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                files = parsed.get("files", [])
                if isinstance(files, list):
                    for item in files:
                        if not isinstance(item, dict):
                            continue
                        original = str(item.get("path", "")).strip().lstrip("./")
                        normalized = self._normalize_candidate_path(original)
                        if original and normalized and original != normalized:
                            replacements[original] = normalized
                        item["path"] = normalized or original
                entry_points = parsed.get("entry_points", [])
                if isinstance(entry_points, list):
                    normalized_entry_points = []
                    for item in entry_points:
                        original = str(item).strip()
                        normalized = self._normalize_candidate_path(original)
                        if original and normalized and original != normalized:
                            replacements[original] = normalized
                        normalized_entry_points.append(normalized or original)
                    parsed["entry_points"] = normalized_entry_points
                text = text[: match.start(1)] + json.dumps(parsed, ensure_ascii=False, indent=2) + text[match.end(1):]
        else:
            for raw_path in re.findall(r"[A-Za-z0-9_./-]+\.py", text):
                normalized = self._normalize_candidate_path(raw_path)
                if raw_path != normalized:
                    replacements[raw_path] = normalized

        for original, normalized in replacements.items():
            text = re.sub(rf"(?<![A-Za-z0-9_./-]){re.escape(original)}(?![A-Za-z0-9_./-])", normalized, text)
        return text

    def _allow_best_effort_finish(self) -> bool:
        env_value = os.getenv("SWE_MAS_LOCATOR_ALLOW_BEST_EFFORT", "").strip().lower()
        if env_value in {"1", "true", "yes", "on"}:
            return True
        if env_value in {"0", "false", "no", "off"}:
            return False
        return bool(self.ALLOW_BEST_EFFORT_FINISH)

    def _record_probe_history(self, command: str) -> None:
        if not self.history:
            return
        item = self.history[-1]
        read_only_probe = self._is_read_only_probe_command(command)
        item["read_only_probe"] = read_only_probe
        if not read_only_probe:
            item["evidence_delta_kind"] = ""
            item["evidence_incremental"] = False
            item["probe_signature"] = {}
            item["inspected_regions_or_symbols"] = []
            return
        item.update(classify_probe_delta(command, self.history[:-1]))

    def _result_metadata(
        self,
        *,
        located_files: str,
        stop_reason: str,
        analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(analysis or {})
        return {
            "stop_reason": stop_reason,
            "evidence_delta_kind": str(metadata.get("evidence_delta_kind", "") or ""),
            "probe_signature": dict(metadata.get("probe_signature", {})),
            "inspected_regions_or_symbols": summarize_regions_or_symbols(self.history),
            "selected_target_candidates": self._selected_target_candidates(located_files),
        }

    def _selected_target_candidates(self, located_files: str) -> list[str]:
        candidates: list[str] = []
        for item in self._extract_locator_payload(located_files):
            path = self._normalize_candidate_path(str(item.get("path", "")))
            if path and path not in candidates:
                candidates.append(path)
        if candidates:
            return candidates
        return summarize_probe_paths(self.history)

    def _extract_locator_payload(self, located_files: str) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        text = str(located_files or "")
        match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except Exception:
                parsed = {}
            files = parsed.get("files", []) if isinstance(parsed, dict) else []
            if isinstance(files, list):
                payload.extend(dict(item) for item in files if isinstance(item, dict))
        for path in re.findall(r"[A-Za-z0-9_./-]+\.py", text):
            normalized = path.lstrip("./")
            if not any(str(item.get("path", "")).strip().lstrip("./") == normalized for item in payload):
                payload.append({"path": normalized})
        return payload

    def _terminal_stop_reason(self) -> str:
        if no_progress_probe_streak(self.history) >= self.MAX_READONLY_STREAK:
            return "true_no_progress"
        incremental = any(bool(item.get("evidence_incremental")) for item in self.history)
        return "budget_exhausted_with_partial_evidence" if incremental else "true_no_progress"
