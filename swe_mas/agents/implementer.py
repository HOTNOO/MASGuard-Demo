"""代码实现Agent"""

import re
import shlex
from typing import Any

from bcmr_swe.recovery.patch_contract import parse_patch_contract_prompt
from bcmr_swe.recovery.patch_intent import parse_patch_intent_prompt
from swe_mas.agents.base import AgentConfig, BaseAgent
from swe_mas.utils.command_classification import (
    looks_like_readonly_probe_command,
    looks_like_validation_command,
)
from swe_mas.utils.logger import get_logger
from swe_mas.utils.path_filters import classify_changed_files, normalize_repo_path
from swe_mas.utils.probe_analysis import (
    classify_probe_delta,
    focused_readonly_probe_summary,
    no_progress_probe_streak,
    summarize_probe_paths,
    summarize_regions_or_symbols,
)

logger = get_logger(__name__)


class ImplementerAgent(BaseAgent):
    """代码实现专家
    
    职责：根据修复计划执行具体的代码修改，生成git diff补丁
    """

    MAX_READONLY_STREAK = 4
    RECOVERY_READONLY_STREAK = 2
    MAX_FOCUSED_READONLY_STREAK = 6
    RECOVERY_FOCUSED_READONLY_STREAK = 4
    
    def __init__(self, *args, **kwargs):
        config = kwargs.pop("config", AgentConfig())
        if not config.phase:
            config.phase = "implement"
        super().__init__(*args, agent_type="implementer", config=config, **kwargs)
        self._recovery_diff_baseline: dict[str, Any] = {}
        self._force_recovery_mode = False

    def set_recovery_diff_baseline(self, baseline: dict[str, Any] | None) -> None:
        self._recovery_diff_baseline = dict(baseline or {})

    def _select_bash_command_from_blocks(self, commands: list[str]) -> str | None:
        """Recovery mode should not drop a later edit/validation block.

        SWE recovery traces show a common failure pattern: the model emits a
        read-only inspection block followed by the actual edit or focused test.
        The base agent historically executed the first block, which silently
        discarded the recovery action's concrete execution.  For implementer
        recovery we choose the first write or validation block, keeping normal
        non-recovery behavior unchanged.
        """

        if not commands:
            return None
        if not getattr(self, "_recovery_mode", False):
            return commands[0]
        write_candidates: list[str] = []
        for command in commands:
            if self._looks_like_write_command(command):
                write_candidates.append(command)
        for command in write_candidates:
            if self._write_command_targets_tmp_only(command):
                continue
            if self._write_command_targets_repo(command):
                return command
        for command in write_candidates:
            if not self._write_command_targets_tmp_only(command):
                return command
        if write_candidates:
            return write_candidates[0]
        for command in commands:
            if self._looks_like_validation_command(command):
                return command
        return commands[0]

    @staticmethod
    def _write_command_targets_tmp_only(command: str) -> bool:
        text = str(command or "")
        if not text.strip():
            return False
        absolute_paths = re.findall(r"(?<![A-Za-z0-9_./-])(/[A-Za-z0-9_./-]+)", text)
        redirection_targets = re.findall(r"(?:^|\s)(?:>|>>)\s*([^\s]+)", text)
        candidates = [token.strip("'\"") for token in absolute_paths + redirection_targets if token.strip("'\"")]
        writable_targets = [
            item
            for item in candidates
            if item.startswith("/")
            and not item.startswith(("/dev/", "/proc/", "/sys/"))
        ]
        return bool(writable_targets) and all(
            item == "/tmp" or item.startswith("/tmp/") for item in writable_targets
        )

    @staticmethod
    def _write_command_targets_repo(command: str) -> bool:
        text = str(command or "")
        if not text.strip():
            return False
        non_repo_abs = r"(?!/tmp/)(?!/dev/)(?!/proc/)(?!/sys/)"
        if re.search(rf"(?:Path\s*\(|open\s*\()\s*['\"]{non_repo_abs}[^'\"]+['\"]", text):
            return True
        if re.search(r"(?:Path\s*\(|open\s*\()\s*['\"](?:\./)?[A-Za-z0-9_./-]+\.pyi?['\"]", text):
            return True
        if re.search(rf"(?:^|\s)(?:>|>>)\s*{non_repo_abs}[^\s]+", text):
            return True
        if re.search(r"(?:^|\s)(?:>|>>)\s*(?:\./)?[A-Za-z0-9_./-]+\.pyi?(?:\s|$)", text):
            return True
        if re.search(r"\bgit\s+apply\b", text):
            return True
        if re.search(r"\b(?:sed|perl)\b[^\n;]*\s-i(?:\s|$)", text) and "/tmp/" not in text:
            return True
        return False
    
    def run(self, fix_plan: str, cwd: str = "", problem_statement: str | None = None) -> dict[str, Any]:
        """实现代码修改
        
        Args:
            fix_plan: 修复计划
            cwd: 工作目录
            problem_statement: 原始问题描述/恢复上下文（可选）
            
        Returns:
            {"patch": "git diff输出", "success": bool, "commands": [...]}
        """
        logger.info(f"[Implementer] 开始实现代码修改...")
        
        # 设置工作目录
        if cwd:
            self.executor.config.cwd = cwd
        self._recovery_mode = self._should_use_recovery_mode(problem_statement)
        self._parc_patch_contract = parse_patch_contract_prompt(problem_statement or "")
        self._car_patch_intent = parse_patch_intent_prompt(problem_statement or "")
        self._masguard_source_edit_contract = "[MASGUARD SOURCE EDIT CONTRACT]" in str(problem_statement or "")
        self._masguard_source_edit_targets = self._extract_masguard_source_edit_targets(problem_statement or "")
        
        # 记录Phase开始（用于恢复）
        self._start_phase({"fix_plan": fix_plan, "cwd": cwd, "problem_statement": problem_statement})
        
        # 检查是否有恢复状态（resume模式）
        if self._recovery_state:
            # Resume模式：恢复之前的对话和执行历史
            self.messages = self._recovery_state.get("messages", [])
            self.history = self._recovery_state.get("history", [])
            self.iteration = self._recovery_state.get("last_iteration", 0) + 1
            logger.info(f"[Implementer] Resume模式 - 从第{self.iteration}次迭代继续")
        else:
            # 正常模式：初始化对话
            system_prompt = self.prompts.get("system", "")
            problem_text = (problem_statement or "").strip()
            if len(problem_text) > 20000:
                problem_text = problem_text[:15000] + "\n...[truncated]...\n" + problem_text[-5000:]
            prompt_cwd = self._prompt_cwd()
            user_prompt = self.render_template(
                self.prompts.get("user", ""),
                fix_plan=fix_plan,
                cwd=prompt_cwd,
                problem_statement=problem_text,
            )
            
            self.messages = []
            self.history = []
            self.add_message("system", system_prompt)
            self.add_message("user", user_prompt)
            self.iteration = 0
        
        # 迭代执行（从self.iteration开始）
        patch = ""
        for iteration in range(self.iteration, self.config.max_iterations):
            self.iteration = iteration
            if self._stop_requested():
                logger.warning(f"[Implementer] 收到停止信号，提前结束于第{iteration}次迭代")
                result = {
                    "patch": "",
                    "success": False,
                    "error": "stop_requested",
                    "commands": self.history.copy(),
                    "messages": self.messages.copy(),
                    **self._result_metadata(
                        patch_summary=self._patch_workspace_summary(),
                        stop_reason="stop_requested",
                    ),
                }
                self._end_phase(result, success=False, cwd=self.executor.config.cwd)
                return result
            try:
                # 查询模型
                response = self.query_model()

                # 提取命令
                command = self.parse_bash_command(response)
                if not command:
                    self.add_message(
                        "user",
                        "格式错误：上一条回复没有可执行命令。下一条回复只能输出一个```bash```代码块；"
                        "不要输出分析、计划、Markdown列表或JSON。",
                    )
                    continue
                force_edit_analysis = self._force_edit_or_validation_analysis(command)
                if force_edit_analysis is not None:
                    focused_paths = list(force_edit_analysis.get("focused_paths", []) or [])
                    reminder = (
                        "提醒：你已经连续多次只读探索（如 ls/cat/head/sed -n），但还没有进入有效的修改或验证。"
                        "下一步必须执行以下两类命令之一：\n"
                        "1. 直接修改目标代码；或\n"
                        "2. 对已经做出的修改运行语法/测试验证。\n"
                        "不要继续重复查看同一文件内容。"
                    )
                    if focused_paths:
                        reminder = (
                            "提醒：你已经围绕同一目标文件连续深读了很多轮，说明定位证据已经足够。"
                            f"当前最集中的目标是：{', '.join(focused_paths[:2])}。\n"
                            "下一步必须直接修改这些源码文件中的具体逻辑，或对已修改内容运行语法/测试验证；"
                            "不要继续只通过新的 sed/head 区间来浏览同一文件。"
                        )
                    if getattr(self, "_recovery_mode", False):
                        reminder = (
                            "提醒：当前是失败后的恢复，不是从头解题。你已经连续多次只读探索，但还没有形成有效源码修改。"
                            "下一步必须直接修改修复计划或原问题里已经指出的规范源码目标，"
                            "或对已有源码修改运行验证；不要继续停留在只读探索，也不要把 build/dist/generated 等目录当主目标。"
                        )
                        if focused_paths:
                            reminder = (
                                "提醒：当前是失败后的恢复，不是从头解题。你已经围绕同一规范源码目标连续深读了很多轮，"
                                f"当前最集中的目标是：{', '.join(focused_paths[:2])}。\n"
                                "下一步必须直接修改这些源码目标或对已有修改运行 focused 验证；"
                                "不要继续停留在只读探索，也不要把 build/dist/generated 等目录当主目标。"
                            )
                        if bool(force_edit_analysis.get("candidate_preserving_overread", False)):
                            reminder = (
                                "提醒：当前是候选补丁精修模式。上一轮已经有源码候选和失败证据，"
                                "本轮不能继续把预算花在重新浏览文件上。\n"
                                f"候选源码目标：{', '.join(focused_paths[:3]) if focused_paths else '见 CAR preserved source candidate paths'}。\n"
                                "下一步必须二选一：直接对候选源码做最小修正，或运行 focused failing test / py_compile 验证当前候选。"
                            )
                        elif bool(force_edit_analysis.get("source_diff_pending_validation", False)):
                            reminder = (
                                "提醒：当前恢复回合已经产生源码 diff，但这次源码候选还没有经过 focused 验证。"
                                f"候选源码目标：{', '.join(focused_paths[:3]) if focused_paths else '见当前 git diff'}。\n"
                                "下一步必须运行失败用例、项目原生 focused test、pytest/nosetests/tox 或 py_compile；"
                                "如果验证失败，再根据失败输出做最小源码精修。不要继续只读浏览。"
                            )
                        elif bool(force_edit_analysis.get("post_evidence_validation_before_source_diff", False)):
                            reminder = (
                                "提醒：当前是 CAR 证据重查后的源码修复动作。"
                                "本轮还没有产生 fresh source diff，不能先运行验证来消耗恢复预算。\n"
                                f"必须先直接修改目标源码：{', '.join(focused_paths[:3]) if focused_paths else '见 CAR target paths'}。"
                                "修改后再运行 focused failing test 或 py_compile。"
                            )
                        elif bool(force_edit_analysis.get("post_evidence_readonly_over_budget", False)):
                            reminder = (
                                "提醒：CAR 已经完成证据重查并选择 LOCAL_REPAIR。"
                                "当前只读查看预算已经用完，继续浏览不会执行恢复动作。\n"
                                f"下一步必须直接修改目标源码：{', '.join(focused_paths[:3]) if focused_paths else '见 CAR target paths'}，"
                                "或明确说明为什么这些目标无法编辑。"
                            )
                        elif bool(force_edit_analysis.get("strict_source_edit_readonly_over_budget", False)):
                            reminder = (
                                "提醒：当前是 MASGuard strict source-edit 恢复分支。"
                                "定位和最小探针证据已经足够进入源码编辑，继续只读浏览会导致 no-diff 失败。\n"
                                f"允许/优先源码目标：{', '.join(focused_paths[:3]) if focused_paths else '见 MASGuard SOURCE EDIT CONTRACT'}。\n"
                                "下一条回复只能输出一个 ```bash``` 代码块，并且必须二选一："
                                "1) 直接对上述源码目标做最小修改；或 2) 明确写入一个无修改放弃标记并说明证据不足。"
                            )
                    self.add_message(
                        "user",
                        reminder,
                    )
                    continue
                if self._stop_requested():
                    logger.warning(f"[Implementer] 收到停止信号，跳过第{iteration}次迭代执行")
                    result = {
                        "patch": "",
                        "success": False,
                        "error": "stop_requested",
                        "commands": self.history.copy(),
                        "messages": self.messages.copy(),
                        **self._result_metadata(
                            patch_summary=self._patch_workspace_summary(),
                            stop_reason="stop_requested",
                        ),
                    }
                    self._end_phase(result, success=False, cwd=self.executor.config.cwd)
                    return result

                brittle_script_feedback = self._masguard_strict_brittle_patch_script_feedback(command)
                if brittle_script_feedback:
                    self.add_message("user", brittle_script_feedback)
                    continue

                # 执行命令
                result = self.execute_command(command)
                self._record_probe_history(command)

                strict_abstain_feedback = self._masguard_strict_abstain_feedback(result)
                if strict_abstain_feedback:
                    self.add_message("user", strict_abstain_feedback)
                    continue

                no_diff_completion_feedback = self._masguard_no_diff_completion_feedback(command, result)
                if no_diff_completion_feedback:
                    self.add_message("user", no_diff_completion_feedback)
                    continue

                failed_validation_feedback = self._masguard_failed_validation_feedback(command, result)
                if failed_validation_feedback:
                    self.add_message("user", failed_validation_feedback)
                    continue

                contract_feedback = self._recovery_source_contract_feedback(command)
                if contract_feedback:
                    self.add_message("user", contract_feedback)
                    continue

                # 如果刚完成一次验证且测试通过，并且工作区已有改动，
                # 直接尝试收尾生成补丁，避免继续空转到最大迭代。
                finalized = self._maybe_finalize_after_successful_validation(command, result)
                if finalized is not None:
                    self._end_phase(finalized, success=True, cwd=self.executor.config.cwd)
                    return finalized
                
                # 检查是否完成（生成了补丁）
                # 兼容两种标记：COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT 和 IMPLEMENTATION_COMPLETE
                if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in result["output"] or "IMPLEMENTATION_COMPLETE" in result["output"]:
                    # 提取git diff输出
                    lines = result["output"].split("\n")
                    # 跳过标记行，提取diff内容
                    patch_lines = []
                    started = False
                    for line in lines:
                        if started or line.strip().startswith("diff --git"):
                            started = True
                            patch_lines.append(line)
                    
                    finalized = self._finalize_effective_patch_result()
                    if finalized is not None:
                        logger.info(f"[Implementer] 实现完成，生成有效补丁")
                        self._end_phase(finalized, success=True, cwd=self.executor.config.cwd)
                        return finalized
                    patch = ""
                    logger.warning(f"[Implementer] 未检测到有效源码补丁，继续尝试")
                
                # 添加观察结果
                observation = f"命令输出:\n{result['output'][:2000]}"
                if result["returncode"] != 0:
                    observation += f"\n返回码: {result['returncode']}"
                self.add_message("user", observation)
                
            except Exception as e:
                logger.error(f"[Implementer] 迭代{iteration}出错: {str(e)}")
                result = {
                    "patch": "",
                    "success": False,
                    "error": str(e),
                    "commands": self.history.copy(),
                    "messages": self.messages.copy(),
                    **self._result_metadata(
                        patch_summary=self._patch_workspace_summary(),
                        stop_reason="runtime_error",
                    ),
                }
                self._end_phase(result, success=False, cwd=self.executor.config.cwd)
                return result
        
        # 达到最大迭代次数，尝试自动收尾：若存在改动则生成补丁
        logger.warning(f"[Implementer] 达到最大迭代次数")

        try:
            finalized = self._finalize_effective_patch_result()
            if finalized is not None:
                logger.info("[Implementer] 自动生成有效补丁（收尾阶段）")
                self._end_phase(finalized, success=True, cwd=self.executor.config.cwd)
                return finalized
        except Exception as e:
            logger.warning(f"[Implementer] 自动收尾生成补丁失败: {str(e)}")

        patch_summary = self._patch_workspace_summary()
        result = {
            "patch": "未生成补丁",
            "success": False,
            "commands": self.history.copy(),
            "messages": self.messages.copy(),
            "patch_summary": patch_summary,
            **self._result_metadata(
                patch_summary=patch_summary,
                stop_reason=self._terminal_stop_reason(patch_summary),
            ),
        }
        self._end_phase(result, success=False, cwd=self.executor.config.cwd)
        return result

    def _maybe_finalize_after_successful_validation(
        self,
        command: str,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if result.get("returncode") != 0:
            return None
        if not self._looks_like_validation_command(command):
            return None
        try:
            finalized = self._finalize_effective_patch_result()
            if finalized is not None:
                logger.info("[Implementer] 验证通过后自动收尾生成有效补丁")
                return finalized
        except Exception as e:
            logger.warning(f"[Implementer] 验证通过后自动收尾失败: {str(e)}")
        return None

    def _patch_workspace_summary(self) -> dict[str, Any]:
        def _git_lines(command: str) -> list[str]:
            result = self.executor.execute(command)
            return [
                line.strip()
                for line in str(result.get("output", "") or "").splitlines()
                if line.strip()
            ]

        def _numstat_entries(command: str) -> dict[str, dict[str, int | bool]]:
            entries: dict[str, dict[str, int | bool]] = {}
            for line in _git_lines(command):
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                added, deleted = parts[0].strip(), parts[1].strip()
                if (added, deleted) in {("0", "0"), ("-", "-")}:
                    continue
                normalized = normalize_repo_path(parts[-1].strip())
                if normalized:
                    binary = added == "-" or deleted == "-"
                    added_count = int(added) if added.isdigit() else 0
                    deleted_count = int(deleted) if deleted.isdigit() else 0
                    current = entries.setdefault(
                        normalized,
                        {"added": 0, "deleted": 0, "total": 0, "binary": False},
                    )
                    current["added"] = int(current.get("added", 0) or 0) + added_count
                    current["deleted"] = int(current.get("deleted", 0) or 0) + deleted_count
                    current["total"] = int(current.get("total", 0) or 0) + added_count + deleted_count
                    current["binary"] = bool(current.get("binary", False) or binary)
            return entries

        untracked_files = [
            normalize_repo_path(path)
            for path in _git_lines("git ls-files --others --exclude-standard")
            if normalize_repo_path(path)
        ]
        numstat: dict[str, dict[str, int | bool]] = {}
        for source in (
            _numstat_entries("git diff --numstat"),
            _numstat_entries("git diff --cached --numstat"),
        ):
            for path, entry in source.items():
                current = numstat.setdefault(path, {"added": 0, "deleted": 0, "total": 0, "binary": False})
                current["added"] = int(current.get("added", 0) or 0) + int(entry.get("added", 0) or 0)
                current["deleted"] = int(current.get("deleted", 0) or 0) + int(entry.get("deleted", 0) or 0)
                current["total"] = int(current.get("total", 0) or 0) + int(entry.get("total", 0) or 0)
                current["binary"] = bool(current.get("binary", False) or entry.get("binary", False))
        for path in untracked_files:
            numstat.setdefault(path, {"added": 0, "deleted": 0, "total": 0, "binary": False})
        changed_files = sorted(
            dict.fromkeys(
                list(numstat)
                + untracked_files
            )
        )
        classes = classify_changed_files(changed_files)
        if not changed_files:
            failure_mode = "no_effective_patch"
        elif classes["generated_files"] and not classes["effective_files"]:
            failure_mode = "generated_only_patch"
        else:
            failure_mode = "effective_patch"
        target_legitimacy = self._target_legitimacy(classes, changed_files)
        source_patch_risk = self._source_patch_risk(numstat, list(classes.get("source_files", []) or []))
        return {
            "changed_files": changed_files,
            "changed_file_classes": classes,
            "patch_numstat": numstat,
            "source_patch_risk": source_patch_risk,
            "failure_mode": failure_mode,
            "target_legitimacy": target_legitimacy,
        }

    def _fresh_source_files_since_recovery_baseline(self) -> list[str] | None:
        baseline = dict(getattr(self, "_recovery_diff_baseline", {}) or {})
        if not baseline:
            return None
        fresh_files = self._fresh_changed_files_since_recovery_baseline()
        if fresh_files is None:
            return None
        return list(classify_changed_files(fresh_files).get("source_files", []) or [])

    def _fresh_changed_files_since_recovery_baseline(self) -> list[str] | None:
        baseline = dict(getattr(self, "_recovery_diff_baseline", {}) or {})
        if not baseline:
            return None
        try:
            summary = self._patch_workspace_summary()
        except Exception:
            return None
        current_files = [
            normalize_repo_path(str(path))
            for path in list(summary.get("changed_files", []) or [])
            if normalize_repo_path(str(path))
        ]
        baseline_digests = dict(baseline.get("file_digests", {}) or {})
        if baseline_digests:
            current_digests = self._current_diff_file_digests()
            return sorted(
                path
                for path, digest in current_digests.items()
                if str(digest or "") != str(baseline_digests.get(path, "") or "")
            )

        baseline_files = {
            normalize_repo_path(str(path))
            for path in list(baseline.get("changed_files", []) or [])
            if normalize_repo_path(str(path))
        }
        return sorted(path for path in current_files if path and path not in baseline_files)

    def _current_diff_file_digests(self) -> dict[str, str]:
        raw_diff = "\n".join(
            text
            for text in (
                str(self.executor.execute("git diff --binary").get("output", "") or ""),
                str(self.executor.execute("git diff --cached --binary").get("output", "") or ""),
            )
            if text.strip()
        )
        return self._diff_file_digests(raw_diff)

    @staticmethod
    def _diff_file_digests(raw_diff: str) -> dict[str, str]:
        file_chunks: dict[str, list[str]] = {}
        current_path = ""
        current_lines: list[str] = []
        for line in str(raw_diff or "").splitlines():
            if line.startswith("diff --git "):
                if current_path:
                    file_chunks[current_path] = list(current_lines)
                current_path = ImplementerAgent._path_from_diff_header(line)
                current_lines = [line]
                continue
            if current_path:
                current_lines.append(line)
        if current_path:
            file_chunks[current_path] = list(current_lines)
        import hashlib

        return {
            path: hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]
            for path, lines in sorted(file_chunks.items())
            if path
        }

    @staticmethod
    def _path_from_diff_header(header: str) -> str:
        parts = header.split()
        if len(parts) < 4:
            return ""
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path == "/dev/null" and len(parts) >= 3:
            path = parts[2]
            if path.startswith("a/"):
                path = path[2:]
        return normalize_repo_path(path)

    def _cleanup_generated_artifacts(self, generated_files: list[str]) -> None:
        if not generated_files:
            return
        quoted = " ".join(shlex.quote(path) for path in generated_files)
        self.executor.execute(f"git restore --staged --worktree -- {quoted} || true")
        self.executor.execute(f"git clean -fd -- {quoted} || true")

    def _cleanup_non_source_recovery_edits(self, paths: list[str]) -> None:
        paths = [path for path in paths if path]
        if not paths:
            return
        quoted = " ".join(shlex.quote(path) for path in paths)
        self.executor.execute(f"git restore --staged --worktree -- {quoted} || true")
        self.executor.execute(f"git clean -fd -- {quoted} || true")

    def _recovery_source_contract_feedback(self, command: str) -> str:
        if not getattr(self, "_recovery_mode", False):
            return ""
        if not self._looks_like_write_command(command):
            return ""

        summary = self._patch_workspace_summary()
        changed_files = list(summary.get("changed_files", []) or [])
        classes = dict(summary.get("changed_file_classes", {}) or {})
        source_files = list(classes.get("source_files", []) or [])
        test_files = list(classes.get("test_files", []) or [])
        generated_files = list(classes.get("generated_files", []) or [])
        other_files = list(classes.get("other_files", []) or [])
        baseline_feedback = self._recovery_baseline_fresh_diff_feedback(
            changed_files=changed_files,
            source_files=source_files,
        )
        if baseline_feedback:
            return baseline_feedback
        if source_files:
            risk_feedback = self._source_patch_risk_feedback(summary)
            if risk_feedback:
                return risk_feedback
        parc_contract = dict(getattr(self, "_parc_patch_contract", {}) or {})
        if parc_contract:
            feedback = self._parc_patch_contract_feedback(
                changed_files=changed_files,
                classes=classes,
                source_files=source_files,
                test_files=test_files,
                generated_files=generated_files,
            )
            if feedback:
                return feedback
        car_intent = dict(getattr(self, "_car_patch_intent", {}) or {})
        if car_intent:
            feedback = self._car_patch_intent_feedback(
                command=command,
                changed_files=changed_files,
                source_files=source_files,
            )
            if feedback:
                return feedback

        if source_files:
            if test_files:
                return (
                    "恢复契约提醒：你已经产生源码改动，但同时也修改了测试文件。"
                    "后续优先继续验证和修正源码；不要再扩大测试文件改动，除非问题明确要求。"
                )
            return ""

        if not changed_files:
            return (
                "恢复契约失败：上一条写命令返回成功，但没有产生任何 git diff。"
                "这通常说明替换模式没有匹配到目标代码。下一步必须检查精确行内容，"
                "然后对规范源码文件产生实际 diff；不要改测试文件来绕过失败。"
            )

        if test_files or generated_files:
            cleanup_paths = test_files + generated_files
            self._cleanup_non_source_recovery_edits(cleanup_paths)
            return (
                "恢复契约失败：当前改动没有包含源码文件，只包含测试/生成文件改动。"
                f"已撤销这些非源码恢复改动：{', '.join(cleanup_paths[:8])}。"
                "下一步必须修改失败证据和可疑路径指向的规范源码文件；"
                "不要通过只改测试文件来完成恢复。"
            )

        if other_files:
            return (
                "恢复契约提醒：当前改动没有包含源码文件，只包含其他非源码文件。"
                "请优先修改可疑源码路径并运行 focused failing test；"
                "如果确实必须修改配置/数据文件，请先用测试失败证据说明原因。"
            )
        return ""

    def _masguard_strict_abstain_feedback(self, result: dict[str, Any]) -> str:
        if not getattr(self, "_recovery_mode", False):
            return ""
        if not getattr(self, "_masguard_source_edit_contract", False):
            return ""
        output = str(dict(result).get("output", "") or "")
        if "MASGUARD_STRICT_ABSTAIN_NO_EDIT" not in output:
            return ""
        reason_match = re.search(r"MASGUARD_ABSTAIN_REASON\s*[:=]\s*(.+)", output)
        reason = reason_match.group(1).strip()[:240] if reason_match else ""
        reason_line = f" 上一轮放弃原因：{reason}。" if reason else ""
        return (
            "MASGuard strict source-edit脚本没有产生源码 diff：上一条脚本进入了"
            "`MASGUARD_STRICT_ABSTAIN_NO_EDIT` 分支，通常说明精确文本 anchor 没匹配。"
            f"{reason_line}"
            "在 strict source-edit 合同里，no-diff/abstain 是被拒绝候选，不是成功收尾。"
            "下一条回复仍然只能输出一个 bash 代码块，但必须换成更稳健的补丁方式："
            "可以先做一个源文件内的最小 bounded probe 来确认 anchor，但同一个 bash 块随后必须"
            "用短正则、行级 anchor 或 ast 定位目标函数/类/分支，再做最小源码修改；"
            "推荐使用 Python `re.subn(..., count=1)` 并断言替换次数等于 1，"
            "或用 ast 节点 lineno/end_lineno 计算行级插入位置。"
            "不要重复大段多行字符串替换；不要在 no-op/abstain 路径后输出"
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT 或 IMPLEMENTATION_COMPLETE。"
            "如果仍无法编辑，必须打印新的 MASGUARD_ABSTAIN_REASON 并以非零码退出。"
        )

    def _masguard_no_diff_completion_feedback(
        self,
        command: str,
        result: dict[str, Any],
    ) -> str:
        if not getattr(self, "_recovery_mode", False):
            return ""
        if not getattr(self, "_masguard_source_edit_contract", False):
            return ""
        output = str(dict(result).get("output", "") or "")
        if (
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in output
            and "IMPLEMENTATION_COMPLETE" not in output
        ):
            return ""
        try:
            summary = self._patch_workspace_summary()
        except Exception:
            summary = {}
        classes = dict(summary.get("changed_file_classes", {}) or {})
        source_files = [
            normalize_repo_path(str(path))
            for path in list(classes.get("source_files", []) or [])
            if normalize_repo_path(str(path))
        ]
        fresh_source_files = self._fresh_source_files_since_recovery_baseline()
        if fresh_source_files is None:
            fresh_source_files = source_files
        if fresh_source_files:
            return ""
        targets = [
            normalize_repo_path(str(path))
            for path in list(getattr(self, "_masguard_source_edit_targets", []) or [])
            if normalize_repo_path(str(path))
        ]
        target_hint = ", ".join(targets[:4]) if targets else "见 MASGuard SOURCE EDIT CONTRACT"
        failure_mode = str(summary.get("failure_mode", "") or "no_fresh_source_diff")
        return (
            "MASGuard strict source-edit 收尾被阻止：上一条命令输出了完成标记，"
            f"但没有产生 fresh canonical source diff（failure_mode={failure_mode}）。"
            "这类 no-diff completion 在当前方法里是被拒绝候选，不能进入 oracle。"
            f"下一条回复只能输出一个 bash 代码块；允许先对目标源码做一个最小局部探针，目标：{target_hint}；"
            "随后必须用 AST 行号、短正则或行级 anchor 产生最小源码 diff，并用 git diff --quiet 断言 diff 存在。"
            "不得只打印完成标记、不得只改测试/生成文件；如果证据仍不足，打印 MASGUARD_STRICT_ABSTAIN_NO_EDIT "
            "和 MASGUARD_ABSTAIN_REASON 并以非零码退出。"
        )

    def _masguard_strict_brittle_patch_script_feedback(self, command: str) -> str:
        if not getattr(self, "_recovery_mode", False):
            return ""
        if not getattr(self, "_masguard_source_edit_contract", False):
            return ""
        if not self._looks_like_write_command(command):
            return ""
        text = str(command or "")
        if "re.subn" in text or "ast.parse" in text or "lineno" in text:
            return ""
        has_noop_success_path = (
            "MASGUARD_STRICT_ABSTAIN_NO_EDIT" in text
            and (
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in text
                or "IMPLEMENTATION_COMPLETE" in text
            )
        )
        large_literal = re.search(
            r"(?:old|before|target)\s*=\s*([\"']{3})(?:(?!\1).){160,}\1",
            text,
            flags=re.DOTALL,
        )
        brittle_replace = bool(large_literal and re.search(r"\.replace\(\s*(?:old|before|target)\s*,", text))
        if not has_noop_success_path and not brittle_replace:
            return ""
        return (
            "MASGuard strict source-edit执行前阻止了一个脆弱补丁脚本："
            "该脚本使用大段精确多行字符串替换，或在 no-op/abstain 路径后仍可能输出完成标记。"
            "下一条回复只能输出一个 bash 代码块，并必须改用稳健补丁模板："
            "使用 `re.subn(..., count=1)` 的短局部 anchor，或用 `ast.parse` 的"
            " lineno/end_lineno 定位函数/类/分支；替换次数必须等于 1，"
            "写入后必须确认 `git diff --quiet` 为非零，再运行 focused validation。"
            "如果无法定位，退出非零并打印附近上下文；不要输出完成标记。"
        )

    def _masguard_failed_validation_feedback(self, command: str, result: dict[str, Any]) -> str:
        if not getattr(self, "_recovery_mode", False):
            return ""
        if not getattr(self, "_masguard_source_edit_contract", False):
            return ""
        if int(dict(result).get("returncode", 0) or 0) == 0:
            return ""
        if not self._looks_like_validation_command(command):
            return ""
        summary = self._patch_workspace_summary()
        classes = dict(summary.get("changed_file_classes", {}) or {})
        source_files = [
            normalize_repo_path(str(path))
            for path in list(classes.get("source_files", []) or [])
            if normalize_repo_path(str(path))
        ]
        if not source_files:
            return ""
        output = str(dict(result).get("output", "") or "")
        excerpt = self._validation_failure_excerpt(output)
        target_hint = ", ".join(source_files[:4])
        return (
            "MASGuard strict source-edit focused validation failed after a source diff. "
            f"Keep the current source target set ({target_hint}) and use the validation failure as the next bounded probe. "
            "Do not restart localization or abandon the patch. The next reply must contain exactly one bash block that "
            "makes a minimal source-only refinement to the existing changed file, then reruns the same focused validation "
            "or `python -m py_compile` on the edited file. Use a robust line-local/regex/AST anchor and require exactly "
            f"one intended edit. Validation excerpt: {excerpt}"
        )

    @staticmethod
    def _validation_failure_excerpt(output: str, limit: int = 900) -> str:
        lines = [line.rstrip() for line in str(output or "").splitlines()]
        interesting = [
            line
            for line in lines
            if any(
                token in line
                for token in (
                    "AssertionError",
                    "Traceback",
                    "E   ",
                    "FAILED",
                    "ERROR",
                    "Exception",
                    "returncode",
                )
            )
        ]
        text = "\n".join(interesting[-12:] if interesting else lines[-12:])
        text = text.strip().replace("\x00", "")
        if len(text) > limit:
            text = text[-limit:]
        return text or "(validation failed without captured output)"

    def _recovery_finalize_block_feedback(self, summary: dict[str, Any], *, reason: str) -> str:
        if not getattr(self, "_recovery_mode", False):
            return ""
        source_files, _fresh_baseline_known = self._recovery_fresh_source_files_for_finalize(summary)
        source_hint = ", ".join(source_files[:4]) if source_files else "no fresh source file"
        if reason == "missing_focused_validation_after_source_edit":
            return (
                "Recovery finalize blocked: a source diff exists but no focused validation has run after the latest "
                f"source edit ({source_hint}). Next reply must run the focused failing test, project-native focused "
                "test, or `python -m py_compile` for the edited file. If it fails, make one minimal source-only "
                "refinement from that output before trying to finish."
            )
        if reason == "no_fresh_source_diff":
            return (
                "Recovery finalize blocked: the current diff is not a fresh source diff for this replay. "
                "Next reply must make a semantically new minimal source edit on the bounded target, using a robust "
                "line-local/regex/AST anchor, then validate it. Do not resubmit the stale patch."
            )
        return ""

    def _recovery_baseline_fresh_diff_feedback(
        self,
        *,
        changed_files: list[str],
        source_files: list[str],
    ) -> str:
        if not self._recovery_requires_fresh_source_diff():
            return ""
        fresh_source_files = self._fresh_source_files_since_recovery_baseline()
        if fresh_source_files is None:
            return ""
        if fresh_source_files:
            return ""
        if not source_files:
            return ""
        normalized_sources = [
            normalize_repo_path(str(path))
            for path in list(source_files or [])
            if normalize_repo_path(str(path))
        ]
        normalized_changed = [
            normalize_repo_path(str(path))
            for path in list(changed_files or [])
            if normalize_repo_path(str(path))
        ]
        target = ", ".join((normalized_sources or normalized_changed)[:5])
        return (
            "CAR fresh-diff 契约失败：上一条写命令之后，工作区虽然有源码 diff，"
            "但这些 diff 与本次 replay 开始前的 baseline 相同，不是本轮 fresh source diff。"
            f"当前陈旧源码 diff：{target or '见 git diff'}。"
            "下一步必须基于当前失败证据对规范源码目标做新的、语义不同的最小修改；"
            "不要再次写回同一个旧补丁，也不要在没有 fresh source diff 的情况下结束。"
        )

    @staticmethod
    def _source_patch_risk(
        numstat: dict[str, dict[str, int | bool]],
        source_files: list[str],
    ) -> dict[str, Any]:
        source_set = {normalize_repo_path(path) for path in source_files if normalize_repo_path(path)}
        large_rewrite_files: list[str] = []
        broad_change_files: list[str] = []
        total_changed_lines = 0
        max_file_changed_lines = 0
        for path in sorted(source_set):
            entry = dict(numstat.get(path, {}) or {})
            added = int(entry.get("added", 0) or 0)
            deleted = int(entry.get("deleted", 0) or 0)
            total = int(entry.get("total", 0) or 0)
            total_changed_lines += total
            max_file_changed_lines = max(max_file_changed_lines, total)
            if total >= 300 or (added >= 120 and deleted >= 120):
                large_rewrite_files.append(path)
            elif total >= 120:
                broad_change_files.append(path)
        risk_level = "large_source_rewrite" if large_rewrite_files else "broad_source_change" if broad_change_files else "low"
        return {
            "risk_level": risk_level,
            "source_file_count": len(source_set),
            "total_changed_lines": total_changed_lines,
            "max_file_changed_lines": max_file_changed_lines,
            "large_rewrite_files": large_rewrite_files,
            "broad_change_files": broad_change_files,
        }

    def _source_patch_risk_feedback(self, summary: dict[str, Any]) -> str:
        risk = dict(summary.get("source_patch_risk", {}) or {})
        risk_level = str(risk.get("risk_level", "") or "")
        if risk_level not in {"large_source_rewrite", "broad_source_change"}:
            return ""
        files = [
            str(path)
            for path in list(
                risk.get("large_rewrite_files", [])
                or risk.get("broad_change_files", [])
                or []
            )
            if str(path).strip()
        ]
        if not files:
            return ""
        self._cleanup_non_source_recovery_edits(files)
        return (
            "CAR补丁规模失败：当前恢复动作产生了疑似整文件或过宽源码改动，"
            f"风险级别：{risk_level}；高风险文件：{', '.join(files[:4])}；"
            f"最大单文件改动约 {int(risk.get('max_file_changed_lines', 0) or 0)} 行。"
            "系统已撤回这些高风险源码改动。恢复补丁必须是最小、可验证的源码改动；"
            "下一步请基于失败证据对目标函数/分支做小范围修改，并立即运行 focused 验证。"
        )

    def _car_patch_intent_feedback(
        self,
        *,
        command: str = "",
        changed_files: list[str],
        source_files: list[str],
    ) -> str:
        intent = dict(getattr(self, "_car_patch_intent", {}) or {})
        if not intent:
            return ""

        selected_action = str(intent.get("selected_action", "") or "")
        paths = dict(intent.get("paths", {}) or {})
        requirements = dict(intent.get("requirements", {}) or {})
        target_paths = [
            normalize_repo_path(str(item))
            for item in list(intent.get("target_paths", paths.get("target_paths", [])) or [])
            if normalize_repo_path(str(item))
        ]
        candidate_paths = [
            normalize_repo_path(str(item))
            for item in list(intent.get("candidate_source_paths", paths.get("candidate_source_paths", [])) or [])
            if normalize_repo_path(str(item))
        ]
        avoid_paths = [
            normalize_repo_path(str(item))
            for item in list(intent.get("avoid_target_paths", paths.get("avoid_target_paths", [])) or [])
            if normalize_repo_path(str(item))
        ]
        require_target_touch = bool(
            intent.get("require_target_touch", requirements.get("touch_intended_target", False))
        )
        preserve_candidate = bool(
            intent.get("preserve_candidate_source", requirements.get("preserve_candidate_source", False))
        )
        max_files = int(intent.get("max_fresh_source_files", requirements.get("max_fresh_source_files", 3)) or 3)
        directives = {
            str(item or "").strip().upper()
            for item in list(intent.get("directives", []) or [])
            if str(item or "").strip()
        }

        require_fresh_source = bool(
            intent.get("require_fresh_source_diff", requirements.get("fresh_source_diff", True))
        )
        if require_fresh_source and changed_files and not source_files:
            return (
                "CAR补丁意图提醒：当前写入没有产生源码 diff。"
                "下一步必须把结构化恢复动作落到规范源码目标，而不是旁路文件。"
            )

        if max_files > 0 and len(source_files) > max_files:
            return (
                "CAR补丁意图提醒：当前源码改动范围过大，"
                f"上限是 {max_files} 个源码文件，当前是 {len(source_files)} 个。"
                "请收敛到反例动作给出的最小源码边界。"
            )

        if avoid_paths and source_files:
            overlap = self._contract_suspect_overlap(source_files, avoid_paths)
            target_overlap = self._contract_suspect_overlap(source_files, target_paths) if target_paths else []
            if overlap and not target_overlap:
                return (
                    "CAR补丁意图提醒：当前源码 diff 回到了已撤回或陈旧的目标路径，"
                    f"路径：{', '.join(overlap[:5])}。"
                    "除非先用 focused evidence 证明它仍是正确目标，否则请转向当前意图目标。"
                )

        intended_paths = candidate_paths if preserve_candidate and candidate_paths else target_paths
        if require_target_touch and intended_paths and source_files:
            overlap = self._contract_suspect_overlap(source_files, intended_paths)
            if not overlap:
                return (
                    "CAR补丁意图提醒：你已经产生源码 diff，但没有触达结构化动作指定的源码目标。"
                    f"CAR目标：{', '.join(intended_paths[:5])}。"
                    "下一步请修改这些目标，或先用失败证据说明为什么必须重新定位。"
                )

        if (
            source_files
            and (
                "RUN_FOCUSED_VALIDATION" in directives
                or preserve_candidate
                or bool(requirements.get("fresh_source_diff", True))
            )
            and self._looks_like_write_command(command)
            and not self._has_validation_after_last_write()
        ):
            focused_hint = ""
            if candidate_paths:
                focused_hint = f"候选文件：{', '.join(candidate_paths[:4])}。"
            return (
                "CAR补丁意图提醒：已经产生源码 diff，但还没有在本次写入之后运行 focused 验证。"
                f"{focused_hint}"
                "下一步必须运行失败用例、py_compile 或项目的 focused test；"
                "验证失败时继续最小精修候选源码，不要直接收尾或继续只读浏览。"
            )

        if selected_action in {"LOCAL_REPAIR", "REPAIR_LOCAL"} and source_files and not target_paths and not candidate_paths:
            return (
                "CAR补丁意图提醒：当前是局部修复动作，但没有明确记录源码目标。"
                "请先基于 located_files 或失败证据确认一个规范源码目标，再继续扩大修改。"
            )
        return ""

    def _has_validation_after_last_write(self) -> bool:
        """Return whether the command history validates after the latest write."""

        last_write_index = -1
        last_write_command = ""
        for index, item in enumerate(getattr(self, "history", []) or []):
            command = str(dict(item).get("command", "") or "")
            if self._looks_like_write_command(command):
                last_write_index = index
                last_write_command = command
        if last_write_index < 0:
            return False
        if self._looks_like_validation_command(last_write_command):
            return True
        for item in list(getattr(self, "history", []) or [])[last_write_index + 1 :]:
            command = str(dict(item).get("command", "") or "")
            if self._looks_like_validation_command(command):
                return True
        return False

    def _parc_patch_contract_feedback(
        self,
        *,
        changed_files: list[str],
        classes: dict[str, list[str]],
        source_files: list[str],
        test_files: list[str],
        generated_files: list[str],
    ) -> str:
        contract = dict(getattr(self, "_parc_patch_contract", {}) or {})
        if not contract:
            return ""
        forbidden = {
            str(item).strip().lower().replace("_", "-")
            for item in list(contract.get("forbidden_path_classes", []) or [])
            if str(item).strip()
        }
        cleanup_paths: list[str] = []
        if "test" in forbidden and test_files:
            cleanup_paths.extend(test_files)
        if "generated" in forbidden and generated_files:
            cleanup_paths.extend(generated_files)
        if cleanup_paths:
            self._cleanup_non_source_recovery_edits(cleanup_paths)
            if source_files:
                return (
                    "PARC阶段边界契约失败：本次写入同时包含源码改动和契约禁止的测试/生成文件改动，"
                    f"已撤销禁止路径：{', '.join(cleanup_paths[:8])}。"
                    "保留源码候选后，下一步必须运行 focused validation；"
                    "如果验证失败，请继续最小修正源码，不要再次修改测试或生成文件。"
                )
            return (
                "PARC阶段边界契约失败：本次写入只修改了契约禁止的测试/生成文件，"
                f"已撤销：{', '.join(cleanup_paths[:8])}。"
                "下一步必须在契约 suspect_paths 指向的源码边界上产生 fresh source diff。"
            )

        require_fresh_source = bool(contract.get("require_fresh_source_diff", True))
        if require_fresh_source and changed_files and not source_files:
            return (
                "PARC阶段边界契约提醒：当前 diff 还没有源码文件。"
                "请把恢复动作落到 suspect_paths 对应的规范源码边界，而不是只改配置、文档或其他旁路文件。"
            )

        suspect_paths = [
            normalize_repo_path(str(item))
            for item in list(contract.get("suspect_paths", []) or [])
            if normalize_repo_path(str(item))
        ]
        if bool(contract.get("require_suspect_touch", False)) and source_files and suspect_paths:
            overlap = self._contract_suspect_overlap(source_files, suspect_paths)
            if not overlap:
                return (
                    "PARC阶段边界契约提醒：你已经产生源码 diff，但没有触达当前跨阶段污染边界的 suspect_paths。"
                    f"契约 suspect_paths：{', '.join(suspect_paths[:5])}。"
                    "下一步请修改这些源码边界，或先用失败证据说明为什么必须刷新定位。"
                )
        max_fresh_source_files = int(contract.get("max_fresh_source_files", 3) or 3)
        if max_fresh_source_files > 0 and len(source_files) > max_fresh_source_files:
            return (
                "PARC阶段边界契约提醒：当前 fresh source diff 触达的源码文件过多，"
                f"上限是 {max_fresh_source_files} 个，当前是 {len(source_files)} 个。"
                "请收敛到最小必要的 suspect source boundary；不要把恢复变成大范围重写。"
            )
        return ""

    @staticmethod
    def _contract_suspect_overlap(source_files: list[str], suspect_paths: list[str]) -> list[str]:
        overlap: list[str] = []
        for source in source_files:
            source_path = normalize_repo_path(source)
            for suspect in suspect_paths:
                suspect_path = normalize_repo_path(suspect)
                if (
                    source_path == suspect_path
                    or source_path.endswith(f"/{suspect_path}")
                    or suspect_path.endswith(f"/{source_path}")
                ):
                    overlap.append(source_path)
                    break
        return sorted(dict.fromkeys(overlap))

    def _recovery_requires_fresh_source_diff(self) -> bool:
        intent = dict(getattr(self, "_car_patch_intent", {}) or {})
        if intent:
            requirements = dict(intent.get("requirements", {}) or {})
            return bool(
                intent.get("require_fresh_source_diff", requirements.get("fresh_source_diff", True))
            )
        contract = dict(getattr(self, "_parc_patch_contract", {}) or {})
        if contract:
            return bool(contract.get("require_fresh_source_diff", True))
        return False

    def _recovery_requires_focused_validation_after_edit(self, source_files: list[str]) -> bool:
        if not getattr(self, "_recovery_mode", False) or not source_files:
            return False

        intent = dict(getattr(self, "_car_patch_intent", {}) or {})
        if intent:
            requirements = dict(intent.get("requirements", {}) or {})
            directives = {
                str(item or "").strip().upper()
                for item in list(intent.get("directives", []) or [])
                if str(item or "").strip()
            }
            return bool(
                "RUN_FOCUSED_VALIDATION" in directives
                or intent.get("preserve_candidate_source", requirements.get("preserve_candidate_source", False))
                or intent.get("require_fresh_source_diff", requirements.get("fresh_source_diff", True))
            )

        contract = dict(getattr(self, "_parc_patch_contract", {}) or {})
        if contract:
            return bool(contract.get("require_focused_validation", False))
        if getattr(self, "_masguard_source_edit_contract", False):
            return True
        return False

    def _recovery_fresh_source_files_for_finalize(self, summary: dict[str, Any]) -> tuple[list[str], bool]:
        fresh_source_files = self._fresh_source_files_since_recovery_baseline()
        if fresh_source_files is not None:
            return (
                [
                    normalize_repo_path(str(path))
                    for path in list(fresh_source_files or [])
                    if normalize_repo_path(str(path))
                ],
                True,
            )
        classes = dict(summary.get("changed_file_classes", {}) or {})
        return (
            [
                normalize_repo_path(str(path))
                for path in list(classes.get("source_files", []) or [])
                if normalize_repo_path(str(path))
            ],
            False,
        )

    @staticmethod
    def _command_mentions_any_path(command: str, paths: list[str]) -> bool:
        text = str(command or "").replace("\\", "/")
        if not text.strip():
            return False
        for raw_path in paths:
            path = normalize_repo_path(str(raw_path))
            if not path:
                continue
            if path in text or f"a/{path}" in text or f"b/{path}" in text:
                return True
        return False

    def _has_validation_after_last_source_write(self, source_files: list[str]) -> bool:
        """Return whether focused validation ran after the latest relevant source write."""

        normalized_sources = [
            normalize_repo_path(str(path))
            for path in list(source_files or [])
            if normalize_repo_path(str(path))
        ]
        last_write_index = -1
        last_source_write_index = -1
        last_source_write_command = ""
        for index, item in enumerate(getattr(self, "history", []) or []):
            command = str(dict(item).get("command", "") or "")
            if not self._looks_like_write_command(command):
                continue
            last_write_index = index
            if not normalized_sources or self._command_mentions_any_path(command, normalized_sources):
                last_source_write_index = index
                last_source_write_command = command
        if last_source_write_index < 0:
            last_source_write_index = last_write_index
            if last_write_index >= 0:
                last_source_write_command = str(
                    dict(list(getattr(self, "history", []) or [])[last_write_index]).get("command", "") or ""
                )
        if last_source_write_index < 0:
            return False
        if self._looks_like_validation_command(last_source_write_command):
            return True
        for item in list(getattr(self, "history", []) or [])[last_source_write_index + 1 :]:
            command = str(dict(item).get("command", "") or "")
            if self._looks_like_validation_command(command):
                return True
        return False

    def _recovery_finalize_block_reason(self, summary: dict[str, Any]) -> str:
        if not getattr(self, "_recovery_mode", False):
            return ""
        source_files, fresh_baseline_known = self._recovery_fresh_source_files_for_finalize(summary)
        if fresh_baseline_known and self._recovery_requires_fresh_source_diff() and not source_files:
            return "no_fresh_source_diff"
        if (
            self._recovery_requires_focused_validation_after_edit(source_files)
            and not self._has_validation_after_last_source_write(source_files)
        ):
            return "missing_focused_validation_after_source_edit"
        return ""

    def _finalize_effective_patch_result(self) -> dict[str, Any] | None:
        summary = self._patch_workspace_summary()
        changed_files = list(summary["changed_files"])
        classes = dict(summary["changed_file_classes"])
        if classes.get("generated_files"):
            self._cleanup_generated_artifacts(list(classes["generated_files"]))
            summary = self._patch_workspace_summary()
            changed_files = list(summary["changed_files"])
            classes = dict(summary["changed_file_classes"])
        effective_files = list(classes.get("effective_files", []))
        if not effective_files:
            return None
        if getattr(self, "_recovery_mode", False) and not classes.get("source_files"):
            return None
        if getattr(self, "_recovery_mode", False):
            source_patch_risk = dict(summary.get("source_patch_risk", {}) or {})
            if str(source_patch_risk.get("risk_level", "") or "") == "large_source_rewrite":
                files = [
                    str(path)
                    for path in list(source_patch_risk.get("large_rewrite_files", []) or [])
                    if str(path).strip()
                ]
                if files:
                    self._cleanup_non_source_recovery_edits(files)
                return None
            forbidden_feedback = self._recovery_forbidden_finalize_feedback(summary)
            if forbidden_feedback:
                self.add_message("user", forbidden_feedback)
                return None
            finalize_block_reason = self._recovery_finalize_block_reason(summary)
            if finalize_block_reason:
                finalize_feedback = self._recovery_finalize_block_feedback(
                    summary,
                    reason=finalize_block_reason,
                )
                if finalize_feedback:
                    self.add_message("user", finalize_feedback)
                return None

        quoted = " ".join(shlex.quote(path) for path in effective_files)
        # Do not stage files just to collect a patch. Reading both unstaged and
        # already-staged diffs is harmless and keeps the artifact faithful when
        # a model happened to run git add itself.
        diff_out = self.executor.execute(f"git diff -- {quoted}")
        cached_diff_out = self.executor.execute(f"git diff --cached -- {quoted}")
        out_text = "\n".join(
            text
            for text in (
                str(diff_out.get("output", "") or ""),
                str(cached_diff_out.get("output", "") or ""),
            )
            if text.strip()
        )
        if not out_text.strip():
            return None
        lines = out_text.split("\n")
        patch_lines = []
        started = False
        for line in lines:
            if started or line.strip().startswith("diff --git"):
                started = True
                patch_lines.append(line)
        auto_patch = "\n".join(patch_lines).strip() or out_text.strip()
        if not auto_patch:
            return None
        return {
            "patch": auto_patch,
            "success": True,
            "commands": self.history.copy(),
            "messages": self.messages.copy(),
            "patch_summary": summary,
            **self._result_metadata(
                patch_summary=summary,
                stop_reason="effective_patch_ready",
            ),
        }

    def _recovery_forbidden_finalize_feedback(self, summary: dict[str, Any]) -> str:
        contract = dict(getattr(self, "_parc_patch_contract", {}) or {})
        if not contract:
            return ""
        forbidden = {
            str(item).strip().lower().replace("_", "-")
            for item in list(contract.get("forbidden_path_classes", []) or [])
            if str(item).strip()
        }
        classes = dict(summary.get("changed_file_classes", {}) or {})
        cleanup_paths: list[str] = []
        if "test" in forbidden:
            cleanup_paths.extend(
                str(path)
                for path in list(classes.get("test_files", []) or [])
                if str(path).strip()
            )
        if "generated" in forbidden:
            cleanup_paths.extend(
                str(path)
                for path in list(classes.get("generated_files", []) or [])
                if str(path).strip()
            )
        if not cleanup_paths:
            return ""
        self._cleanup_non_source_recovery_edits(cleanup_paths)
        return (
            "PARC阶段边界契约阻止收尾：最终 diff 仍包含契约禁止的测试/生成文件，"
            f"已撤销：{', '.join(cleanup_paths[:8])}。"
            "请只保留规范源码 diff 并运行 focused validation 后再完成。"
        )

    def _force_edit_or_validation_analysis(self, command: str) -> dict[str, Any] | None:
        post_evidence_repair = self._post_evidence_repair_without_diff_analysis(command)
        if post_evidence_repair is not None:
            return post_evidence_repair
        if not self._is_read_only_probe_command(command):
            return None
        analysis = classify_probe_delta(command, self.history)
        strict_source_edit = self._strict_source_edit_without_diff_analysis(command, analysis)
        if strict_source_edit is not None:
            return strict_source_edit
        candidate_overread = self._candidate_preserving_overread_analysis(command, analysis)
        if candidate_overread is not None:
            return candidate_overread
        pending_validation = self._source_diff_pending_validation_analysis(command, analysis)
        if pending_validation is not None:
            return pending_validation
        synthetic_history = [
            *self.history,
            {
                "read_only_probe": True,
                "probe_signature": dict(analysis.get("probe_signature", {})),
                "inspected_regions_or_symbols": list(analysis.get("inspected_regions_or_symbols", []) or []),
                "evidence_incremental": bool(analysis.get("evidence_incremental")),
            },
        ]
        focused_summary = focused_readonly_probe_summary(synthetic_history)
        focused_limit = (
            self.RECOVERY_FOCUSED_READONLY_STREAK
            if getattr(self, "_recovery_mode", False)
            else self.MAX_FOCUSED_READONLY_STREAK
        )
        if (
            focused_summary["streak"] >= focused_limit
            and focused_summary["unique_path_count"] <= 2
            and focused_summary["dominant_count"] >= focused_limit - 1
            and focused_summary["dominant_paths"]
        ):
            enriched = dict(analysis)
            enriched["focused_paths"] = list(focused_summary["dominant_paths"][:2])
            enriched["focused_probe_streak"] = int(focused_summary["streak"])
            enriched["evidence_delta_kind"] = "focused_deep_read"
            enriched["evidence_incremental"] = False
            return enriched
        if analysis["evidence_incremental"]:
            return None
        limit = self.RECOVERY_READONLY_STREAK if getattr(self, "_recovery_mode", False) else self.MAX_READONLY_STREAK
        if no_progress_probe_streak(self.history) >= limit - 1:
            return analysis
        return None

    def _strict_source_edit_without_diff_analysis(
        self,
        command: str,
        analysis: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not getattr(self, "_recovery_mode", False):
            return None
        if not getattr(self, "_masguard_source_edit_contract", False):
            return None
        if self._looks_like_write_command(command) or self._looks_like_validation_command(command):
            return None
        fresh_source_files = self._fresh_source_files_since_recovery_baseline()
        if fresh_source_files is None:
            try:
                summary = self._patch_workspace_summary()
            except Exception:
                summary = {}
            classes = dict(summary.get("changed_file_classes", {}) or {})
            fresh_source_files = [
                normalize_repo_path(str(path))
                for path in list(classes.get("source_files", []) or [])
                if normalize_repo_path(str(path))
            ]
        if fresh_source_files:
            return None
        readonly_without_diff = 1
        for item in reversed(list(getattr(self, "history", []) or [])):
            prior_command = str(dict(item).get("command", "") or "")
            if not prior_command:
                continue
            if self._looks_like_write_command(prior_command) or self._looks_like_validation_command(prior_command):
                break
            if self._is_read_only_probe_command(prior_command):
                readonly_without_diff += 1
        if readonly_without_diff < 2:
            return None
        enriched = dict(analysis)
        enriched["focused_paths"] = list(getattr(self, "_masguard_source_edit_targets", []) or [])
        enriched["strict_source_edit_readonly_over_budget"] = True
        enriched["evidence_delta_kind"] = "strict_source_edit_readonly_over_budget"
        enriched["evidence_incremental"] = False
        return enriched

    def _post_evidence_repair_without_diff_analysis(self, command: str) -> dict[str, Any] | None:
        """Keep post-evidence LOCAL_REPAIR from spending replay on no-diff work.

        After CAR has selected `LOCAL_REPAIR` with a post-evidence source-repair
        intent, running validation before any source diff only confirms the old
        failure and consumes the bounded replay.  This gate is method-level: it
        uses the CAR intent contract, not instance names or file-specific rules.
        """

        if not getattr(self, "_recovery_mode", False):
            return None
        intent = dict(getattr(self, "_car_patch_intent", {}) or {})
        if not intent:
            return None
        if self._looks_like_write_command(command):
            return None
        requirements = dict(intent.get("requirements", {}) or {})
        require_fresh_source = bool(
            intent.get("require_fresh_source_diff", requirements.get("fresh_source_diff", True))
        )
        if not require_fresh_source:
            return None
        directives = {
            str(item or "").strip().upper()
            for item in list(intent.get("directives", []) or [])
            if str(item or "").strip()
        }
        if not (
            "POST_EVIDENCE_SOURCE_REPAIR" in directives
            or "DO_NOT_SPEND_REPLAY_ON_READONLY_DIAGNOSIS" in directives
        ):
            return None
        fresh_source_files = self._fresh_source_files_since_recovery_baseline()
        if fresh_source_files is None:
            try:
                summary = self._patch_workspace_summary()
            except Exception:
                summary = {}
            classes = dict(summary.get("changed_file_classes", {}) or {})
            fresh_source_files = [
                normalize_repo_path(str(path))
                for path in list(classes.get("source_files", []) or [])
                if normalize_repo_path(str(path))
            ]
        if fresh_source_files:
            return None

        paths = dict(intent.get("paths", {}) or {})
        target_paths = [
            normalize_repo_path(str(item))
            for item in list(intent.get("target_paths", paths.get("target_paths", [])) or [])
            if normalize_repo_path(str(item))
        ]
        candidate_paths = [
            normalize_repo_path(str(item))
            for item in list(intent.get("candidate_source_paths", paths.get("candidate_source_paths", [])) or [])
            if normalize_repo_path(str(item))
        ]
        focused_paths = (candidate_paths or target_paths)[:3]
        if self._looks_like_validation_command(command):
            return {
                "focused_paths": focused_paths,
                "post_evidence_validation_before_source_diff": True,
                "evidence_delta_kind": "post_evidence_validation_before_source_diff",
                "evidence_incremental": False,
            }
        if not self._is_read_only_probe_command(command):
            return None

        readonly_without_diff = 1
        for item in reversed(list(getattr(self, "history", []) or [])):
            prior_command = str(dict(item).get("command", "") or "")
            if not prior_command:
                continue
            if self._looks_like_write_command(prior_command) or self._looks_like_validation_command(prior_command):
                break
            if self._is_read_only_probe_command(prior_command):
                readonly_without_diff += 1
        if readonly_without_diff < 2:
            return None
        analysis = classify_probe_delta(command, self.history)
        enriched = dict(analysis)
        enriched["focused_paths"] = focused_paths
        enriched["post_evidence_readonly_over_budget"] = True
        enriched["evidence_delta_kind"] = "post_evidence_readonly_over_budget"
        enriched["evidence_incremental"] = False
        return enriched

    def _source_diff_pending_validation_analysis(
        self,
        command: str,
        analysis: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Stop browsing once a recovery replay has an unvalidated source diff."""

        if not getattr(self, "_recovery_mode", False):
            return None
        if not self.history:
            return None
        if self._has_validation_after_last_write():
            return None
        if not any(
            self._looks_like_write_command(str(dict(item).get("command", "") or ""))
            for item in list(getattr(self, "history", []) or [])
        ):
            return None
        try:
            summary = self._patch_workspace_summary()
        except Exception:
            return None
        classes = dict(summary.get("changed_file_classes", {}) or {})
        fresh_source_files = self._fresh_source_files_since_recovery_baseline()
        if fresh_source_files is None:
            source_files = [
                normalize_repo_path(str(path))
                for path in list(classes.get("source_files", []) or [])
                if normalize_repo_path(str(path))
            ]
        else:
            source_files = fresh_source_files
        if not source_files:
            return None
        enriched = dict(analysis)
        enriched["focused_paths"] = source_files[:3]
        enriched["source_diff_pending_validation"] = True
        enriched["evidence_delta_kind"] = "source_diff_pending_validation"
        enriched["evidence_incremental"] = False
        return enriched

    def _candidate_preserving_overread_analysis(
        self,
        command: str,
        analysis: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Tighten the read-only budget once a source candidate already exists.

        Candidate-preserving recovery is not a fresh localization phase.  The
        model already has candidate files plus failing-test evidence, so repeated
        read-only probing usually means the typed recovery action is not being
        converted into a refinement.  This generic gate keeps the method focused
        on edit-or-validate behavior without encoding instance-specific rules.
        """

        intent = dict(getattr(self, "_car_patch_intent", {}) or {})
        if not intent:
            return None
        requirements = dict(intent.get("requirements", {}) or {})
        paths = dict(intent.get("paths", {}) or {})
        preserve_candidate = bool(
            intent.get("preserve_candidate_source", requirements.get("preserve_candidate_source", False))
        )
        candidate_paths = [
            normalize_repo_path(str(item))
            for item in list(intent.get("candidate_source_paths", paths.get("candidate_source_paths", [])) or [])
            if normalize_repo_path(str(item))
        ]
        if not preserve_candidate and not candidate_paths:
            return None

        readonly_since_last_decision = 1
        for item in reversed(list(getattr(self, "history", []) or [])):
            prior_command = str(dict(item).get("command", "") or "")
            if not prior_command:
                continue
            if self._looks_like_write_command(prior_command) or self._looks_like_validation_command(prior_command):
                break
            if self._is_read_only_probe_command(prior_command):
                readonly_since_last_decision += 1
        if readonly_since_last_decision < 2:
            return None

        enriched = dict(analysis)
        enriched["focused_paths"] = candidate_paths[:3]
        enriched["candidate_preserving_overread"] = True
        enriched["evidence_delta_kind"] = "candidate_preserving_overread"
        enriched["evidence_incremental"] = False
        return enriched

    def _should_force_edit_or_validation(self, command: str) -> bool:
        if self._force_edit_or_validation_analysis(command) is not None:
            return True
        legacy_readonly = [
            item
            for item in getattr(self, "history", [])
            if isinstance(item, dict)
            and "read_only_probe" not in item
            and self._is_read_only_probe_command(str(item.get("command", "")))
        ]
        limit = self.RECOVERY_READONLY_STREAK if getattr(self, "_recovery_mode", False) else self.MAX_READONLY_STREAK
        return self._is_read_only_probe_command(command) and len(legacy_readonly) >= limit

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

    def _target_legitimacy(self, classes: dict[str, list[str]], changed_files: list[str]) -> str:
        source_files = list(classes.get("source_files", []) or [])
        test_files = list(classes.get("test_files", []) or [])
        generated_files = list(classes.get("generated_files", []) or [])
        other_files = list(classes.get("other_files", []) or [])
        if not changed_files:
            return "no_diff"
        if source_files and not test_files and not generated_files and not other_files:
            return "source_only"
        if source_files and (test_files or other_files):
            return "source_mixed"
        if test_files and not source_files:
            return "tests_only"
        if generated_files and not source_files:
            return "generated_only"
        return "non_source_only"

    def _result_metadata(self, *, patch_summary: dict[str, Any], stop_reason: str) -> dict[str, Any]:
        return {
            "stop_reason": stop_reason,
            "evidence_delta_kind": str(self.history[-1].get("evidence_delta_kind", "") or "") if self.history else "",
            "probe_signature": dict(self.history[-1].get("probe_signature", {})) if self.history else {},
            "inspected_regions_or_symbols": summarize_regions_or_symbols(self.history),
            "selected_target_candidates": self._selected_target_candidates(patch_summary),
            "target_legitimacy": str(patch_summary.get("target_legitimacy", "") or ""),
            "source_patch_risk": dict(patch_summary.get("source_patch_risk", {}) or {}),
        }

    def _selected_target_candidates(self, patch_summary: dict[str, Any]) -> list[str]:
        candidates = [
            str(path).strip()
            for path in list(patch_summary.get("changed_files", []) or [])
            if str(path).strip()
        ]
        if candidates:
            return candidates
        return summarize_probe_paths(self.history)

    def _terminal_stop_reason(self, patch_summary: dict[str, Any]) -> str:
        limit = self.RECOVERY_READONLY_STREAK if getattr(self, "_recovery_mode", False) else self.MAX_READONLY_STREAK
        if no_progress_probe_streak(self.history) >= limit:
            return "true_no_progress"
        incremental = any(bool(item.get("evidence_incremental")) for item in self.history)
        if incremental:
            return "budget_exhausted_with_partial_evidence"
        if str(patch_summary.get("failure_mode", "") or "") == "no_effective_patch":
            return "true_no_progress"
        return "budget_exhausted_with_partial_evidence"

    def _is_recovery_problem(self, problem_statement: str | None) -> bool:
        text = str(problem_statement or "")
        return (
            "[RECOVERY CONTRACT]" in text
            or "[PARC PATCH CONTRACT]" in text
            or "[CAR PATCH INTENT]" in text
            or "[MASGUARD SOURCE EDIT CONTRACT]" in text
            or "失败后的恢复" in text
            or "恢复场景" in text
        )

    def _should_use_recovery_mode(self, problem_statement: str | None) -> bool:
        return bool(getattr(self, "_force_recovery_mode", False) or self._is_recovery_problem(problem_statement))

    @staticmethod
    def _extract_masguard_source_edit_targets(problem_statement: str | None) -> list[str]:
        text = str(problem_statement or "")
        targets: list[str] = []

        def add(value: str) -> None:
            path = normalize_repo_path(value)
            if path and path not in targets:
                targets.append(path)

        primary = re.search(r'"primary_source_target"\s*:\s*"([^"]+)"', text)
        if primary:
            add(primary.group(1))
        allowed = re.search(r'"allowed_source_files"\s*:\s*\[(.*?)\]', text)
        if allowed:
            for match in re.findall(r'"([^"]+)"', allowed.group(1)):
                add(match)
        return targets

    def _is_read_only_probe_command(self, command: str) -> bool:
        return looks_like_readonly_probe_command(command)

    def _looks_like_validation_command(self, command: str) -> bool:
        return looks_like_validation_command(command)
