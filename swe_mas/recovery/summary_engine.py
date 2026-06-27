"""基于状态树的自适应摘要生成（支持LLM压缩，预留进化接口）

设计目标：
- 根据 (角色, 故障类型, 资源条件) 自适应选择摘要策略；
- 对内部认知故障（loop/hallucination/misalignment/injected），从“最后健康意图”作为锚点切断错误分支；
- 生成面向“接替 Agent”的可执行交接摘要，而不仅是回顾性描述。
"""
from __future__ import annotations
from typing import Any
import re
from swe_mas.memory import MemoryRecorder
from swe_mas.recovery.models import RecoveryRequest, RecoveryStrategy, ResourceLevel, Granularity
from swe_mas.perception.models import FaultType
from swe_mas.recovery.interfaces import LLMClientProtocol
from swe_mas.evolution.summary_policy import SummaryPolicy


class SummaryEngine:
    def __init__(
        self,
        recorder: MemoryRecorder,
        llm: LLMClientProtocol | None = None,
        role_focus: dict[str, list[str]] | None = None,
        summary_policy: SummaryPolicy | None = None,
    ):
        self.recorder = recorder
        self.llm = llm
        self.role_focus = role_focus or {
            "planner": ["intent", "constraints", "goals"],
            "implementer": ["intent", "code", "diff", "state"],
            "verifier": ["tests", "logs", "state"],
            "analyzer": ["intent", "context"],
        }
        self.summary_policy = summary_policy

    def _is_internal_fault(self, fault_type: FaultType | str | None) -> bool:
        """统一内部故障判定：包含 hallucination / misalignment / loop / injected 等语义故障。

        fault_type 可能来自枚举或字符串注入，这里做宽松兼容。
        """
        if fault_type is None:
            return False
        if isinstance(fault_type, FaultType):
            name = fault_type.name
            value = fault_type.value
        else:
            name = str(fault_type).upper()
            value = str(fault_type).lower()
        cognitive_names = {"HALLUCINATION", "MISALIGNMENT", "LOOP", "INJECTED"}
        cognitive_values = {"hallucination", "misalignment", "loop", "injected"}
        return name in cognitive_names or value in cognitive_values

    def decide_strategy(self, request: RecoveryRequest) -> RecoveryStrategy:
        if self.summary_policy:
            strategy_id, strategy = self.summary_policy.select_strategy(request)
            if hasattr(request, "details") and isinstance(request.details, dict):
                request.details["summary_strategy_id"] = strategy_id
                request.details["summary_bucket_id"] = self.summary_policy.bucket_id(request)
            return strategy
        low_token = request.resource.token_left is not None and request.resource.token_left < 800
        if request.resource.bandwidth == ResourceLevel.LOW or low_token:
            return RecoveryStrategy(
                granularity=Granularity.CONCISE,
                focus_areas=["intent", "last_state"],
                discard_recent_branch=False,
                reuse_healthy_intent=True,
                max_history=20,
            )
        if self._is_internal_fault(request.fault_type):
            return RecoveryStrategy(
                granularity=Granularity.VERBOSE,
                focus_areas=["intent", "actions", "state", "branches"],
                discard_recent_branch=True,
                reuse_healthy_intent=True,
                max_history=80,
            )
        if request.fault_type in {FaultType.NETWORK_ERROR, FaultType.EXTERNAL}:
            return RecoveryStrategy(
                granularity=Granularity.STANDARD,
                focus_areas=["intent", "state", "env"],
                discard_recent_branch=False,
                reuse_healthy_intent=True,
                max_history=40,
            )
        focus = ["intent", "actions", "state"]
        focus.extend(self.role_focus.get(request.target_role, []))
        return RecoveryStrategy(
            granularity=Granularity.STANDARD,
            focus_areas=focus,
            discard_recent_branch=False,
            reuse_healthy_intent=True,
            max_history=40,
        )

    def _collect_nodes(self, request: RecoveryRequest, strategy: RecoveryStrategy) -> list[dict[str, Any]]:
        nodes = self.recorder.get_nodes_filtered(
            session_id=request.session_id,
            node_types=["THOUGHT", "PLAN", "ACTION", "OBSERVATION", "CHECKPOINT", "MESSAGE"],
            limit=strategy.max_history,
        )
        fault_ids = set()
        if request.details and request.details.get("fault_node_ids"):
            fault_ids = set(request.details.get("fault_node_ids") or [])
        if fault_ids:
            # 确保故障节点纳入上下文
            all_nodes = self.recorder.get_nodes(request.session_id)
            id_map = {n["node_id"]: n for n in all_nodes}
            for fid in fault_ids:
                if fid in id_map and id_map[fid] not in nodes:
                    nodes.append(id_map[fid])
            # 按时间排序
            nodes = sorted(nodes, key=lambda n: n.get("timestamp", 0))
        return nodes

    def _last_of_type(self, nodes: list[dict], types: set[str]):
        for n in reversed(nodes):
            if n.get("node_type") in types:
                return n
        return None

    def _last_healthy(self, nodes: list[dict]) -> dict | None:
        """返回“最后健康意图”的节点。

        优先 THOUGHT/PLAN（认知/规划），退而求其次任何 VALID 节点。
        """
        # 1) 优先找健康的 THOUGHT/PLAN
        for n in reversed(nodes):
            if n.get("validity", "VALID").upper() != "INVALID" and n.get("node_type") in {"THOUGHT", "PLAN"}:
                return n
        # 2) 找不到则退回到最后一个 VALID 节点
        for n in reversed(nodes):
            if n.get("validity", "VALID").upper() != "INVALID":
                return n
        return None

    def _split_healthy_faulty(self, nodes: list[dict], request: RecoveryRequest) -> tuple[list[dict], list[dict]]:
        """按本次 fault 的 node_ids 优先切分，其次按 validity 切分（按时间顺序）。"""
        if not nodes:
            return [], []
        fault_ids: set[str] = set()
        if request.details and isinstance(request.details.get("fault_node_ids"), list):
            fault_ids = {x for x in request.details["fault_node_ids"] if isinstance(x, str)}

        if fault_ids:
            first_fault_idx = None
            for i, n in enumerate(nodes):
                if n.get("node_id") in fault_ids:
                    first_fault_idx = i
                    break
            if first_fault_idx is not None:
                return nodes[:first_fault_idx], nodes[first_fault_idx:]

        first_invalid = None
        for i, n in enumerate(nodes):
            if n.get("validity") == "INVALID":
                first_invalid = i
                break
        if first_invalid is None:
            return nodes, []
        return nodes[:first_invalid], nodes[first_invalid:]

    def _history_segments(self, nodes: list[dict], max_items: int = 5) -> list[str]:
        segs = []
        for n in nodes[-max_items:]:
            segs.append(f"- {n.get('node_type')} {n.get('role')}: {n.get('content','')[:300]}")
        return segs

    def _format_nodes_text(self, nodes: list[dict]) -> str:
        """生成可阅读的上下文文本，附带必要元信息，限制长度以避免爆炸。"""
        parts = []
        for n in nodes:
            line = (
                f"[{n.get('node_id')}] {n.get('node_type')}/{n.get('step_type','')}"
                f" role={n.get('role','')} ts={n.get('timestamp','')}"
                f" validity={n.get('validity','')}"
            )
            content = (n.get("content") or "").strip()
            parts.append(line + "\n" + content)
        text = "\n\n".join(parts)
        if len(text) > 10000:
            text = text[:10000] + "\n...[truncated]"
        return text

    def _nearest_prev_snapshot(self, nodes: list[dict], node_id: str) -> str:
        """向前搜索最近的快照，用于 THOUGHT/PLAN 未记录 env_snapshot_hash 的场景。"""
        if not node_id:
            return ""
        idx = None
        for i, n in enumerate(nodes):
            if n.get("node_id") == node_id:
                idx = i
                break
        if idx is None:
            return ""
        for j in range(idx, -1, -1):
            snap = nodes[j].get("env_snapshot_hash")
            if snap:
                return snap
        return ""

    def _default_summary(self, request: RecoveryRequest, strategy: RecoveryStrategy, nodes: list[dict]) -> dict[str, Any]:
        last_intent = self._last_of_type(nodes, {"THOUGHT", "PLAN"})
        last_action = self._last_of_type(nodes, {"ACTION"})
        last_obs = self._last_of_type(nodes, {"OBSERVATION"})
        last_healthy = self._last_healthy(nodes)
        healthy_nodes, faulty_nodes = self._split_healthy_faulty(nodes, request)
        healthy_snapshot = ""
        if last_healthy:
            healthy_snapshot = last_healthy.get("env_snapshot_hash") or self._nearest_prev_snapshot(nodes, last_healthy.get("node_id", ""))

        return {
            "role": request.target_role,
            "fault_type": str(request.fault_type),
            "granularity": strategy.granularity.value,
            "healthy_anchor": {
                "node_id": (last_healthy or {}).get("node_id"),
                "content": (last_healthy or {}).get("content", "")[:300],
                "snapshot": healthy_snapshot,
            },
            "last_intent": {
                "node_id": (last_intent or {}).get("node_id"),
                "content": (last_intent or {}).get("content", "")[:300],
            },
            "last_action": {
                "node_id": (last_action or {}).get("node_id"),
                "content": (last_action or {}).get("content", "")[:200],
            },
            "last_observation": {
                "node_id": (last_obs or {}).get("node_id"),
                "content": (last_obs or {}).get("content", "")[:200],
                "snapshot": (last_obs or {}).get("env_snapshot_hash"),
            },
            "recent": self._history_segments(nodes, max_items=8 if strategy.granularity == Granularity.VERBOSE else 5),
            "healthy_path": self._history_segments(healthy_nodes, max_items=6) if healthy_nodes else [],
            "faulty_path": self._history_segments(faulty_nodes, max_items=6) if faulty_nodes else [],
            "hint": "请验证最后步骤意图是否合理，并从 healthy_anchor 继续。",
        }

    def _format_summary_text(self, struct: dict[str, Any]) -> str:
        parts = [
            f"[任务摘要]\n- 角色: {struct.get('role')}\n- 故障类型: {struct.get('fault_type')}\n- 粒度: {struct.get('granularity')}",
            "[健康锚点]\n"
            f"- node: {struct.get('healthy_anchor', {}).get('node_id')}\n"
            f"- 摘要: {struct.get('healthy_anchor', {}).get('content','')}",
            "[最近意图/动作/观测]\n"
            f"- 意图: {struct.get('last_intent', {}).get('content','')}\n"
            f"- 动作: {struct.get('last_action', {}).get('content','')}\n"
            f"- 观测: {struct.get('last_observation', {}).get('content','')}",
        ]
        if struct.get("recent"):
            parts.append("[近期轨迹]\n" + "\n".join(struct["recent"]))
        if struct.get("faulty_path"):
            parts.append("[故障片段]\n" + "\n".join(struct["faulty_path"]))
        parts.append("[提示]\n" + struct.get("hint", ""))
        return "\n".join(parts)

    def _llm_summarize(self, request: RecoveryRequest, strategy: RecoveryStrategy, nodes: list[dict]) -> str:
        """调用 LLM 生成“接替 Agent 可执行”的交接摘要。"""
        if not self.llm:
            return ""
        depth_limit = 3 if strategy.granularity == Granularity.CONCISE else 6 if strategy.granularity == Granularity.STANDARD else 12
        healthy_nodes, faulty_nodes = self._split_healthy_faulty(nodes, request)
        # 选取上下文：优先给出锚点附近和故障片段
        context_nodes: list[dict]
        if self._is_internal_fault(request.fault_type):
            context_nodes = (healthy_nodes + faulty_nodes)[-depth_limit:]
        else:
            context_nodes = nodes[-depth_limit:]
        last_healthy = self._last_healthy(nodes)
        known_healthy_id = (last_healthy or {}).get("node_id") or ""
        known_healthy_snapshot = ""
        if known_healthy_id:
            known_healthy_snapshot = (last_healthy or {}).get("env_snapshot_hash") or self._nearest_prev_snapshot(nodes, known_healthy_id)

        lines = []
        for n in context_nodes:
            snap = n.get("env_snapshot_hash") or ""
            node_id = n.get("node_id") or "unknown"
            step_type = n.get("step_type") or ""
            lines.append(
                f"- id={node_id} | {n.get('node_type')}/{step_type} {n.get('role')}: {n.get('content','')[:300]} | "
                f"snapshot={snap} | validity={n.get('validity','VALID')}"
            )
        # 对内部故障，明确健康路径 vs 故障片段，让LLM做“诊断+纠偏”摘要
        extra_instr = ""
        if self._is_internal_fault(request.fault_type):
            extra_instr = (
                "\nThe trajectory contains a healthy prefix (before the first INVALID node) "
                "and a faulty segment after that. Identify where the agent's intent diverged from "
                "the original goal, highlight the wrong assumptions, and suggest how a new agent "
                "should resume from the last healthy state without repeating the same mistake."
            )
            # 对“注入故障”但没有真实故障片段的实验场景，明确禁止编造失败现象
            fault_upper = str(request.fault_type).upper()
            if "INJECTED" in fault_upper and not faulty_nodes:
                extra_instr += (
                    "\nNote: this fault was injected for an experiment and there may be no real post-divergence "
                    "failure segment in the trajectory. Do NOT invent test failures or stack traces. "
                    "Only cite evidence explicitly present in the trajectory; otherwise say 'unknown'."
                )
        prompt = (
            "You are a recovery summarizer for a multi-agent SWE system. "
            "Given the trajectory and detected fault, produce a concise handoff brief for the next agent.\n"
            f"Target role: {request.target_role}\nFault: {request.fault_type}\n"
            f"Granularity: {strategy.granularity.value}\n"
            f"Known healthy anchor (computed): node_id={known_healthy_id or 'unknown'} snapshot={known_healthy_snapshot or 'unknown'}\n"
            "Focus on: " + ", ".join(strategy.focus_areas) + extra_instr + "\n"
            "Trajectory (most recent first):\n" + "\n".join(reversed(lines)) + "\n"
            "Return strictly in this template:\n"
            "[任务摘要]\n- 故障类型: ...\n- 当前快照: <snapshot or unknown>\n"
            "[健康锚点]\n- node_id: ...\n- 摘要: ...\n"
            "[错误意图/行动]\n- 最近错误点: ...\n- 证据: ...\n"
            "[接任建议]\n"
            "1. 你是接替前任工作的 Agent，请基于健康锚点继续完成整个修复任务，而不是只做一步。\n"
            "2. 需要重新验证的关键假设: ...\n"
            "3. 列出接下来 3-5 个具体可执行步骤（按顺序给出，每步一句话，供后续 Agent 直接遵循）。\n"
            "4. 指出必须避免重复的错误路径或假设: ...\n"
            "Rules:\n"
            "- When you output node_id, it MUST be exactly copied from one of the trajectory lines (the value after 'id='). If unknown, output 'unknown'.\n"
            "- Prefer the computed healthy anchor node_id above when it is available.\n"
            "Keep it short, structured, and actionable."
        )
        messages = [
            {"role": "system", "content": "Generate recovery brief."},
            {"role": "user", "content": prompt},
        ]
        try:
            return self.llm.query(messages)["content"]
        except Exception:
            return ""

    def _summarize_external_fault(self, request: RecoveryRequest, strategy: RecoveryStrategy, phase_snapshot: dict | None) -> dict[str, Any]:
        """外部故障恢复：需要完整的累积上下文
        
        核心思想：Agent在一次run中会累积上下文（self.messages），外部故障中断后需要恢复这些上下文
        """
        # 优先使用Phase快照的node_ids快速获取节点
        nodes = []
        if phase_snapshot and phase_snapshot.get("node_ids"):
            node_ids = phase_snapshot["node_ids"]
            all_nodes = self.recorder.get_nodes(request.session_id)
            node_map = {n["node_id"]: n for n in all_nodes}
            nodes = [node_map[nid] for nid in node_ids if nid in node_map]
            logger = getattr(self, '_logger', None)
            if logger:
                logger.debug(f"[Recovery] 从Phase快照快速获取{len(nodes)}个节点")
        
        if not nodes:
            # Fallback: 按role和phase查询
            nodes = self.recorder.get_nodes(request.session_id, role=request.target_role, phase=request.target_role)
        
        if not nodes:
            nodes = self._collect_nodes(request, strategy)
        
        # 提取初始输入和累积上下文
        initial_inputs = {}
        if phase_snapshot:
            initial_inputs = phase_snapshot.get("inputs", {})
        elif nodes:
            # Fallback: 从第一个节点的payload中尝试恢复输入
            first_node = nodes[0]
            payload = first_node.get("payload", {})
            if "inputs" in payload:
                initial_inputs = payload.get("inputs", {})
        
        accumulated_context = self._extract_accumulated_context(nodes)
        last_obs = self._last_of_type(nodes, {"OBSERVATION"})
        last_action = self._last_of_type(nodes, {"ACTION"})
        last_edit_action = None
        for n in reversed(nodes):
            if n.get("node_type") != "ACTION":
                continue
            payload = n.get("payload") or {}
            cmd = payload.get("command") or n.get("content", "")
            if self._looks_like_write_command(cmd):
                last_edit_action = n
                break
        last_test_status = ""
        last_test_command = ""
        for n in reversed(nodes):
            if n.get("node_type") != "OBSERVATION":
                continue
            payload = n.get("payload") or {}
            if payload.get("step_type") == "test_result" or payload.get("test_status"):
                last_test_status = payload.get("test_status") or ""
                last_test_command = payload.get("command") or ""
                break
        last_snapshot = ""
        if last_obs:
            last_snapshot = last_obs.get("env_snapshot_hash", "")
        elif phase_snapshot:
            last_snapshot = phase_snapshot.get("env_snapshot_end", "") or phase_snapshot.get("env_snapshot_start", "")
        last_iteration = max((n.get("iteration", 0) for n in nodes), default=0)
        files_touched = self._collect_files_touched(nodes)
        planned_files, planned_tests, patch_strategy = self._collect_structured_hints(nodes)
        effective_iters, edit_iters = self._effective_iteration_counts(accumulated_context)
        last_completed_iter = None
        for ctx in reversed(accumulated_context):
            if (ctx.get("action") or "").strip() or (ctx.get("observation") or "").strip():
                last_completed_iter = ctx.get("iteration")
                break
        if last_completed_iter is None:
            last_completed_iter = last_iteration
        
        # 生成简洁的恢复摘要（中文）
        summary_text = self._format_external_fault_summary(
            role=request.target_role,
            initial_inputs=initial_inputs,
            accumulated_context=accumulated_context,
            fault_message=request.details.get("message", "") if request.details else "",
            granularity=strategy.granularity,
            last_snapshot=last_snapshot,
            last_iteration=last_iteration,
            files_touched=files_touched,
            planned_files=planned_files,
            planned_tests=planned_tests,
            patch_strategy=patch_strategy,
            effective_iters=effective_iters,
            edit_iters=edit_iters,
            last_completed_iter=last_completed_iter,
            last_action=last_action,
            last_edit_action=last_edit_action,
            last_test_status=last_test_status,
            last_test_command=last_test_command,
            last_observation=last_obs,
        )
        
        # 提取messages和history用于resume
        resume_state = self._extract_resume_state(nodes)
        
        return {
            "strategy": strategy,
            "summary": summary_text,
            "summary_struct": {
                "fault_type": "external",
                "role": request.target_role,
                "initial_inputs": initial_inputs,
                "accumulated_context": accumulated_context,
                "last_snapshot": last_snapshot,
                "last_iteration": last_iteration,
                "files_touched": files_touched,
                "planned_files": planned_files,
                "planned_tests": planned_tests,
                "patch_strategy": patch_strategy,
                "effective_iterations": effective_iters,
                "edit_iterations": edit_iters,
                "last_completed_iteration": last_completed_iter,
                "last_edit_command": (last_edit_action or {}).get("payload", {}).get("command") if last_edit_action else "",
                "last_test_status": last_test_status,
                "last_test_command": last_test_command,
                "granularity": strategy.granularity.value,
                "role_focus": list(strategy.focus_areas),
                "resource": {
                    "bandwidth": request.resource.bandwidth.value,
                    "token_left": request.resource.token_left,
                },
                "resume_state": resume_state,  # 新增：用于Agent resume
            },
            "trace": {
                "target_role": request.target_role,
                "fault_type": str(request.fault_type),
                "node_count": len(nodes),
                "context_iterations": len(accumulated_context),
                "summary_len": len(summary_text),
            },
            "last_snapshot": last_snapshot,
            "healthy_snapshot": last_snapshot,  # 外部故障时，最后一个快照就是健康快照
            "healthy_node_id": nodes[-1].get("node_id") if nodes else None,
        }

    def _summarize_internal_fault(self, request: RecoveryRequest, strategy: RecoveryStrategy, phase_snapshot: dict | None) -> dict[str, Any]:
        """内部故障恢复：诊断错误意图，从健康锚点重新开始
        
        核心思想：Agent陷入幻觉/循环，需要找到最后健康的节点，丢弃错误分支
        """
        nodes = self._collect_nodes(request, strategy)
        healthy_nodes, faulty_nodes = self._split_healthy_faulty(nodes, request)
        last_healthy = self._last_healthy(healthy_nodes)
        planned_files, planned_tests, patch_strategy = self._collect_structured_hints(nodes)
        
        # 找到健康Phase的输入（可能需要回到上游Agent）
        healthy_phase = None
        if last_healthy:
            healthy_phase_name = last_healthy.get("phase") or request.target_role
            healthy_phase = self.recorder.get_phase_snapshot(request.session_id, phase=healthy_phase_name, validity="VALID")
        
        if not healthy_phase and phase_snapshot:
            # 回退到当前Phase的初始状态
            healthy_phase = phase_snapshot
        
        # 如果仍然没有Phase快照，从健康节点构造最小上下文
        if not healthy_phase and last_healthy:
            healthy_phase = {
                "phase": last_healthy.get("phase", request.target_role),
                "role": last_healthy.get("role", request.target_role),
                "inputs": {},  # 无法恢复inputs，只能空着
                "env_snapshot_start": last_healthy.get("env_snapshot_hash", ""),
                "env_snapshot_end": last_healthy.get("env_snapshot_hash", ""),
            }
        
        # 诊断错误原因（使用LLM）
        diagnosis = self._diagnose_internal_fault(request, healthy_nodes, faulty_nodes, last_healthy)
        
        # 生成恢复摘要（中文，明确角色边界）
        summary_text = self._format_internal_fault_summary(
            role=request.target_role,
            healthy_phase=healthy_phase,
            diagnosis=diagnosis,
            last_healthy_node=last_healthy,
            planned_files=planned_files,
            planned_tests=planned_tests,
            patch_strategy=patch_strategy,
        )
        
        healthy_snapshot = ""
        if last_healthy:
            healthy_snapshot = last_healthy.get("env_snapshot_hash") or self._nearest_prev_snapshot(nodes, last_healthy.get("node_id", ""))
        elif healthy_phase:
            healthy_snapshot = healthy_phase.get("env_snapshot_end") or healthy_phase.get("env_snapshot_start", "")
        
        return {
            "strategy": strategy,
            "summary": summary_text,
            "summary_struct": {
                "fault_type": "internal",
                "role": request.target_role,
                "healthy_phase": healthy_phase,
                "diagnosis": diagnosis,
                "healthy_node": last_healthy,
                "planned_files": planned_files,
                "planned_tests": planned_tests,
                "patch_strategy": patch_strategy,
                "granularity": strategy.granularity.value,
                "role_focus": list(strategy.focus_areas),
                "resource": {
                    "bandwidth": request.resource.bandwidth.value,
                    "token_left": request.resource.token_left,
                },
            },
            "trace": {
                "target_role": request.target_role,
                "fault_type": str(request.fault_type),
                "healthy_count": len(healthy_nodes),
                "faulty_count": len(faulty_nodes),
                "summary_len": len(summary_text),
            },
            "last_snapshot": healthy_snapshot,
            "healthy_snapshot": healthy_snapshot,
            "healthy_node_id": (last_healthy or {}).get("node_id"),
        }

    def _extract_accumulated_context(self, nodes: list[dict]) -> list[dict]:
        """提取Agent在本次run中累积的上下文（迭代历史）
        
        优化：按iteration字段分组，而不是按THOUGHT切分
        """
        # 按iteration分组
        iteration_groups: dict[int, list[dict]] = {}
        for node in nodes:
            iter_num = node.get("iteration", 0)
            if iter_num not in iteration_groups:
                iteration_groups[iter_num] = []
            iteration_groups[iter_num].append(node)
        
        # 提取每次迭代的关键信息
        iterations = []
        for iter_num in sorted(iteration_groups.keys()):
            iter_nodes = iteration_groups[iter_num]
            thought_parts = []
            action_parts = []
            observation_parts = []
            
            for node in iter_nodes:
                node_type = node.get("node_type", "")
                content = (node.get("content") or "").strip()
                
                if node_type == "THOUGHT":
                    thought_parts.append(content)
                elif node_type == "ACTION":
                    action_parts.append(content)
                elif node_type == "OBSERVATION":
                    observation_parts.append(content)
            
            # 合并同一迭代的多个节点
            iteration_ctx = {
                "iteration": iter_num,
                "thought": " | ".join(thought_parts)[:500],  # 限制长度
                "action": " && ".join(action_parts)[:300],
                "observation": "\n".join(observation_parts)[:500],
            }
            
            # 只保留有实际内容的迭代
            if iteration_ctx["thought"] or iteration_ctx["action"] or iteration_ctx["observation"]:
                iterations.append(iteration_ctx)
        
        return iterations[-10:]  # 最多保留最近10次迭代

    def _compact_text(self, text: str, limit: int = 200) -> str:
        if not text:
            return ""
        text = re.sub(r"```.*?```", "[code omitted]", text, flags=re.S)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > limit:
            text = text[:limit].rstrip() + "..."
        return text

    def _collect_files_touched(self, nodes: list[dict]) -> list[str]:
        files: list[str] = []
        seen: set[str] = set()
        for n in nodes:
            for f in n.get("files_touched") or []:
                if isinstance(f, str) and f not in seen:
                    seen.add(f)
                    files.append(f)
            payload = n.get("payload") or {}
            for f in payload.get("files_intended") or []:
                if isinstance(f, str) and f not in seen:
                    seen.add(f)
                    files.append(f)
        if files:
            return files[:10]

        pattern = r"[A-Za-z0-9_./\\-]+\\.(?:py|rst|md|txt|yaml|yml)"
        for n in nodes:
            content = n.get("content", "") or ""
            for m in re.findall(pattern, content):
                if m not in seen:
                    seen.add(m)
                    files.append(m)
            if len(files) >= 10:
                break
        if len(files) >= 10:
            return files[:10]
        for n in nodes:
            payload = n.get("payload") or {}
            cmd = payload.get("command") or ""
            for m in re.findall(pattern, cmd):
                if m not in seen:
                    seen.add(m)
                    files.append(m)
            if len(files) >= 10:
                break
        return files[:10]

    def _collect_structured_hints(self, nodes: list[dict]) -> tuple[list[str], list[str], str]:
        planned_files: list[str] = []
        planned_tests: list[str] = []
        patch_strategy = ""
        seen_files: set[str] = set()
        seen_tests: set[str] = set()
        for n in nodes:
            payload = n.get("payload") or {}
            for f in payload.get("files_intended") or []:
                if isinstance(f, str) and f not in seen_files:
                    seen_files.add(f)
                    planned_files.append(f)
            for t in payload.get("test_plan") or []:
                if isinstance(t, str) and t not in seen_tests:
                    seen_tests.add(t)
                    planned_tests.append(t)
            if not patch_strategy:
                strat = payload.get("patch_strategy")
                if isinstance(strat, str) and strat.strip():
                    patch_strategy = strat.strip()
        return planned_files[:10], planned_tests[:10], patch_strategy

    def _looks_like_write_command(self, command: str) -> bool:
        cmd = (command or "").strip()
        if not cmd:
            return False
        lowered = cmd.lower()
        if re.search(r"\bsed\b.*\s-i(\s|$)", lowered):
            return True
        if re.search(r"\bperl\b.*\s-pi(\s|$)", lowered):
            return True
        if re.search(r"\btee\b(\s|$)", lowered):
            return True
        if re.search(r"\b(cat|printf|echo)\b\s+.*\s>\s*\S+", cmd):
            return True
        for token in (" rm ", " mv ", " cp ", " touch ", " mkdir "):
            if token in f" {lowered} ":
                return True
        return False

    def _effective_iteration_counts(self, accumulated_context: list[dict]) -> tuple[int, int]:
        effective_iters = 0
        edit_iters = 0
        for ctx in accumulated_context:
            has_action = bool((ctx.get("action") or "").strip())
            has_obs = bool((ctx.get("observation") or "").strip())
            if has_action or has_obs:
                effective_iters += 1
            if self._looks_like_write_command(ctx.get("action", "")):
                edit_iters += 1
        return effective_iters, edit_iters

    def _select_recent_context(self, accumulated_context: list[dict]) -> list[dict]:
        if not accumulated_context:
            return []
        edit_ctx = [ctx for ctx in accumulated_context if self._looks_like_write_command(ctx.get("action", ""))]
        if edit_ctx:
            selected = [edit_ctx[-1]]
            last_ctx = accumulated_context[-1]
            if last_ctx is not selected[-1]:
                selected.append(last_ctx)
            return selected[-3:]
        return accumulated_context[-3:]

    def _format_external_fault_summary(
        self,
        role: str,
        initial_inputs: dict,
        accumulated_context: list[dict],
        fault_message: str,
        granularity: Granularity,
        last_snapshot: str,
        last_iteration: int,
        files_touched: list[str],
        planned_files: list[str],
        planned_tests: list[str],
        patch_strategy: str,
        effective_iters: int,
        edit_iters: int,
        last_completed_iter: int,
        last_action: dict | None,
        last_edit_action: dict | None,
        last_test_status: str,
        last_test_command: str,
        last_observation: dict | None,
    ) -> str:
        """格式化外部故障恢复摘要（中文）"""
        role_names = {
            "analyzer": "问题分析专家",
            "locator": "代码定位专家",
            "planner": "方案规划专家",
            "implementer": "代码实现专家",
            "verifier": "测试验证专家",
        }
        role_name = role_names.get(role, role)
        
        parts = [
            "# 恢复摘要（外部故障）",
            f"\n## 故障信息",
            f"- 故障类型：外部故障（网络中断/API超时）",
            f"- 故障位置：{role_name}阶段",
            f"- 故障原因：{fault_message or '外部服务暂时不可用'}",
            f"\n## 任务上下文",
            f"### 初始输入",
        ]
        
        for key, value in initial_inputs.items():
            if key in {"fix_plan", "problem_statement", "problem_analysis", "located_files"}:
                value_str = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                parts.append(f"- {key}: {value_str}")
        
        role_focus = self.role_focus.get(role, [])
        parts.append("\n## 角色视图")
        if role_focus:
            parts.append(f"- 关注点: {', '.join(role_focus)}")
        if planned_files:
            parts.append(f"- 计划修改: {', '.join(planned_files)}")
        if files_touched and (not planned_files or files_touched != planned_files):
            parts.append(f"- 已触及文件: {', '.join(files_touched)}")
        if planned_tests:
            parts.append(f"- 计划测试: {', '.join(planned_tests)}")
        if patch_strategy:
            parts.append(f"- 实施策略: {self._compact_text(patch_strategy, 140)}")

        parts.append("\n## 关键状态")
        parts.append(f"- 最后快照: {last_snapshot or 'unknown'}")
        parts.append(
            f"- 有效迭代数: {effective_iters} (编辑迭代: {edit_iters}, "
            f"最后完成迭代号: {last_completed_iter}, 最近迭代号: {last_iteration})"
        )
        if files_touched:
            parts.append(f"- 涉及文件: {', '.join(files_touched)}")
        action_node = last_edit_action or last_action
        if action_node:
            payload = action_node.get("payload") or {}
            cmd = payload.get("command") or action_node.get("content", "")
            label = "最近编辑命令" if last_edit_action else "最近命令"
            parts.append(f"- {label}: {self._compact_text(cmd, 160)}")
        if last_observation:
            parts.append(f"- 最近观测: {self._compact_text(last_observation.get('content', ''), 200)}")
        if last_test_status or last_test_command:
            status = last_test_status or "unknown"
            cmd = self._compact_text(last_test_command, 120)
            parts.append(f"- 最近测试: {status} {f'({cmd})' if cmd else ''}")
        if edit_iters == 0:
            parts.append("- 备注: 暂未检测到有效编辑（请优先执行代码修改）")

        include_recent = granularity != Granularity.CONCISE
        if accumulated_context and include_recent:
            recent = self._select_recent_context(accumulated_context)
            if granularity == Granularity.VERBOSE and len(accumulated_context) > len(recent):
                recent = accumulated_context[-5:]
            elif granularity == Granularity.STANDARD:
                recent = recent[-3:]
            else:
                recent = recent[-1:]
            parts.append(f"\n## 近期迭代摘要（最近{len(recent)}次）")
            for ctx in recent:
                iter_num = ctx.get("iteration", "?")
                parts.append(f"\n**迭代 {iter_num}:**")
                if ctx.get("thought"):
                    parts.append(f"- 思考: {self._compact_text(ctx['thought'], 160)}")
                if ctx.get("action"):
                    parts.append(f"- 执行: {self._compact_text(ctx['action'], 160)}")
                if ctx.get("observation"):
                    parts.append(f"- 观测: {self._compact_text(ctx['observation'], 180)}")
        
        # 计算下一步迭代次数
        next_iteration = last_completed_iter + 1

        next_steps = [
            "基于 fix_plan 继续完成剩余修改",
            "检查涉及文件的改动是否符合预期",
            "补充/更新相关测试并运行关键用例",
        ]
        if edit_iters == 0:
            next_steps.insert(0, "先完成代码修改（当前未检测到有效编辑）")
        last_obs_text = (last_observation.get("content") or "") if last_observation else ""
        last_obs_lower = last_obs_text.lower()
        if last_test_status and str(last_test_status).lower() in {"fail", "failed", "error"}:
            next_steps.insert(0, "先解决最近一次测试失败原因（见上方最近观测）")
            if "importerror" in last_obs_lower or "modulenotfounderror" in last_obs_lower:
                next_steps.insert(1, "先安装/构建依赖或扩展模块，再重跑测试")
        if planned_tests and not any(t for t in planned_tests if t in next_steps):
            next_steps.append(f"按计划运行测试: {planned_tests[0]}")
        
        parts.extend([
            f"\n## 恢复指令",
            f"1. **你的身份**: 你是{role_name}，负责{self._get_role_responsibility(role)}",
            f"2. **当前状态**: 工作环境已包含前{effective_iters}次有效迭代的修改（锚点=第{last_completed_iter}次）",
            f"3. **恢复方式**: 从第{next_iteration}次迭代继续（以上次完成迭代为锚点）",
            f"4. **下一步建议**: ",
            *[f"   - {step}" for step in next_steps[:5]],
        ])
        
        return "\n".join(parts)

    def _format_internal_fault_summary(
        self,
        role: str,
        healthy_phase: dict | None,
        diagnosis: str,
        last_healthy_node: dict | None,
        planned_files: list[str],
        planned_tests: list[str],
        patch_strategy: str,
    ) -> str:
        """格式化内部故障恢复摘要（中文）"""
        role_names = {
            "analyzer": "问题分析专家",
            "locator": "代码定位专家",
            "planner": "方案规划专家",
            "implementer": "代码实现专家",
            "verifier": "测试验证专家",
        }
        role_name = role_names.get(role, role)
        
        parts = [
            "# 恢复摘要（内部故障）",
            f"\n## 故障信息",
            f"- 故障类型：内部认知故障（幻觉/循环/方向偏离）",
            f"- 故障位置：{role_name}阶段",
            f"\n## 故障诊断",
            diagnosis,
            f"\n## 角色视图",
        ]

        role_focus = self.role_focus.get(role, [])
        if role_focus:
            parts.append(f"- 关注点: {', '.join(role_focus)}")
        if planned_files:
            parts.append(f"- 计划修改: {', '.join(planned_files)}")
        if planned_tests:
            parts.append(f"- 计划测试: {', '.join(planned_tests)}")
        if patch_strategy:
            parts.append(f"- 实施策略: {self._compact_text(patch_strategy, 140)}")

        parts.append(f"\n## 健康锚点")
        
        if healthy_phase:
            parts.append(f"- 阶段: {healthy_phase.get('phase', 'unknown')}")
            ts = healthy_phase.get('timestamp_start')
            if ts:
                parts.append(f"- 时间: {ts:.2f}")
            
            inputs = healthy_phase.get("inputs") or {}
            if inputs:
                parts.append("\n### 健康状态的输入")
                for key, value in inputs.items():
                    value_str = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    parts.append(f"- {key}: {value_str}")
            else:
                parts.append("\n### 健康状态的输入")
                parts.append("- （无法恢复原始输入，请基于最后健康节点的上下文继续）")
        
        parts.extend([
            f"\n## 恢复指令",
            f"1. **你的身份**: 你是{role_name}，负责{self._get_role_responsibility(role)}",
            f"2. **工作范围**: **仅限于{role_name}阶段的职责**，不要越界到其他阶段",
            f"3. **恢复方式**: 从上述【健康锚点】重新开始，忽略之前的错误尝试",
            f"4. **必须避免**: {self._extract_mistakes_from_diagnosis(diagnosis)}",
        ])
        
        return "\n".join(parts)

    def _get_role_responsibility(self, role: str) -> str:
        """获取角色职责描述"""
        responsibilities = {
            "analyzer": "理解问题本质，生成结构化分析报告",
            "locator": "在代码库中定位相关文件和函数",
            "planner": "基于分析和定位结果，设计详细的修复方案",
            "implementer": "根据修复计划执行具体的代码修改，生成补丁",
            "verifier": "运行测试验证修复效果",
        }
        return responsibilities.get(role, "完成分配的任务")

    def _extract_resume_state(self, nodes: list[dict]) -> dict:
        """从节点中提取Agent的resume状态（messages + history）
        
        用于外部故障恢复时，让Agent从中断点继续，而不是重新开始
        """
        # 重建messages（THOUGHT作为assistant，OBSERVATION作为user）
        messages = []
        history = []
        last_iteration = 0
        
        for node in nodes:
            node_type = node.get("node_type", "")
            content = (node.get("content") or "").strip()
            iteration = node.get("iteration", 0)
            last_iteration = max(last_iteration, iteration)
            
            if node_type == "THOUGHT":
                messages.append({"role": "assistant", "content": content})
            elif node_type == "OBSERVATION":
                # 构造user消息
                payload = node.get("payload", {})
                command = payload.get("command", "")
                returncode = payload.get("returncode", 0)
                obs_text = f"命令输出:\n{content[:2000]}"
                if returncode != 0:
                    obs_text += f"\n返回码: {returncode}"
                messages.append({"role": "user", "content": obs_text})
            elif node_type == "ACTION":
                # 记录到history
                payload = node.get("payload", {})
                history.append({
                    "command": payload.get("command", content),
                    "output": "",  # 从OBSERVATION节点获取
                    "returncode": 0,
                })
        
        return {
            "messages": messages[-20:],  # 最多保留最后20条消息
            "history": history[-10:],  # 最多保留最后10条命令
            "last_iteration": last_iteration,
        }
    
    def _extract_mistakes_from_diagnosis(self, diagnosis: str) -> str:
        """从诊断中提取需要避免的错误"""
        # 简单提取关键信息（实际可以用LLM优化）
        if "auth.py" in diagnosis.lower():
            return "不要再修改 auth.py，问题可能在其他文件"
        if "循环" in diagnosis or "loop" in diagnosis.lower():
            return "不要重复相同的操作"
        if "假设" in diagnosis:
            return "重新验证之前的假设，可能存在误判"
        return "参考上述诊断，避免重复错误路径"

    def _diagnose_internal_fault(self, request: RecoveryRequest, healthy_nodes: list[dict], faulty_nodes: list[dict], last_healthy: dict | None) -> str:
        """使用LLM诊断内部故障原因"""
        if not self.llm or not faulty_nodes:
            return "检测到内部故障，但无法生成详细诊断。建议从最后健康状态重新开始。"
        
        # 构建诊断prompt（中文）
        healthy_summary = self._history_segments(healthy_nodes, max_items=3)
        faulty_summary = self._history_segments(faulty_nodes, max_items=5)
        
        prompt = f"""你是一个多Agent系统的故障诊断专家。请分析以下Agent的工作轨迹，诊断其失败的原因。

**角色**: {request.target_role}

**健康阶段（最后3步）**:
{chr(10).join(healthy_summary)}

**故障阶段（最近5步）**:
{chr(10).join(faulty_summary)}

请用1-2段话回答：
1. Agent在哪个环节开始偏离正确方向？
2. 导致偏离的可能原因是什么（如：错误假设、遗漏关键信息、陷入循环等）？
3. 给出具体的纠正建议（哪些文件/路径应该关注，哪些应该避免）。

要求简洁、可操作。"""
        
        try:
            response = self.llm.query([
                {"role": "system", "content": "你是故障诊断专家，分析Agent行为偏差。"},
                {"role": "user", "content": prompt},
            ])
            return response.get("content", "诊断失败")[:800]
        except Exception:
            return "LLM诊断服务暂时不可用，请基于健康锚点重新开始。"

    def _normalize_llm_summary(self, text: str, *, healthy_node_id: str, allowed_node_ids: set[str]) -> str:
        """修正 LLM 摘要中关键字段，避免 node_id 乱填导致恢复提示自相矛盾。"""
        if not text or not healthy_node_id or healthy_node_id not in allowed_node_ids:
            return text
        lines = text.splitlines()
        in_anchor = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[健康锚点]"):
                in_anchor = True
                continue
            if stripped.startswith("[") and stripped.endswith("]") and not stripped.startswith("[健康锚点]"):
                in_anchor = False
            if in_anchor and stripped.startswith("- node_id:"):
                lines[i] = line.split(":", 1)[0] + f": {healthy_node_id}"
                break
            # 兼容少数模型输出没有 '-' 的情况
            if in_anchor and stripped.startswith("node_id:"):
                lines[i] = line.split(":", 1)[0] + f": {healthy_node_id}"
                break
        return "\n".join(lines)

    def summarize(self, request: RecoveryRequest) -> dict[str, Any]:
        """生成恢复摘要 - 根据故障类型采用不同策略
        
        外部故障：保留完整累积上下文（Agent宕机前的所有迭代）
        内部故障：切断错误分支，从最后健康意图重新开始
        """
        from swe_mas.utils.logger import get_logger
        logger = get_logger(__name__)
        
        strategy = self.decide_strategy(request)
        logger.info(f"[Recovery] 开始生成摘要 - role={request.target_role}, fault_type={request.fault_type}, granularity={strategy.granularity.value}")
        
        # 优先使用Phase快照（更准确）
        phase_snapshot = self.recorder.get_phase_snapshot(request.session_id)
        if phase_snapshot:
            logger.debug(f"[Recovery] 找到Phase快照 - phase={phase_snapshot.get('phase')}, inputs={list(phase_snapshot.get('inputs', {}).keys())}")
        else:
            logger.warning(f"[Recovery] 未找到Phase快照，将从细粒度节点恢复")
        
        if self._is_internal_fault(request.fault_type):
            # 内部故障：诊断+切断错误分支
            logger.info("[Recovery] 使用内部故障恢复策略（诊断+健康锚点）")
            return self._summarize_internal_fault(request, strategy, phase_snapshot)
        else:
            # 外部故障：累积上下文恢复
            logger.info("[Recovery] 使用外部故障恢复策略（累积上下文）")
            return self._summarize_external_fault(request, strategy, phase_snapshot)
