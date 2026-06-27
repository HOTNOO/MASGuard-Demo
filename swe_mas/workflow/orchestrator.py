"""多Agent协作编排器 - 使用LangGraph实现工作流"""

from typing import Any, TypedDict, Annotated, Protocol
from operator import add
from uuid import uuid4
from pathlib import Path

from langgraph.graph import StateGraph, END

from swe_mas.agents import (
    AnalyzerAgent,
    LocatorAgent,
    PlannerAgent,
    ImplementerAgent,
    VerifierAgent,
)
from swe_mas.agents.reproducer import ReproducerAgent
from swe_mas.utils.env_executor import LocalExecutor
from swe_mas.utils.logger import get_logger
from swe_mas.utils.git_utils import ensure_git_repo, check_git_available, setup_git_config
from swe_mas.memory import MemoryRecorder
from swe_mas import OUTPUT_DIR
from swe_mas.perception import PerceptionManager, GraphAnomalyDetector, FaultSignal, FaultType
from swe_mas.recovery import RecoveryManager, ResourceCondition, GitRollbacker, NoopRollbacker
from swe_mas.workflow.perception_worker import PerceptionWorker
from swe_mas.evolution.policy_store import PolicyStore
from swe_mas.evolution.bandit import BetaBandit
from swe_mas.evolution.summary_policy import SummaryPolicy
from swe_mas.evolution.perception_policy import PerceptionPolicy
from swe_mas.evolution.perception_templates import PERCEPTION_POLICIES

logger = get_logger(__name__)


class ChatModelProtocol(Protocol):
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        ...

    def get_template_vars(self) -> dict[str, Any]:
        ...


class WorkflowState(TypedDict):
    """工作流状态"""
    # 输入
    problem_statement: str
    working_directory: str
    recovery_context: str  # 显式的恢复上下文（与problem_statement分离）
    
    # 中间结果
    analysis: str
    located_files: str
    located_static: str
    reproduction: str
    plan: str
    patch: str
    verification: str
    recovery_summary: str
    recovery_struct: dict
    healthy_node_id: str
    last_snapshot: str
    fault_type: str
    recovery_count: int

    # 状态标记
    current_step: str
    success: bool
    error: str | None
    static_success: bool
    dynamic_success: bool

    # 完整历史
    all_messages: Annotated[list[dict], add]
    all_commands: Annotated[list[dict], add]


class SWEOrchestrator:
    """SWE-bench多Agent协作编排器
    
    使用LangGraph管理5个Agent的协作流程：
    Analyzer → Locator → Planner → Implementer → Verifier
    """
    
    def __init__(
        self,
        model: ChatModelProtocol,
        executor: LocalExecutor,
        max_iterations_per_agent: int = 10,
        models_by_role: dict[str, ChatModelProtocol] | None = None,
        recorder: MemoryRecorder | None = None,
        perception: PerceptionManager | None = None,
        recovery: RecoveryManager | None = None,
        resource_condition: ResourceCondition | None = None,
        fault_injection: bool = False,
        session_id: str | None = None,
        anchor_node_id: str | None = None,
        disable_perception: bool = False,
        disable_recovery: bool = False,
        disable_auto_relay: bool = False,
    ):
        """初始化编排器

        Args:
            model: 通义千问模型
            executor: 命令执行器
            max_iterations_per_agent: 每个Agent的最大迭代次数
            models_by_role: 可选，按角色覆盖的模型配置
        """
        # 统一的模型映射，默认为所有角色复用同一个模型
        roles = [
            "analyzer",
            "locator_static",
            "locator_dynamic",
            "planner",
            "implementer",
            "verifier",
        ]
        if models_by_role:
            self.models: dict[str, ChatModelProtocol] = {
                role: models_by_role.get(role, model) for role in roles
            }
        else:
            self.models = {role: model for role in roles}
        self.executor = executor
        self.max_iterations = max_iterations_per_agent
        self.recorder = recorder or MemoryRecorder(OUTPUT_DIR / "memory.db")
        # 允许外部指定session_id，用于跨运行共享同一记忆图
        self.session_id: str | None = session_id
        # 允许外部指定健康锚点，用于摘要恢复时从健康节点分叉
        self.anchor_node_id: str | None = anchor_node_id
        policy_path = OUTPUT_DIR / "evolution_policy.json"
        self.policy_store = PolicyStore.load(policy_path)
        self.summary_policy = SummaryPolicy(recorder=self.recorder, store=self.policy_store, bandit=BetaBandit())
        self.perception_policy = PerceptionPolicy(store=self.policy_store, bandit=BetaBandit())
        default_perception_policy = PERCEPTION_POLICIES["P0_balanced"].config
        # 默认感知：使用可调策略的循环检测（可外部传入替换）
        if disable_perception:
            if perception is not None:
                logger.warning("disable_perception=True，忽略外部传入的 perception。")
            self.perception = None
        else:
            self.perception = perception or PerceptionManager(
                graph_detector=GraphAnomalyDetector(
                    self.recorder,
                    policy=default_perception_policy,
                    perception_policy=self.perception_policy,
                )
            )

        # 默认恢复：使用当前模型做摘要（可外部传入小模型/其他LLM）
        if disable_recovery:
            if recovery is not None:
                logger.warning("disable_recovery=True，忽略外部传入的 recovery。")
            self.recovery = None
        else:
            self.recovery = recovery or RecoveryManager(recorder=self.recorder, llm=model, summary_policy=self.summary_policy)
        self.resource_condition = resource_condition or ResourceCondition()
        self.fault_injection = fault_injection
        # 回滚器：有 git 用 git，否则 noop
        self.rollbacker = GitRollbacker() if check_git_available() else NoopRollbacker()
        self.perception_worker: PerceptionWorker | None = None
        self._fault_signal = None
        self.disable_auto_relay = disable_auto_relay
        
        # 创建工作流图
        self.workflow = self._build_workflow()

    def _step_to_role(self, step: str) -> str:
        mapping = {
            "analyze": "analyzer",
            "locate_static": "locator",
            "locate_dynamic": "locator",
            "merge_locate": "locator",
            "plan": "planner",
            "implement": "implementer",
            "verify": "verifier",
        }
        base = step.replace("_fault", "").replace("_fault_injected", "") if step else ""
        return mapping.get(base, base or "unknown")

    def _is_prefault_stop(self, fault: FaultSignal | None) -> bool:
        if not fault:
            return False
        if getattr(fault, "suggested_action", None) == "stop":
            return True
        extra = getattr(fault, "extra", None) or {}
        return bool(extra.get("is_prefault_stop"))

    def _should_stop_agent(self) -> bool:
        if not self.perception_worker:
            return False
        fault = self.perception_worker.peek_fault()
        return self._is_prefault_stop(fault)

    def _maybe_perceive(self, state: WorkflowState) -> WorkflowState:
        """调用感知层，若发现故障则生成摘要并自动接力恢复"""
        mapped_role = self._step_to_role(state.get("current_step", ""))
        fault = self.perception_worker.consume_fault() if self.perception_worker else None
        # 若当前 state 已经记录了"模型调用异常"等外部错误，构造一个外部故障信号触发恢复链路
        if not fault and state.get("error") and isinstance(state.get("error"), str):
            err_text = state["error"]
            if "模型调用异常" in err_text or "API调用失败" in err_text or "MODEL_ERROR" in err_text or "AllocationQuota" in err_text:
                fault = FaultSignal(
                    fault_type=FaultType.EXTERNAL,
                    severity="error",
                    agent_id=mapped_role,
                    message=err_text,
                    suggested_action="retry_with_backup_or_summary",
                    extra={"source": "orchestrator_error"},
                    node_ids=None,
                )
        if not fault and self.perception:
            signals = self.perception.run(session_id=self.session_id or state.get("session_id", ""))
            fault = signals[0] if signals else None
        if not fault:
            return state

        if self._is_prefault_stop(fault):
            state["success"] = False
            state["error"] = fault.message or "Prefault stop"
            state["fault_type"] = str(fault.fault_type)
            state["current_step"] = f"{state.get('current_step', '')}_stopped"
            return state
        
        # 发现故障 - 开始恢复流程
        state["recovery_count"] = state.get("recovery_count", 0) + 1
        logger.warning(f"检测到故障 [{fault.fault_type}]，开始恢复流程（第{state['recovery_count']}次）")
        
        if self.recovery:
            summary = self.recovery.handle_fault(
                fault=fault,
                session_id=self.session_id or state.get("session_id", ""),
                target_role=mapped_role,
                resource=self.resource_condition,
                fault_index=state["recovery_count"],
            )
            state["recovery_summary"] = summary.get("summary", "")
            state["recovery_struct"] = summary.get("summary_struct", {})
            state["recovery_trace"] = summary.get("trace", {})
            state["healthy_node_id"] = summary.get("healthy_node_id", "")
            
            # 回滚到健康快照
            snapshot_to_use = summary.get("healthy_snapshot") or summary.get("last_snapshot", "")
            state["last_snapshot"] = snapshot_to_use
            state["fault_type"] = str(fault.fault_type)
            
            if snapshot_to_use:
                logger.info(f"回滚到快照: {snapshot_to_use[:12]}...")
                self.rollbacker.rollback(snapshot_to_use, state.get("working_directory", ""))
            
            # 🔑 关键改进：自动接力 - 将恢复摘要显式存入recovery_context
            recovery_summary_text = state.get("recovery_summary", "")
            if recovery_summary_text:
                if not self.disable_auto_relay:
                    # 自动接力模式：设置recovery_context，继续工作流
                    state["recovery_context"] = recovery_summary_text
                    logger.info("✓ [自动接力] 已设置recovery_context，Agent将基于恢复上下文继续工作")
                    
                    # 设置健康锚点，让Agent从健康节点分叉
                    if state.get("healthy_node_id"):
                        self.anchor_node_id = state["healthy_node_id"]
                    
                    # ✅ 标记为可继续（而不是中断工作流）
                    state["success"] = True
                    state["error"] = None
                    logger.info(f"✓ [自动接力] 恢复完成，继续执行 {mapped_role} 阶段")
                else:
                    # 手动接力模式：保留摘要但中断工作流（兼容旧实验脚本）
                    state["success"] = False
                    state["error"] = f"故障已检测并生成恢复摘要（禁用自动接力）"
                    logger.info("✓ [手动接力] 恢复摘要已生成，工作流将中断，等待手动重启")
            else:
                # 没有恢复摘要，直接中断
                state["success"] = False
                state["error"] = fault.message or f"Fault: {fault.fault_type}"
        else:
            # 没有恢复机制，只能中断
            state["success"] = False
            state["error"] = fault.message or f"Fault: {fault.fault_type}"
        
        state["current_step"] = f"{state.get('current_step', '')}_recovered"
        return state

    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流
        
        Returns:
            编译好的工作流图
        """
        # 创建状态图
        workflow = StateGraph(WorkflowState)

        # 添加节点（每个Agent一个节点）
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("locate_static", self._locate_static_node)
        workflow.add_node("locate_dynamic", self._locate_dynamic_node)
        workflow.add_node("merge_locate", self._merge_locate_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("implement", self._implement_node)
        workflow.add_node("verify", self._verify_node)

        # 定义边（工作流）
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "locate_static")
        workflow.add_edge("analyze", "locate_dynamic")
        workflow.add_edge("locate_static", "merge_locate")
        workflow.add_edge("locate_dynamic", "merge_locate")
        workflow.add_edge("merge_locate", "plan")
        workflow.add_edge("plan", "implement")
        workflow.add_edge("implement", "verify")
        workflow.add_edge("verify", END)

        # 编译工作流
        return workflow.compile()
    
    def _analyze_node(self, state: WorkflowState) -> WorkflowState:
        """分析节点 - 运行AnalyzerAgent"""
        logger.info("=" * 60)
        logger.info("步骤 1/6: 问题分析")
        logger.info("=" * 60)
        
        state["current_step"] = "analyze"
        
        try:
            agent = AnalyzerAgent(
                model=self.models["analyzer"],
                executor=self.executor,
                recorder=self.recorder,
                session_id=self.session_id,
            )
            # 设置健康锚点（如果有恢复发生）
            anchor = self.anchor_node_id or state.get("healthy_node_id")
            if anchor:
                agent.set_anchor_node(anchor)
                logger.info(f"[Analyzer] 设置健康锚点: {anchor[:12]}...")
            
            # 构建完整的问题描述（包含恢复上下文）
            problem_with_context = state.get("problem_statement", "")
            recovery_ctx = state.get("recovery_context", "")
            if recovery_ctx:
                problem_with_context = f"{recovery_ctx}\n\n---\n\n# 原始问题\n\n{problem_with_context}"
            
            result = agent.run(problem_statement=problem_with_context)
            
            state["analysis"] = result["analysis"]
            state["all_messages"].extend(result.get("messages", []))
            
            if not result["success"]:
                state["success"] = False
                state["error"] = "分析阶段失败"
                
            logger.info(f"分析完成")
            
        except Exception as e:
            logger.error(f"分析阶段异常: {str(e)}")
            state["success"] = False
            state["error"] = f"分析异常: {str(e)}"
            state["analysis"] = ""
        
        return self._maybe_perceive(state)
    
    def _locate_static_node(self, state: WorkflowState) -> WorkflowState:
        """静态定位节点 - 运行LocatorAgent"""
        logger.info("=" * 60)
        logger.info("步骤 2a/6: 静态代码定位")
        logger.info("=" * 60)

        if not state.get("success", True):
            logger.warning("跳过静态定位（前序步骤失败）")
            state["located_static"] = ""
            state["static_success"] = False
            return state

        try:
            agent = LocatorAgent(
                model=self.models["locator_static"],
                executor=self.executor,
                recorder=self.recorder,
                session_id=self.session_id,
            )
            # 设置健康锚点
            anchor = self.anchor_node_id or state.get("healthy_node_id")
            if anchor:
                agent.set_anchor_node(anchor)
                logger.info(f"[Locator-Static] 设置健康锚点: {anchor[:12]}...")
            agent.config.max_iterations = self.max_iterations

            result = agent.run(
                problem_analysis=state["analysis"],
                cwd=state["working_directory"],
            )

            logger.info("静态定位完成")
            state["located_static"] = result.get("located_files", "")
            state["static_success"] = result.get("success", False)
            state["all_messages"].extend(result.get("messages", []))
            state["all_commands"].extend(result.get("commands", []))

        except Exception as e:
            logger.error(f"静态定位异常: {str(e)}")
            state["static_success"] = False
            state["error"] = f"静态定位异常: {str(e)}"
            state["located_static"] = ""

        return self._maybe_perceive(state)

    def _locate_dynamic_node(self, state: WorkflowState) -> WorkflowState:
        """动态复现节点 - 运行ReproducerAgent"""
        logger.info("=" * 60)
        logger.info("步骤 2b/6: 动态复现")
        logger.info("=" * 60)

        if not state.get("success", True):
            logger.warning("跳过动态复现（前序步骤失败）")
            state["reproduction"] = ""
            state["dynamic_success"] = False
            return state

        try:
            agent = ReproducerAgent(
                model=self.models["locator_dynamic"],
                executor=self.executor,
                recorder=self.recorder,
                session_id=self.session_id,
            )
            # 设置健康锚点
            anchor = self.anchor_node_id or state.get("healthy_node_id")
            if anchor:
                agent.set_anchor_node(anchor)
                logger.info(f"[Locator-Dynamic] 设置健康锚点: {anchor[:12]}...")
            agent.config.max_iterations = self.max_iterations

            result = agent.run(
                problem_analysis=state["analysis"],
                cwd=state["working_directory"],
            )

            logger.info("动态复现完成")
            state["reproduction"] = result.get("reproduction", "")
            state["dynamic_success"] = result.get("success", False)
            state["all_messages"].extend(result.get("messages", []))
            state["all_commands"].extend(result.get("commands", []))

        except Exception as e:
            logger.error(f"动态复现异常: {str(e)}")
            state["dynamic_success"] = False
            state["error"] = f"动态复现异常: {str(e)}"
            state["reproduction"] = ""

        return self._maybe_perceive(state)

    def _merge_locate_node(self, state: WorkflowState) -> WorkflowState:
        """合并定位结果"""
        logger.info("=" * 60)
        logger.info("步骤 3/6: 合并定位结果")
        logger.info("=" * 60)

        static_part = state.get("located_static", "")
        dynamic_part = state.get("reproduction", "")

        # 若两个阶段都完全没有任何定位/复现内容，才视为致命失败；
        # 否则允许带着低置信度结果继续后续规划与实现，由后续阶段和感知/恢复机制纠偏。
        if not static_part and not dynamic_part:
            state["success"] = False
            state["error"] = "定位信息为空"
            state["located_files"] = ""
            state["current_step"] = "merge_locate"
            return state

        combined = []
        if static_part:
            combined.append("## Static findings\n" + static_part)
        if dynamic_part:
            combined.append("## Dynamic reproduction\n" + dynamic_part)

        logger.info("定位结果合并完成")
        state["located_files"] = "\n\n".join(combined)
        state["current_step"] = "merge_locate"
        return self._maybe_perceive(state)
    
    def _plan_node(self, state: WorkflowState) -> WorkflowState:
        """规划节点 - 运行PlannerAgent"""
        logger.info("=" * 60)
        logger.info("步骤 4/6: 方案规划")
        logger.info("=" * 60)
        
        state["current_step"] = "plan"
        
        if not state.get("success", True):
            logger.warning("跳过规划阶段（前序步骤失败）")
            state["plan"] = ""
            return state
        
        try:
            agent = PlannerAgent(
                model=self.models["planner"],
                executor=self.executor,
                recorder=self.recorder,
                session_id=self.session_id,
            )
            # 设置健康锚点
            anchor = self.anchor_node_id or state.get("healthy_node_id")
            if anchor:
                agent.set_anchor_node(anchor)
                logger.info(f"[Planner] 设置健康锚点: {anchor[:12]}...")
            
            result = agent.run(
                problem_analysis=state["analysis"],
                located_files=state["located_files"],
                reproduction_info=state.get("reproduction", ""),
            )
            
            state["plan"] = result["plan"]
            state["all_messages"].extend(result.get("messages", []))
            
            if not result["success"]:
                state["success"] = False
                state["error"] = "规划阶段失败"
                
            logger.info(f"规划完成")
            
        except Exception as e:
            logger.error(f"规划阶段异常: {str(e)}")
            state["success"] = False
            state["error"] = f"规划异常: {str(e)}"
            state["plan"] = ""
        
        return self._maybe_perceive(state)
    
    def _implement_node(self, state: WorkflowState) -> WorkflowState:
        """实现节点 - 运行ImplementerAgent"""
        logger.info("=" * 60)
        logger.info("步骤 5/6: 代码实现")
        logger.info("=" * 60)
        
        state["current_step"] = "implement"
        
        if not state.get("success", True):
            logger.warning("跳过实现阶段（前序步骤失败）")
            state["patch"] = ""
            return state
        
        # 故障注入：在实现阶段开始时可注入模拟"内部认知故障"，触发恢复链路
        if self.fault_injection:
            logger.warning("🧪 注入故障: implement 阶段开始前 (fault_injection=True)")
            # 使用标准 FaultSignal，类型为 INJECTED，语义等价于"模拟内部认知故障"
            fault = FaultSignal(
                fault_type=FaultType.INJECTED,
                severity="warn",
                agent_id="implementer",
                message="Injected internal fault before implement",
                suggested_action="rollback_to_healthy_anchor",
                extra={"injected": True},
                node_ids=None,
            )
            state["fault_type"] = str(fault.fault_type)
            state["recovery_count"] = state.get("recovery_count", 0) + 1

            if not self.recovery:
                logger.warning("🧪 故障已注入，但 recovery 已禁用：跳过摘要生成与回滚。")
                state["success"] = False
                state["error"] = fault.message or "Injected internal fault"
                state["current_step"] = "implement_fault_injected"
                return state

            summary = self.recovery.handle_fault(
                fault=fault,
                session_id=self.session_id or state.get("session_id", ""),
                target_role="implementer",
                resource=self.resource_condition,
                fault_index=state["recovery_count"],
            )
            state["recovery_summary"] = summary.get("summary", "")
            state["recovery_struct"] = summary.get("summary_struct", {})
            state["healthy_node_id"] = summary.get("healthy_node_id", "")
            
            # 回滚到健康锚点
            state["last_snapshot"] = summary.get("healthy_snapshot") or summary.get("last_snapshot", "")
            if state.get("last_snapshot"):
                logger.info(f"🧪 回滚到健康快照: {state['last_snapshot'][:12]}...")
                self.rollbacker.rollback(state["last_snapshot"], state.get("working_directory", ""))
            
            # 🔑 自动接力：设置恢复上下文，让Agent继续工作（而不是中断）
            recovery_summary_text = state.get("recovery_summary", "")
            if recovery_summary_text:
                if not self.disable_auto_relay:
                    # 自动接力模式
                    state["recovery_context"] = recovery_summary_text
                    logger.info("🧪 [自动接力] 已设置recovery_context，Agent将基于恢复上下文继续实现")
                    
                    if state.get("healthy_node_id"):
                        self.anchor_node_id = state["healthy_node_id"]
                    
                    # ✅ 标记为可继续（实验时也让其自动接力，而不是手动重启）
                    state["success"] = True
                    state["error"] = None
                    logger.info("🧪 [自动接力] 故障注入完成，继续执行 implementer 阶段")
                    # 继续执行，不return
                else:
                    # 手动接力模式：中断工作流
                    logger.info("🧪 [手动接力] 故障注入完成，工作流将中断")
            state["current_step"] = "implement_fault_injected"
            return state

        try:
            agent = ImplementerAgent(
                model=self.models["implementer"],
                executor=self.executor,
                recorder=self.recorder,
                session_id=self.session_id,
            )
            agent.set_stop_checker(self._should_stop_agent)
            # 设置健康锚点（如果有恢复发生）
            anchor = self.anchor_node_id or state.get("healthy_node_id")
            if anchor:
                agent.set_anchor_node(anchor)
                logger.info(f"设置健康锚点: {anchor[:12]}...")
            
            # 设置恢复状态（如果有）- 用于Agent resume
            recovery_struct = state.get("recovery_struct", {})
            if recovery_struct and recovery_struct.get("resume_state"):
                agent.set_recovery_state(recovery_struct["resume_state"])
                logger.info(f"✓ 设置Resume状态 - 从第{recovery_struct['resume_state'].get('last_iteration', 0)+1}次迭代继续")
            
            agent.config.max_iterations = self.max_iterations
            
            # 构建完整的问题描述（包含恢复上下文）
            problem_with_context = state.get("problem_statement", "")
            recovery_ctx = state.get("recovery_context", "")
            if recovery_ctx:
                problem_with_context = f"{recovery_ctx}\n\n---\n\n# 原始问题\n\n{problem_with_context}"
            
            result = agent.run(
                fix_plan=state["plan"],
                cwd=state["working_directory"],
                problem_statement=problem_with_context,
            )
            
            state["patch"] = result["patch"]
            state["all_messages"].extend(result.get("messages", []))
            state["all_commands"].extend(result.get("commands", []))
            
            if not result["success"]:
                state["success"] = False
                state["error"] = result.get("error") or "实现阶段失败"
                
            logger.info(f"实现完成")
            
        except Exception as e:
            logger.error(f"实现阶段异常: {str(e)}")
            state["success"] = False
            state["error"] = f"实现异常: {str(e)}"
            state["patch"] = ""
        
        return self._maybe_perceive(state)
    
    def _verify_node(self, state: WorkflowState) -> WorkflowState:
        """验证节点 - 运行VerifierAgent"""
        logger.info("=" * 60)
        logger.info("步骤 6/6: 验证测试")
        logger.info("=" * 60)
        
        state["current_step"] = "verify"
        
        if not state.get("success", True):
            logger.warning("跳过验证阶段（前序步骤失败）")
            state["verification"] = ""
            return state
        
        try:
            agent = VerifierAgent(
                model=self.models["verifier"],
                executor=self.executor,
                recorder=self.recorder,
                session_id=self.session_id,
            )
            # 设置健康锚点
            anchor = self.anchor_node_id or state.get("healthy_node_id")
            if anchor:
                agent.set_anchor_node(anchor)
                logger.info(f"[Verifier] 设置健康锚点: {anchor[:12]}...")
            agent.config.max_iterations = self.max_iterations
            
            # 构建完整的问题描述（包含恢复上下文）
            problem_with_context = state.get("problem_statement", "")
            recovery_ctx = state.get("recovery_context", "")
            if recovery_ctx:
                problem_with_context = f"{recovery_ctx}\n\n---\n\n# 原始问题\n\n{problem_with_context}"
            
            result = agent.run(
                problem_statement=problem_with_context,
                implementation=state["patch"],
                cwd=state["working_directory"],
            )
            
            state["verification"] = result["verification"]
            state["all_messages"].extend(result.get("messages", []))
            state["all_commands"].extend(result.get("commands", []))
            verify_success = result.get("success", None)
            infrastructure_error = bool(result.get("infrastructure_error"))
            status = (result.get("status") or "").strip()
            status_norm = status.lower()
            failed_by_status = False
            needs_improvement = False
            if status:
                if "失败" in status or "fail" in status_norm or status_norm in {"failed", "fail"}:
                    failed_by_status = True
                if "需要改进" in status or "needs" in status_norm or "improve" in status_norm:
                    failed_by_status = True
                    needs_improvement = True
            if infrastructure_error:
                state["success"] = False
                state["error"] = result.get("error") or "MODEL_ERROR: verifier_infrastructure_failure"
            elif verify_success is False or failed_by_status:
                state["success"] = False
                if not state.get("error"):
                    state["error"] = result.get("error") or ("验证需要改进" if needs_improvement else "验证阶段失败")
            logger.info(f"验证完成: {status or '未知'}")
            
        except Exception as e:
            logger.error(f"验证阶段异常: {str(e)}")
            state["error"] = f"验证异常: {str(e)}"
            state["verification"] = ""
        
        return self._maybe_perceive(state)
    
    def run(
        self,
        problem_statement: str,
        working_directory: str,
        *,
        start_step: str | None = None,
        stop_before_step: str | None = None,
        prefilled_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """运行完整的多Agent协作流程
        
        Args:
            problem_statement: 问题描述
            working_directory: 工作目录
            start_step: 可选，从指定步骤开始（跳过之前步骤），如 "implement"
            stop_before_step: 可选，在指定步骤前停止，用于生成前置快照
            prefilled_state: 可选，预填充中间结果（analysis/located_files/reproduction/plan等），用于resume
            
        Returns:
            包含所有结果的字典
        """
        logger.info("🚀 开始多Agent协作解决SWE-bench问题")
        logger.info(f"工作目录: {working_directory}")
        logger.info(
            "运行配置: "
            f"fault_injection={self.fault_injection}, "
            f"perception={'on' if self.perception else 'off'}, "
            f"recovery={'on' if self.recovery else 'off'}; "
            f"start_step={start_step or 'default'}, "
            f"stop_before_step={stop_before_step or 'none'}"
        )

        # 为本次运行生成唯一session_id
        if not self.session_id:
            self.session_id = str(uuid4())
        
        # 检查git是否可用
        if not check_git_available():
            logger.error("❌ Git未安装或不可用，无法生成补丁")
            return {
                "problem_statement": problem_statement,
                "working_directory": working_directory,
                "analysis": "",
                "located_files": "",
                "located_static": "",
                "reproduction": "",
                "plan": "",
                "patch": "",
                "verification": "",
                "current_step": "init",
                "success": False,
                "error": "Git未安装或不可用",
                "static_success": False,
                "dynamic_success": False,
                "all_messages": [],
                "all_commands": [],
            }
        
        # 配置Git使用SSH（如果环境变量设置了USE_GIT_SSH=true）
        import os
        if os.getenv("USE_GIT_SSH", "false").lower() == "true":
            logger.info("🔐 配置Git使用SSH...")
            setup_git_config(use_ssh=True)
        
        # 确保是git仓库
        if not ensure_git_repo(working_directory):
            logger.warning("⚠️  无法初始化Git仓库，补丁生成可能失败")
        
        # 初始化状态
        initial_state: WorkflowState = {
            "problem_statement": problem_statement,
            "working_directory": working_directory,
            "recovery_context": "",  # 初始为空，故障恢复时会填充
            "analysis": "",
            "located_files": "",
            "located_static": "",
            "reproduction": "",
            "plan": "",
            "patch": "",
            "verification": "",
            "recovery_summary": "",
            "recovery_struct": {},
            "healthy_node_id": "",
            "last_snapshot": "",
            "fault_type": "",
            "recovery_count": 0,
            "current_step": "",
            "success": True,
            "error": None,
            "static_success": True,
            "dynamic_success": True,
            "all_messages": [],
            "all_commands": [],
            "session_id": self.session_id,
        }
        # 预填充已有的中间结果（用于resume）
        if prefilled_state:
            for k, v in prefilled_state.items():
                if k in initial_state:
                    initial_state[k] = v
        
        # 启动后台感知线程
        if self.perception:
            self.perception_worker = PerceptionWorker(self.perception, self.session_id, self.recorder._event, interval=2.0)
            self.recorder.subscribe(lambda _: None)  # 确保有事件触发
            self.perception_worker.start()

        state = initial_state
        steps = [
            ("analyze", self._analyze_node),
            ("locate_static", self._locate_static_node),
            ("locate_dynamic", self._locate_dynamic_node),
            ("merge_locate", self._merge_locate_node),
            ("plan", self._plan_node),
            ("implement", self._implement_node),
            ("verify", self._verify_node),
        ]

        # 计算起始步
        start_idx = 0
        if start_step:
            for i, (name, _) in enumerate(steps):
                if name == start_step:
                    start_idx = i
                    break

        try:
            for name, step_fn in steps[start_idx:]:
                if stop_before_step and name == stop_before_step:
                    logger.info(f"检测到 stop_before_step={stop_before_step}，提前停止。")
                    break
                state = step_fn(state)
                if not state.get("success", True):
                    # 预留接力钩子：若有恢复摘要，可在此切换新Agent继续
                    break
        except RuntimeError as e:
            logger.warning(f"感知到故障，中断流程: {e}")
        except Exception as e:
            logger.error(f"❌ 工作流执行异常: {str(e)}")
            state["success"] = False
            state["error"] = f"工作流异常: {str(e)}"
        finally:
            if self.perception_worker:
                self.perception_worker.stop()

        logger.info("=" * 60)
        if state.get("success", False) and state.get("patch"):
            logger.info("✅ 多Agent协作完成，成功生成补丁")
        else:
            logger.warning(f"⚠️  流程完成但可能存在问题: {state.get('error', '未知')}")
        logger.info("=" * 60)

        # 简单的进化埋点：根据整体成功/失败为本次会话打一个奖励标签
        try:
            if self.recorder and self.session_id:
                nodes = self.recorder.get_nodes(self.session_id)
                if nodes:
                    reward = 1.0 if state.get("success") else -1.0
                    # 使用最后一个节点作为代表，记录策略信息（例如是否发生故障）
                    strategy_label = state.get("fault_type") or "normal"
                    self.recorder.log_feedback(
                        nodes[-1]["node_id"],
                        reward=reward,
                        strategy=strategy_label,
                        policy_name="episode_result",
                        strategy_json={
                            "fault_type": state.get("fault_type"),
                            "used_recovery": bool(state.get("recovery_summary")),
                        },
                    )
                    # 将本 episode 内 reward=0 的反馈统一回填
                    self.recorder.apply_episode_reward(self.session_id, reward)
            if self.policy_store:
                self.policy_store.save()
        except Exception:
            pass

        return state
