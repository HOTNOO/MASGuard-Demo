"""方案规划Agent"""

from typing import Any

from swe_mas.agents.base import AgentConfig, BaseAgent
from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


class PlannerAgent(BaseAgent):
    """方案规划专家
    
    职责：基于问题分析和代码定位，设计详细的修复方案
    """
    
    def __init__(self, *args, **kwargs):
        config = kwargs.pop("config", AgentConfig())
        if not config.phase:
            config.phase = "plan"
        super().__init__(*args, agent_type="planner", config=config, **kwargs)
    
    def run(self, problem_analysis: str, located_files: str, reproduction_info: str = "") -> dict[str, Any]:
        """规划修复方案
        
        Args:
            problem_analysis: 问题分析结果
            located_files: 定位到的相关文件
            reproduction_info: 动态复现信息（错误日志/失败路径）
            
        Returns:
            {"plan": "修复计划", "success": bool}
        """
        logger.info(f"[Planner] 开始规划方案...")
        
        # 记录Phase开始
        self._start_phase({
            "problem_analysis": problem_analysis,
            "located_files": located_files,
            "reproduction_info": reproduction_info,
        })
        
        # 构建prompt
        system_prompt = self.prompts.get("system", "")
        user_prompt = self.render_template(
            self.prompts.get("user", ""),
            problem_analysis=problem_analysis,
            located_files=located_files,
            reproduction_info=reproduction_info,
        )
        
        # 初始化对话
        self.messages = []
        self.history = []
        self.add_message("system", system_prompt)
        self.add_message("user", user_prompt)
        
        # 查询模型
        try:
            plan = self.query_model()
            logger.info(f"[Planner] 规划完成")
            
            result = {
                "plan": plan,
                "success": True,
                "messages": self.messages.copy(),
            }
            self._end_phase(result, success=True)
            return result
        except Exception as e:
            logger.error(f"[Planner] 规划失败: {str(e)}")
            result = {
                "plan": f"规划失败: {str(e)}",
                "success": False,
                "messages": self.messages.copy(),
            }
            self._end_phase(result, success=False)
            return result
