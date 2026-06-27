"""问题分析Agent"""

from typing import Any

from swe_mas.agents.base import BaseAgent, AgentConfig
from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


class AnalyzerAgent(BaseAgent):
    """问题分析专家
    
    职责：理解问题本质，生成结构化分析报告
    """
    
    def __init__(self, *args, **kwargs):
        config = kwargs.pop("config", AgentConfig())
        if not config.phase:
            config.phase = "analyze"
        super().__init__(*args, agent_type="analyzer", config=config, **kwargs)
    
    def run(self, problem_statement: str) -> dict[str, Any]:
        """分析问题
        
        Args:
            problem_statement: 问题描述
            
        Returns:
            {"analysis": "分析结果文本", "success": bool}
        """
        logger.info(f"[Analyzer] 开始分析问题...")
        
        # 记录Phase开始
        self._start_phase({"problem_statement": problem_statement})
        
        # 构建prompt
        system_prompt = self.prompts.get("system", "")
        user_prompt = self.render_template(
            self.prompts.get("user", ""),
            problem_statement=problem_statement
        )
        
        # 初始化对话
        self.messages = []
        self.history = []
        self.add_message("system", system_prompt)
        self.add_message("user", user_prompt)
        
        # 查询模型
        try:
            analysis = self.query_model()
            logger.info(f"[Analyzer] 分析完成")
            
            result = {
                "analysis": analysis,
                "success": True,
                "messages": self.messages.copy(),
            }
            self._end_phase(result, success=True)
            return result
        except Exception as e:
            logger.error(f"[Analyzer] 分析失败: {str(e)}")
            result = {
                "analysis": f"分析失败: {str(e)}",
                "success": False,
                "messages": self.messages.copy(),
            }
            self._end_phase(result, success=False)
            return result
