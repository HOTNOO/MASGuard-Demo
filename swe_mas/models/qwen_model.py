"""通义千问模型接口实现"""

import os
from dataclasses import dataclass, field
from typing import Any

from dashscope import Generation
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class QwenModelConfig:
    """通义千问模型配置"""
    model_name: str = "qwen-max"  # qwen-max, qwen-plus, qwen-turbo
    api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 2000
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY未设置，请在.env文件中配置")


class QwenModel:
    """通义千问模型封装类
    
    符合mini-swe-agent的Model Protocol:
    - config: 配置对象
    - cost: 累计成本（通义千问暂时返回0）
    - n_calls: 调用次数
    - query(messages) -> dict
    - get_template_vars() -> dict
    """
    
    def __init__(self, config: QwenModelConfig | None = None, **kwargs):
        """初始化模型
        
        Args:
            config: 模型配置对象
            **kwargs: 配置参数，用于创建QwenModelConfig
        """
        if config is None:
            config = QwenModelConfig(**kwargs)
        self.config = config
        self.cost = 0.0  # 通义千问目前不返回成本信息，保持为0
        self.n_calls = 0
        
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """查询模型
        
        Args:
            messages: 对话消息列表，格式 [{"role": "user/assistant/system", "content": "..."}]
            **kwargs: 额外参数覆盖配置
            
        Returns:
            {"content": "模型回复内容", "extra": {...}}
        """
        self.n_calls += 1
        
        # 合并配置参数
        params = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "api_key": self.config.api_key,
            "result_format": "message",  # 返回消息格式
        }
        
        try:
            response = Generation.call(**params)
            
            # 检查响应状态
            if response.status_code != 200:
                raise RuntimeError(
                    f"通义千问API调用失败: status_code={response.status_code}, "
                    f"code={response.code}, message={response.message}"
                )
            
            # 提取回复内容
            content = response.output.choices[0].message.content
            
            return {
                "content": content,
                "extra": {
                    "request_id": response.request_id,
                    "usage": response.usage,
                    "model": self.config.model_name,
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"通义千问模型调用异常: {str(e)}") from e
    
    def get_template_vars(self) -> dict[str, Any]:
        """获取模板变量
        
        Returns:
            包含模型信息的字典，用于模板渲染
        """
        return {
            "model_name": self.config.model_name,
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "temperature": self.config.temperature,
        }

