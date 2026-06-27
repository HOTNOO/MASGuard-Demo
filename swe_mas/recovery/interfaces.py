"""通用接口定义"""
from __future__ import annotations
from typing import Protocol, Any

class LLMClientProtocol(Protocol):
    def query(self, messages: list[dict[str,str]], **kwargs) -> dict[str, Any]:
        ...
