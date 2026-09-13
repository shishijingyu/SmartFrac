"""插件注册表：新增数据格式 / 模型只需注册，不改现有代码（NFR-1/NFR-2）。"""
from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def deco(cls: type) -> type:
            self._items[name] = cls
            return cls
        return deco

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(f"未注册的项: {name}; 已注册: {list(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        return list(self._items)


# 全局注册表
PARSER_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
