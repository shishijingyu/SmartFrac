"""Parser 插件基类与自动分发。"""
from __future__ import annotations

from pathlib import Path

from smartfrac.core.datatypes import AtomicSample
from smartfrac.core.registry import PARSER_REGISTRY


class DataParser:
    """所有格式解析器基类。子类用 @PARSER_REGISTRY.register(".xyz") 注册。"""

    ext: str = ""

    def parse(self, path: Path) -> list[AtomicSample]:
        raise NotImplementedError


def parse_file(path: str | Path) -> list[AtomicSample]:
    """按扩展名自动选择 parser。"""
    path = Path(path)
    ext = path.suffix.lower()
    parser_cls = PARSER_REGISTRY.get(ext)
    return parser_cls().parse(path)
