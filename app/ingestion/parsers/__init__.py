"""各种文档格式的解析器。

统一入口是 app.ingestion.loader.load()，不要直接 import 具体解析器。
数据结构见 app.ingestion.parsers.base。
"""

from app.ingestion.parsers.base import Page, ParseResult

__all__ = ["Page", "ParseResult"]
