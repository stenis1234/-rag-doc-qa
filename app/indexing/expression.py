"""Milvus 过滤表达式拼接。

单独抽出来、且不 import pymilvus，是为了能脱离环境做单元测试
（tests/test_expression.py 直接用 python -m unittest 跑）。

语法提醒：Milvus 的 expr 不是 SQL，字符串只能用双引号。
"""

from __future__ import annotations


def quote(value: str) -> str:
    """把值包成 Milvus 的字符串字面量，并转义可能在值里的双引号和反斜杠。

    文件名、标题里出现 `"` 是常事，不转义会直接语法错误。
    """
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_filter(
    kb_ids: list[str] | None = None,
    doc_ids: list[str] | None = None,
    extra: str | None = None,
) -> str:
    """拼出 `kb_id in [...] and doc_id in [...]` 这样的表达式。

    参数都是可选；传 None 表示不加这个条件。
    空列表会被当成「一个都不要」，直接返回一个恒假表达式——
    而不是悄悄忽略条件导致检索到别的库里去（这是很危险的一类 bug）。
    """
    parts: list[str] = []

    if kb_ids is not None:
        if not kb_ids:
            return "1 == 0"
        joined = ", ".join(quote(x) for x in kb_ids)
        parts.append(f"kb_id in [{joined}]")

    if doc_ids is not None:
        if not doc_ids:
            return "1 == 0"
        joined = ", ".join(quote(x) for x in doc_ids)
        parts.append(f"doc_id in [{joined}]")

    if extra:
        parts.append(f"({extra})")

    return " and ".join(parts)
