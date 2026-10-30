"""Markdown / 纯文本解析。

最简单也最可靠的一种：文件本身就是文本，直接按空行分段、保留标题，
切分器能识别 `#` 标题，所以标题路径和页码之外的元信息都是白送的。
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers.base import Page, ParseResult


def parse(path: Path) -> ParseResult:
    text = path.read_text(encoding="utf-8", errors="replace")
    # 整篇算一页：Markdown 没有页码，page_no=0 表示「无页码」
    return ParseResult(pages=[Page(page_no=0, text=text)], parser="markdown")
