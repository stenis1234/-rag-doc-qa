"""解析层的公共数据结构。

所有解析器都输出「页」的列表，而不是一整个字符串。
原因：引用要精确到页码，只有保留页边界才能做到。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Page:
    """一页（或一个逻辑分节）的文本。

    page_no 从 1 开始；0 表示这份文档没有页码概念（比如 Markdown、docx）。
    """

    page_no: int
    text: str


@dataclass(slots=True)
class ParseResult:
    pages: list[Page] = field(default_factory=list)
    # 解析器名字，写进日志方便排查「为什么这个 PDF 解析出来是空的」
    parser: str = ""
    # 解析过程中的警告（比如「检测到扫描件，已跳过 OCR」）
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text_length(self) -> int:
        return sum(len(p.text) for p in self.pages)
