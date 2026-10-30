"""按扩展名分发到对应解析器。

新增格式只需要两步：
    1. 在 parsers/ 下加一个模块，实现 parse(path) -> ParseResult
    2. 在下面的 PARSERS 里注册扩展名
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers import docx, markdown, pdf
from app.ingestion.parsers.base import Page, ParseResult

PARSERS = {
    ".md": markdown.parse,
    ".markdown": markdown.parse,
    ".txt": markdown.parse,
    ".pdf": pdf.parse,
    ".docx": docx.parse,
}

SUPPORTED = sorted(PARSERS)


class UnsupportedFormat(ValueError):
    pass


def load(path: Path) -> ParseResult:
    """解析文件。调用方负责保证文件存在。"""
    suffix = path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise UnsupportedFormat(
            f"暂不支持 {suffix} 格式。目前支持：{', '.join(SUPPORTED)}"
        )
    return parser(path)
