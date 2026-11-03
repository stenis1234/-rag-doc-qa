"""docx 解析：配了 MINERU_TOKEN 就走 MinerU，否则用 python-docx。

python-docx 的优点是能读到段落样式，把 Heading 1/2/3 还原成 Markdown 的 #/##/###，
切分器就能识别标题层级。缺点是表格排版会丢、图片里的文字读不到。

MinerU 对 docx 的版面还原更好，但同样**没有页码**。
失败直接抛 MineruError，不降级。
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers import mineru
from app.ingestion.parsers.base import Page, ParseResult


def parse(path: Path) -> ParseResult:
    """解析 docx。MinerU 优先；没配 token 才用 python-docx。"""
    if mineru.is_enabled():
        return mineru.parse(path)
    return _parse_with_python_docx(path)


def _parse_with_python_docx(path: Path) -> ParseResult:
    """读段落 + 表格，把 Heading 样式转成 Markdown 标题。"""
    from docx import Document as DocxDocument

    document = DocxDocument(str(path))
    lines: list[str] = []

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = (para.style.name or "") if para.style else ""
        if style.startswith("Heading"):
            # "Heading 2" → "##"
            try:
                level = int(style.split()[-1])
            except ValueError:
                level = 1
            lines.append(f"{'#' * min(level, 6)} {text}")
        else:
            lines.append(text)

    # 表格里往往是最关键的信息（参数表、对比表），必须抽出来
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")

    return ParseResult(
        pages=[Page(page_no=0, text="\n\n".join(lines))], parser="python-docx"
    )
