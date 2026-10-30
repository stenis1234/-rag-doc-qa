"""docx 解析。

python-docx 能读到段落样式，所以这里把 Heading 1/2/3 还原成 Markdown 的 #/##/###，
交给切分器就能识别出标题层级——不用为 docx 单独写一套结构识别逻辑。
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers.base import Page, ParseResult


def parse(path: Path) -> ParseResult:
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
