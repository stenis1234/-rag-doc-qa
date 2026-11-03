"""PDF 解析：配了 MINERU_TOKEN 就走 MinerU，否则用 pypdf。

两种方式的差别：
    MinerU  版面还原、表格、扫描件 OCR 都能处理，输出 Markdown；
            但**没有页码**，引用只能定位到文档级。
    pypdf   按页抽取纯文本，版面信息基本没有，但**页码准确**，
            引用能精确到「第 7 页」。

注意：MinerU 失败时**直接抛错**，不再自动降级到 pypdf。
      想回到 pypdf，把 .env 里的 MINERU_TOKEN 留空即可。
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.parsers import mineru
from app.ingestion.parsers.base import Page, ParseResult

log = logging.getLogger(__name__)


def parse(path: Path) -> ParseResult:
    """解析 PDF。MinerU 优先；没配 token 才用 pypdf。"""
    if mineru.is_enabled():
        # 失败会抛 MineruError，由上传任务记进 job.error 展示给用户
        return mineru.parse(path)
    return _parse_with_pypdf(path)


def _parse_with_pypdf(path: Path) -> ParseResult:
    """按页抽文本。页码准确，但拿不到版面、表格和扫描件内容。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    warnings: list[str] = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - 单个坏页不该毁掉整份文档
            warnings.append(f"第 {i} 页抽取失败：{exc}")
            text = ""
        if text.strip():
            pages.append(Page(page_no=i, text=text))

    if not pages:
        warnings.append("没有抽到任何文本，可能是扫描件（需要 OCR）")

    return ParseResult(pages=pages, parser="pypdf", warnings=warnings)
