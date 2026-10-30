"""PDF 解析：优先 MinerU，兜底 pypdf。

两种模式的区别很重要：
    MinerU  擅长版面还原、表格、扫描件 OCR，输出 Markdown，但**会丢掉页码**
            （它给的是整篇 markdown），所以引用只能定位到文档级。
    pypdf   按页抽取纯文本，版面信息基本没有，但**页码是准的**，
            引用能精确到「第 7 页」。

本项目默认优先 MinerU（质量高），解析结果只有一页；
如果你更需要页码精确，把 .env 里的 MINERU_TOKEN 留空即可。
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import settings
from app.ingestion.parsers.base import Page, ParseResult

log = logging.getLogger(__name__)


def parse(path: Path) -> ParseResult:
    if settings.mineru_token:
        try:
            return _parse_with_mineru(path)
        except Exception as exc:  # noqa: BLE001
            # MinerU 挂了不应该让整个上传失败，降级到 pypdf 更稳妥
            log.warning("MinerU 解析失败（%s），降级到 pypdf", exc)
    return _parse_with_pypdf(path)


def _parse_with_mineru(path: Path) -> ParseResult:
    """走 MinerU 云端/本地 SDK。

    SDK 用法和你的学习笔记一致：
        from mineru import MinerU
        client = MinerU(token=...)
        result = client.extract(path, ocr=True)
        result.markdown
    """
    from mineru import MinerU  # 延迟导入：没装这个包时不影响其它格式

    client = MinerU(token=settings.mineru_token)
    result = client.extract(str(path), ocr=True)
    markdown = getattr(result, "markdown", "") or ""
    return ParseResult(
        pages=[Page(page_no=0, text=markdown)],
        parser="mineru",
        warnings=[] if markdown.strip() else ["MinerU 返回了空内容"],
    )


def _parse_with_pypdf(path: Path) -> ParseResult:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    warnings: list[str] = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - 单个坏页不应该毁掉整份文档
            warnings.append(f"第 {i} 页抽取失败：{exc}")
            text = ""
        if text.strip():
            pages.append(Page(page_no=i, text=text))

    if not pages:
        warnings.append("没有抽到任何文本，可能是扫描件（需要 OCR）")

    return ParseResult(pages=pages, parser="pypdf", warnings=warnings)
