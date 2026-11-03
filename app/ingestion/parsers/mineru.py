"""MinerU 解析：把文件交给 MinerU，拿回整篇 Markdown。

本项目的统一入口（PDF / DOCX 都走它）：
    - 擅长版面还原、表格、扫描件 OCR
    - 输出整篇 Markdown，**没有页码**，所以只算一页（page_no=0），引用只能到文档级
    - 失败直接抛 MineruError，不再自动降级 —— 由调用方决定怎么处理
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.ingestion.parsers.base import Page, ParseResult


class MineruError(RuntimeError):
    """MinerU 解析失败（未配置 token / 网络错误 / 返回空内容）。"""


def is_enabled() -> bool:
    """是否配置了 MINERU_TOKEN。"""
    return bool(settings.mineru_token)


def parse(path: Path) -> ParseResult:
    """调用 MinerU 解析一个文件。

    参数：
        path  文件路径（pdf / docx / pptx / xlsx / 图片等 MinerU 支持的格式）
    返回：
        ParseResult，只有一页（page_no=0），正文是 Markdown
    异常：
        MineruError  没有 token、调用失败、或返回空内容
    """
    if not settings.mineru_token:
        raise MineruError(
            "未配置 MINERU_TOKEN，无法使用 MinerU 解析。"
            "请在项目根目录的 .env 里填写 MINERU_TOKEN（参考 env.example），然后重启服务。"
        )

    # 延迟导入：只有真正用到时才加载 SDK（它依赖较多，导入较慢）
    from mineru import MinerU

    try:
        client = MinerU(token=settings.mineru_token)
        result = client.extract(str(path), ocr=True)
    except Exception as exc:  # noqa: BLE001 - 统一包装成 MineruError
        raise MineruError(f"MinerU 解析失败：{exc}") from exc

    markdown = getattr(result, "markdown", "") or ""
    if not markdown.strip():
        raise MineruError("MinerU 返回了空内容（文件损坏或格式不支持）")

    return ParseResult(pages=[Page(page_no=0, text=markdown)], parser="mineru")
