"""按扩展名分发到对应解析器。

分三类：

    纯文本（md / txt）          直接读，不经过任何解析模型
    有本地兜底的（pdf / docx）  配了 MINERU_TOKEN 就走 MinerU，否则用 pypdf / python-docx
    只能靠 MinerU 的（ppt / xlsx / doc / 图片 / html）
                                必须先配 token，否则直接拒绝上传

MinerU 解析失败会抛 MineruError（不会降级），由上传任务写进 job.error。
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers import docx, markdown, mineru, pdf
from app.ingestion.parsers.base import ParseResult

# 纯文本：本身就带结构（# 标题），直接读最快
TEXT_PARSERS = {
    ".md": markdown.parse,
    ".markdown": markdown.parse,
    ".txt": markdown.parse,
}

# 有本地兜底的格式：没配 token 时用本地库解析
LOCAL_PARSERS = {
    ".pdf": pdf.parse,
    ".docx": docx.parse,
}

# 只有 MinerU 才能处理的格式（配了 token 才允许上传）
MINERU_ONLY = {
    ".doc",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".html",
    ".htm",
}


class UnsupportedFormat(ValueError):
    """继承 ValueError，上层 except ValueError 能接住并转成 HTTP 400。"""


def supported_extensions() -> list[str]:
    """当前能上传的全部后缀（给错误提示和前端用）。"""
    return sorted(set(TEXT_PARSERS) | set(LOCAL_PARSERS) | MINERU_ONLY)


def is_supported(suffix: str) -> bool:
    """后缀是否可上传（大小写不敏感）。"""
    s = suffix.lower()
    return s in TEXT_PARSERS or s in LOCAL_PARSERS or s in MINERU_ONLY


def load(path: Path) -> ParseResult:
    """解析文件。调用方负责保证文件存在（上传流程刚写完盘，不必再查一次）。"""
    suffix = path.suffix.lower()

    if suffix in TEXT_PARSERS:
        return TEXT_PARSERS[suffix](path)

    if suffix in LOCAL_PARSERS:
        # 内部自己判断走 MinerU 还是本地库
        return LOCAL_PARSERS[suffix](path)

    if suffix in MINERU_ONLY:
        if not mineru.is_enabled():
            raise UnsupportedFormat(
                f"{suffix} 只能靠 MinerU 解析，但当前没有配置 MINERU_TOKEN。\n"
                "请在项目根目录的 .env 里加上 MINERU_TOKEN=...（参考 env.example），然后重启服务再上传。"
            )
        return mineru.parse(path)

    raise UnsupportedFormat(
        f"暂不支持 {suffix} 格式。目前支持：{', '.join(supported_extensions())}"
    )
