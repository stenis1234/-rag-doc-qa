"""日志。

用标准库 logging，不引第三方：新手项目没必要上 structlog。
每个 logger 名字带模块路径，出问题一眼能定位到文件。
"""

from __future__ import annotations

import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    """进程启动时调一次（main.py 里调用）。"""
    # Windows 控制台默认是 GBK，中文日志会变乱码，emoji 还会直接抛 UnicodeEncodeError。
    # 这里强制把标准输出切成 UTF-8；改不动（比如被重定向到管道）就算了。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    level = logging.DEBUG if settings.debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # 第三方库的日志太吵，单独压一下
    # dashscope 在 DEBUG 级别会把整个请求体（含所有 chunk 正文）打到日志里，必须关掉
    for noisy in ("httpx", "httpcore", "urllib3", "pymilvus", "dashscope", "dashscope.api", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
