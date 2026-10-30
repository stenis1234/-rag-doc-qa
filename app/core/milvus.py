"""Milvus 连接管理。

只负责「连上」，不负责 schema 和查询——那是 indexing/milvus_store.py 的事。
分开的原因：连接是基础设施，schema 是业务，混在同一个文件里以后很难复用。
"""

from __future__ import annotations

from pymilvus import MilvusClient

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    """进程内单例。MilvusClient 内部有连接池，反复 new 会浪费连接。"""
    global _client
    if _client is None:
        kwargs: dict = {"uri": settings.milvus_uri}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        log.info("连接 Milvus: %s", settings.milvus_uri)
        _client = MilvusClient(**kwargs)
    return _client


def reset_client() -> None:
    """测试用：强制下次重新连接。"""
    global _client
    _client = None


def ping() -> bool:
    """健康检查。启动时调用，连不上就在日志里明确告诉你，而不是等第一次检索才炸。"""
    try:
        get_client().list_collections()
        return True
    except Exception as exc:  # noqa: BLE001 - 这里就是要兜住所有异常
        log.error("Milvus 连接失败：%s", exc)
        return False
