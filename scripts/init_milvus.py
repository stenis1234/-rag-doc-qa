"""初始化 Milvus collection。

第一次部署、或者改了 schema 之后跑一次：
    .venv\\Scripts\\python.exe scripts\\init_milvus.py

带 --drop 会先删掉重建（**会丢数据**，只在你确定要重建索引时用）：
    .venv\\Scripts\\python.exe scripts\\init_milvus.py --drop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.core.milvus import ping  # noqa: E402
from app.indexing.milvus_store import MilvusStore  # noqa: E402

log = get_logger("init_milvus")


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化 Milvus collection")
    parser.add_argument("--drop", action="store_true", help="先删除已有 collection（会丢数据）")
    args = parser.parse_args()

    setup_logging()

    if not ping():
        log.error("连不上 Milvus：%s", settings.milvus_uri)
        log.error("请确认 Milvus 已启动，或检查 .env 里的 MILVUS_URI")
        return 1

    store = MilvusStore()
    log.info("目标 collection：%s（维度 %d）", settings.collection, settings.embed_dim)
    store.create(drop=args.drop)
    log.info("当前行数：%d", store.count())
    log.info("完成。接下来可以启动服务：uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
