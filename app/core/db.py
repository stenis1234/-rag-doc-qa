"""业务数据库（SQLAlchemy 2.0）。

注意职责划分：
    Milvus     → 只管「向量 + 检索用的文本」
    SQLAlchemy → 管「知识库、文档、chunk 镜像、会话、任务」

chunk 在两边各存一份，SQLite/PG 里那份是**主数据**，Milvus 是可以随时重建的索引。
这条原则的意义：Milvus 挂了 / 想换库 / 想加一路检索，都只是重建索引，不会丢内容。
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

# SQLite 需要 check_same_thread=False：FastAPI 的后台任务在别的线程里跑，
# 会复用同一个连接池。PostgreSQL 不需要这个参数。
_connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,  # 连接断了自动重连，避免长时间空闲后第一个请求报错
    echo=False,
)


if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_pragma(dbapi_conn, _record):  # pragma: no cover - 只在 SQLite 下生效
        """SQLite 默认不开 WAL，读写并发会互相阻塞；打开后体验好很多。"""
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


def get_db() -> Iterator[Session]:
    """FastAPI 依赖注入用：每个请求一个 session，请求结束自动关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """建表。

    M0/M1 阶段直接用 create_all 够了；等表结构开始改，再引 Alembic 做迁移
    （create_all 不会改已存在的表，这是它唯一的硬伤）。
    """
    from app.models import orm  # noqa: F401  必须导入，否则 Base 里没有表定义

    settings.ensure_dirs()
    Base.metadata.create_all(bind=engine)
