"""数据库层：连接池、会话工厂、建表。

Milvus 只存向量和检索文本；业务数据（知识库/文档/chunk 副本/会话/任务）在这里。
chunk 以关系库为主数据，Milvus 里的索引可以随时重建。
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

# SQLite 默认禁止跨线程复用连接，而 FastAPI 的同步接口和后台任务跑在线程池里
_connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,  # 借连接前探活，避免空闲被断开后的首个请求报错
    echo=False,
)


if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_pragma(dbapi_conn, _record):
        """每条新连接建立时执行一次。"""
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")  # 读写并发，写不阻塞读
        # SQLite 默认不检查外键，不开这条就会产生孤儿数据而没有任何报错
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# autoflush=False：写入时机完全由 commit 控制
# expire_on_commit=False：commit 后对象属性仍可读，返回响应时不必再查库
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类，维护「类 → 表」的注册表。"""


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：每个请求一个 session，请求结束自动关闭。

    后台任务不能用它（那时请求已结束、session 已关闭），要自己 SessionLocal()。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """建表。启动时调一次。

    注意 create_all 只建缺失的表，不会修改已存在的表结构；改字段要删库重建或上 Alembic。
    """
    # 必须 import：模型类只有被执行到定义时才会注册进 Base.metadata
    from app.models import orm  # noqa: F401

    settings.ensure_dirs()
    Base.metadata.create_all(bind=engine)
    _ensure_columns()


def _ensure_columns() -> None:
    """给已存在的表补新列。

    create_all 只建缺失的表，不会给老表加列；表已经存在时新字段会静默缺失，
    直到查询时报 “no such column”。这里手工做一次轻量迁移；
    字段多了以后应该换成 Alembic。
    """
    from sqlalchemy import inspect, text

    wanted = {
        "conversation": [
            ("summary", "TEXT DEFAULT ''"),
            ("summary_until_id", "INTEGER DEFAULT 0"),
        ],
    }

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in wanted.items():
            if table not in tables:
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl in columns:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
