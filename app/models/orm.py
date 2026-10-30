"""数据表定义。

一共六张表，对应一个文档问答系统绕不开的六个实体：

    KnowledgeBase  知识库（一组文档的集合，检索时的隔离单位）
    Document       原始文件（一行 = 一个上传的文件）
    Chunk          切分后的片段（Milvus 里的镜像，主数据在这里）
    Conversation   一次会话
    Message        会话里的一条消息（含引用信息）
    IngestionJob   索引任务（前端靠它显示进度）

为什么 chunk 要在关系库里再存一份？
    Milvus 只适合存「向量 + 少量标量字段」，做统计、join、分页很不方便。
    把正文和页码偏移存在这里，前端展示引用、后台重建索引都靠它。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(ForeignKey("knowledge_base.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    # 文件落盘的绝对路径；删除文档时要一起删掉
    stored_path: Mapped[str] = mapped_column(String(512))
    # 内容 sha256：用来做幂等——同一个文件重复上传不会重复建索引
    file_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    # pending / parsing / indexing / indexed / failed
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Chunk(Base):
    """chunk 的「主数据」副本。chunk_id 与 Milvus 主键一致。"""

    __tablename__ = "chunk"

    # 格式：{doc_id}_{seq:05d}，例如 12_00003。确定性 ID 让重复索引变成覆盖写
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("document.id"), index=True)
    kb_id: Mapped[int] = mapped_column(Integer, index=True)
    seq: Mapped[int] = mapped_column(Integer)  # 块在文档里的顺序，用来取前后文
    content: Mapped[str] = mapped_column(Text)
    # 标题路径，例如 "四、深度学习 > 4.2 CNN"。比只存 h2 好：同名标题不会混
    title: Mapped[str] = mapped_column(String(512), default="")
    page_no: Mapped[int] = mapped_column(Integer, default=0)   # 从 1 开始；0 表示无页码
    char_start: Mapped[int] = mapped_column(Integer, default=0)  # 页内字符偏移
    char_end: Mapped[int] = mapped_column(Integer, default=0)


class Conversation(Base):
    __tablename__ = "conversation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(String(128), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Message(Base):
    __tablename__ = "message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversation.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text)
    # 引用列表的 JSON 字符串。用 Text 而不是 JSON 类型，是为了 SQLite/PG 都能跑
    citations: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class IngestionJob(Base):
    __tablename__ = "ingestion_job"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    doc_id: Mapped[int] = mapped_column(ForeignKey("document.id"), index=True)
    kb_id: Mapped[int] = mapped_column(Integer, index=True)
    # pending / running / done / failed
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    stage: Mapped[str] = mapped_column(String(32), default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
