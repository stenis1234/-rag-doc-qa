"""数据表定义（SQLAlchemy ORM）。

每个类 = 一张表，每个类属性 = 一列。六张表：

    KnowledgeBase  知识库：文档的集合，也是检索时的隔离单位
    Document       文档：一行 = 一次上传的文件
    Chunk          切片：正文 + 标题 + 页码偏移（主数据，Milvus 里是它的索引副本）
    Conversation   会话：一次对话
    Message        消息：一问一答，助手消息带引用
    IngestionJob   索引任务：进度、阶段、失败原因

为什么 chunk 要在关系库里也存一份？
    Milvus 只擅长「向量 + 标量过滤」，做统计、join、分页很别扭。
    正文和偏移存这里，重建索引、展示引用都靠它。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class KnowledgeBase(Base):
    """知识库：一组文档的集合。

    检索时按 kb_id 过滤，所以它也是「数据隔离的边界」——
    问 A 库的问题不会检索到 B 库的文档。
    """

    __tablename__ = "knowledge_base"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 库名，全局唯一：重名会让用户分不清
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Document(Base):
    """文档：一行 = 一个上传的文件。

    只记录「文件本身」的信息，不记录处理进度（那在 IngestionJob 里）。
    status 是给列表页显示的最终结果。
    """

    __tablename__ = "document"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 外键：这份文档属于哪个知识库
    kb_id: Mapped[int] = mapped_column(ForeignKey("knowledge_base.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    # 文件落盘的绝对路径；删除文档时要把文件一起删掉
    stored_path: Mapped[str] = mapped_column(String(512))
    # 内容 sha256：用于判重，同一份文件重复上传会被拒绝
    file_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    # pending / parsing / indexing / indexed / failed
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    # 失败原因（成功时为 None），前端在文档详情里展示
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Chunk(Base):
    """切片：检索的最小单位。

    和 Milvus 里的记录一一对应（chunk_id 相同），但这里是**主数据**：
    Milvus 的数据丢了可以靠这张表重建。
    """

    __tablename__ = "chunk"

    # 确定性主键：{doc_id}_{seq:05d}，例如 12_00003。
    # 好处是重复索引同一文档时是覆盖写，不会产生重复数据。
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("document.id"), index=True)
    kb_id: Mapped[int] = mapped_column(Integer, index=True)
    # 块在文档内的顺序号，将来做「取前后文」靠它
    seq: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # 标题路径，例如 "四、深度学习 > 4.2 CNN"；只存 h2 会让同名标题混在一起
    title: Mapped[str] = mapped_column(String(512), default="")
    # 页码：从 1 开始；0 表示该文档没有页码概念（md / docx）
    page_no: Mapped[int] = mapped_column(Integer, default=0)
    # 页内字符偏移（左闭右开），用于引用跳转和原文高亮
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)


class Conversation(Base):
    """会话：一次连续的对话。

    换知识库会开新会话，避免上下文串库。
    """

    __tablename__ = "conversation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(Integer, index=True)
    # 默认用第一个问题前 30 个字做标题，方便在列表里辨认
    title: Mapped[str] = mapped_column(String(128), default="新会话")
    # 早期对话的摘要：消息超过阈值后自动生成，每次问答时注入 prompt
    summary: Mapped[str] = mapped_column(Text, default="")
    # 摘要已经覆盖到哪条 message.id；<= 这个 id 的消息不再重复摘要
    summary_until_id: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Message(Base):
    """消息：会话里的一问一答。

    role=user 是提问，role=assistant 是回答；
    只有助手消息的 citations 有内容（指向答案里的 [1][2]）。
    """

    __tablename__ = "message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversation.id"), index=True)
    # user / assistant
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    # 引用列表的 JSON 字符串。用 Text 而不是 JSON 类型，是为了 SQLite / PG 都能跑
    citations: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class IngestionJob(Base):
    """索引任务：一次「把文档写进向量库」的进度记录。

    后台任务每完成一步就更新它，前端轮询 /api/v1/jobs/{id} 画进度条。
    一个文档重新索引会新建一条，所以历史失败原因不会被覆盖。
    """

    __tablename__ = "ingestion_job"

    # uuid 而不是自增：这个 id 会暴露给前端，自增容易被人猜
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("document.id"), index=True)
    kb_id: Mapped[int] = mapped_column(Integer, index=True)
    # pending / running / done / failed
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    # 0~100，进度条用
    progress: Mapped[int] = mapped_column(Integer, default=0)
    # 当前在干什么：解析文件 / 清洗与切分 / 向量化 / 写入 Milvus
    stage: Mapped[str] = mapped_column(String(32), default="queued")
    # 失败原因；因为后台任务的异常传不回前端，只能存这里
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    # onupdate：每次 UPDATE 自动刷新，不用业务代码手写
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
