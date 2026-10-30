"""知识库的增删查。

删除知识库是个「跨系统」操作，顺序不能错：
    1. 先删 Milvus 里的向量（按 kb_id 过滤）
    2. 再删关系库里的文档、chunk、会话
    3. 最后删原始文件

如果反过来先删关系库，Milvus 删除又失败了，你就永远找不到这些
「孤儿向量」了——它们还会被检索到，但点开引用显示不出任何内容。
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.indexing.milvus_store import MilvusStore
from app.models.orm import Chunk, Conversation, Document, IngestionJob, KnowledgeBase, Message

log = get_logger(__name__)


def create_kb(db: Session, name: str, description: str = "") -> KnowledgeBase:
    if db.query(KnowledgeBase).filter(KnowledgeBase.name == name).first():
        raise ValueError(f"知识库「{name}」已存在")
    kb = KnowledgeBase(name=name, description=description)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


def list_kbs(db: Session) -> list[KnowledgeBase]:
    return db.query(KnowledgeBase).order_by(KnowledgeBase.id).all()


def get_kb(db: Session, kb_id: int) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise KeyError(f"知识库 {kb_id} 不存在")
    return kb


def delete_kb(db: Session, kb_id: int) -> None:
    kb = get_kb(db, kb_id)

    # 1. 向量
    MilvusStore().delete_by_kb(str(kb_id))

    # 2. 原始文件 + 关系库记录
    docs = db.query(Document).filter(Document.kb_id == kb_id).all()
    for doc in docs:
        path = Path(doc.stored_path)
        if path.exists():
            path.unlink(missing_ok=True)
    doc_ids = [d.id for d in docs]

    if doc_ids:
        db.query(Chunk).filter(Chunk.doc_id.in_(doc_ids)).delete(synchronize_session=False)
        db.query(IngestionJob).filter(IngestionJob.doc_id.in_(doc_ids)).delete(
            synchronize_session=False
        )
    db.query(Document).filter(Document.kb_id == kb_id).delete(synchronize_session=False)

    # 会话要先删消息再删会话：message 有外键指向 conversation，
    # SQLite 开了 PRAGMA foreign_keys=ON，顺序错了会直接 IntegrityError。
    conv_ids = [
        row[0]
        for row in db.query(Conversation.id).filter(Conversation.kb_id == kb_id).all()
    ]
    if conv_ids:
        db.query(Message).filter(Message.conversation_id.in_(conv_ids)).delete(
            synchronize_session=False
        )
    db.query(Conversation).filter(Conversation.kb_id == kb_id).delete(synchronize_session=False)
    db.delete(kb)
    db.commit()
    log.info("知识库 %s 已删除", kb_id)
