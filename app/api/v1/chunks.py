"""按 chunk_id 取原文片段。

前端点引用角标时调它，用来展示「这段话在原文里到底长什么样」。
M1 阶段先返回这一个 chunk；M2 可以扩成「返回前后各一个 chunk」，
让用户看到上下文（seq 字段就是为这个留的）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.models.orm import Chunk, Document

router = APIRouter(prefix="/chunks", tags=["片段"])


@router.get("/{chunk_id}", summary="取 chunk 原文")
def get_chunk(chunk_id: str, db: DbSession) -> dict:
    chunk: Chunk | None = db.get(Chunk, chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"chunk {chunk_id} 不存在")

    doc: Document | None = db.get(Document, chunk.doc_id)
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_name": doc.filename if doc else "",
        "title": chunk.title,
        "page_no": chunk.page_no,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "content": chunk.content,
    }
