"""知识库接口。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import DbSession
from app.models.schemas import KBCreate, KBOut
from app.services import kb_service

router = APIRouter(prefix="/kb", tags=["知识库"])


@router.post("", response_model=KBOut, summary="新建知识库")
def create_kb(payload: KBCreate, db: DbSession) -> KBOut:
    try:
        kb = kb_service.create_kb(db, payload.name, payload.description)
    except ValueError as exc:
        # 409 而不是 400：这是「资源冲突」，前端可以提示「换个名字」
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return KBOut.model_validate(kb)


@router.get("", response_model=list[KBOut], summary="知识库列表")
def list_kbs(db: DbSession) -> list[KBOut]:
    return [KBOut.model_validate(kb) for kb in kb_service.list_kbs(db)]


@router.delete("/{kb_id}", summary="删除知识库（连带文档、向量、会话）")
def delete_kb(kb_id: int, db: DbSession) -> dict:
    try:
        kb_service.delete_kb(db, kb_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}
