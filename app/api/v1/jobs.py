"""索引任务查询。

前端上传完文件后轮询这个接口画进度条。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import DbSession
from app.models.schemas import JobOut
from app.services import doc_service

router = APIRouter(prefix="/jobs", tags=["任务"])


@router.get("/{job_id}", response_model=JobOut, summary="查询索引进度")
def get_job(job_id: str, db: DbSession) -> JobOut:
    try:
        job = doc_service.get_job(db, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobOut.model_validate(job)
