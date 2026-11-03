"""文档接口：上传、列表、删除。

上传接口是全项目唯一需要「后台任务」的地方：
    POST 立刻返回 job_id，真正的解析和向量化在 BackgroundTasks 里跑。

M2 会把 BackgroundTasks 换成 Celery，好处是：
    - 进程重启不会丢任务
    - 可以限制并发数（避免同时索引 10 个 PDF 把 embedding 额度打满）
接口契约不用改，这就是为什么一开始就把 job_id 暴露出去。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import DbSession
from app.ingestion.pipeline import ingest_document
from app.models.schemas import DocumentDetail, DocumentOut, DocumentPage, UploadResult
from app.services import doc_service

router = APIRouter(prefix="/kb", tags=["文档"])

# 不同后缀对应的 Content-Type。
# 为什么要手动指定？
#   FastAPI 的 FileResponse 会根据文件路径猜，但 .md / .markdown 猜不出来，
#   会退化成 application/octet-stream → 浏览器直接下载，而不是打开预览。
_TEXT_TYPES = {**{".md": "text/plain; charset=utf-8", ".markdown": "text/plain; charset=utf-8",
                 ".txt": "text/plain; charset=utf-8", ".html": "text/html; charset=utf-8",
                 ".htm": "text/html; charset=utf-8"},
               **{".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".bmp": "image/bmp", ".webp": "image/webp"}}


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _TEXT_TYPES:
        return _TEXT_TYPES[suffix]
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix == ".pptx":
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if suffix == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"

# db: 作用是对数据库进行各种操作
@router.post("/{kb_id}/documents", response_model=UploadResult, summary="上传文档")
async def upload_document(
    kb_id: int,
    background_tasks: BackgroundTasks,
    db: DbSession,
    file: UploadFile = File(..., description="支持 pdf/docx/md/txt"),
) -> UploadResult:
    content = await file.read()
    try:
        doc, job = doc_service.save_upload(db, kb_id, file.filename or "unnamed", content)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 把「索引这份文档」丢到后台执行，接口立刻返回。
    #
    # add_task(要执行的函数, 参数1, 参数2) 等价于登记一次调用：
    #     ingest_document(doc.id, job.id)
    # 但**不是现在执行**——等这次 HTTP 响应发送完毕，FastAPI 才在线程池里跑它。
    #
    # ingest_document 干的事（见 app/ingestion/pipeline.py）：
    #     解析文件 → 清洗 → 切分 → 向量化 → 写 Milvus → 写关系库副本
    #     每完成一步就更新 job 的 status / progress / stage，前端据此画进度条。
    # 这一步通常要几秒到几分钟，不能卡在请求里，否则浏览器会超时。
    #
    # 为什么只传 doc.id 和 job.id 这两个整数：
    #     不能传 doc 对象或 db —— 请求一结束，session 就被关闭了，后台再用会报错；
    #     后台任务内部会自己开一个 SessionLocal()，按 id 重新查。
    #
    # 前端拿到返回的 job_id 后，每秒请求一次 /api/v1/jobs/{job_id}，直到 status 变成 done。
    background_tasks.add_task(ingest_document, doc.id, job.id)

    return UploadResult(
        doc_id=doc.id, job_id=job.id, filename=doc.filename, status=doc.status
    )


@router.get("/{kb_id}/documents", response_model=list[DocumentOut], summary="文档列表")
def list_documents(kb_id: int, db: DbSession) -> list[DocumentOut]:
    return [DocumentOut.model_validate(d) for d in doc_service.list_documents(db, kb_id)]


@router.get("/documents/{doc_id}", response_model=DocumentDetail, summary="文档详情（含解析后文本）")
def get_document(doc_id: int, db: DbSession) -> DocumentDetail:
    """点文档名时调用：返回文档信息 + 按页拼回的文本。

    为什么要把 pages 一起返回，而不是让前端再请求一次？
        一次请求就够，前端少一次往返，弹窗打开更快。
    """
    try:
        doc = doc_service.get_document(db, doc_id)
        pages = doc_service.document_pages(db, doc_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return DocumentDetail(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        page_count=doc.page_count,
        # 注意是 chunk 数（切了几个块），不是页数
        chunk_count=doc_service.count_chunks(db, doc_id),
        size_bytes=doc.size_bytes,
        error=doc.error,
        pages=[DocumentPage(**p) for p in pages],
    )


@router.get("/documents/{doc_id}/file", summary="打开原始文件")
def open_document_file(doc_id: int, db: DbSession, download: bool = False):
    """把原始文件直接返回给浏览器。

    参数：
        download=True  强制下载（带文件名）；
        download=False 让浏览器自己决定（PDF 会在标签页里直接预览）
    """
    try:
        path = doc_service.document_file(db, doc_id)
        doc = doc_service.get_document(db, doc_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc

    # 关键点：不传 filename 时 FastAPI 不加 Content-Disposition，
    # 浏览器就会「内联打开」（PDF 直接预览）；传了 filename 就会变成下载。
    if download:
        return FileResponse(path, filename=doc.filename, media_type=_media_type(path))
    return FileResponse(path, media_type=_media_type(path))


@router.delete("/documents/{doc_id}", summary="删除文档")
def delete_document(doc_id: int, db: DbSession) -> dict:
    try:
        doc_service.delete_document(db, doc_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}
