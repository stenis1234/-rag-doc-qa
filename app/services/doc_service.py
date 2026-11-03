"""文档的上传、查询、删除。

上传接口的关键设计：**立刻返回，索引在后台做**。
一个 50 页的 PDF，解析 + 向量化要几十秒，如果同步等，前端会转圈到超时。
所以这里只做「落盘 + 建任务」，解析交给后台任务，进度靠 job 接口轮询。
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.loader import is_supported, supported_extensions
from app.models.orm import Chunk, Document, IngestionJob
from app.services.kb_service import get_kb

log = get_logger(__name__)


def save_upload(
    db: Session, kb_id: int, filename: str, content: bytes
) -> tuple[Document, IngestionJob]:
    """保存上传的文件并创建索引任务。

    幂等：同一个知识库里，内容完全相同（sha256 一致）且已索引成功的文档会被拒绝。
    这能挡住「手滑点了两次上传」和「同一份资料换个名字又传一遍」两种情况。
    """
    get_kb(db, kb_id)  # 不存在会抛 KeyError

    suffix = Path(filename).suffix.lower()
    if not is_supported(suffix):
        raise ValueError(
            f"不支持的格式 {suffix}，目前支持：{', '.join(supported_extensions())}"
        )
    if not content:
        raise ValueError("文件内容为空")

    file_hash = hashlib.sha256(content).hexdigest()
    duplicate = (
        db.query(Document)
        .filter(
            Document.kb_id == kb_id,
            Document.file_hash == file_hash,
            Document.status == "indexed",
        )
        .first()
    )
    if duplicate:
        raise ValueError(f"相同内容的文档已存在（文档 id={duplicate.id}）")

    doc = Document(
        kb_id=kb_id,
        filename=filename,
        stored_path="",  # 拿到 id 之后再定路径
        file_hash=file_hash,
        size_bytes=len(content),
        status="pending",
    )
    db.add(doc)
    db.flush()  # 拿到自增 id，但先不提交

    # 用 id 命名，避免重名文件互相覆盖；保留扩展名让解析器能判断类型
    path = settings.upload_dir / f"{doc.id}{suffix}"
    path.write_bytes(content)
    doc.stored_path = str(path)

    job = IngestionJob(
        id=str(uuid.uuid4()),
        doc_id=doc.id,
        kb_id=kb_id,
        status="pending",
        progress=0,
        stage="排队中",
    )
    db.add(job)
    db.commit()
    db.refresh(doc)
    db.refresh(job)
    log.info("上传成功 doc=%s job=%s file=%s", doc.id, job.id, filename)
    return doc, job


def list_documents(db: Session, kb_id: int) -> list[Document]:
    return (
        db.query(Document)
        .filter(Document.kb_id == kb_id)
        .order_by(Document.id.desc())
        .all()
    )


def delete_document(db: Session, doc_id: int) -> None:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise KeyError(f"文档 {doc_id} 不存在")

    # 顺序：先删向量，再删数据库记录，最后删文件（原因见 kb_service 的说明）
    from app.indexing.milvus_store import MilvusStore

    MilvusStore().delete_by_doc(str(doc_id))

    db.query(Chunk).filter(Chunk.doc_id == doc_id).delete(synchronize_session=False)
    db.query(IngestionJob).filter(IngestionJob.doc_id == doc_id).delete(
        synchronize_session=False
    )
    path = Path(doc.stored_path)
    db.delete(doc)
    db.commit()

    if path.exists():
        path.unlink(missing_ok=True)
    log.info("文档 %s 已删除", doc_id)


def get_job(db: Session, job_id: str) -> IngestionJob:
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise KeyError(f"任务 {job_id} 不存在")
    return job


def get_document(db: Session, doc_id: int) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise KeyError(f"文档 {doc_id} 不存在")
    return doc


def document_pages(db: Session, doc_id: int) -> list[dict]:
    """把 chunk 按页拼回「一页一段文本」，供前端预览。

    为什么从 chunk 拼、而不是重新解析一遍文件？
        ① 重新解析很慢（PDF 走 MinerU 要几十秒），用户点一下不该等这么久
        ② 用户真正想看的是「系统读到了什么」，而不是文件长什么样——
           PDF 解析出乱码时，看原文件是看不出来的，看这里才看得出来
        ③ chunk 已经是清洗过的文本，比原始抽取结果更接近模型看到的输入

    参数：
        doc_id  文档 id
    返回：
        [{"page_no": 3, "text": "..."}, ...]，按页码升序。
        page_no=0 的（md/docx）排最前面。
    """
    chunks = (
        db.query(Chunk).filter(Chunk.doc_id == doc_id).order_by(Chunk.seq).all()
    )

    # 用 dict 聚合：key 是页码，value 是该页所有 chunk 的正文。
    # dict 在 Python 3.7+ 保持插入顺序，所以后面排序只是为了页码乱序的情况。
    pages: dict[int, list[str]] = {}
    for chunk in chunks:
        pages.setdefault(chunk.page_no, []).append(chunk.content)

    return [
        {"page_no": page_no, "text": "\n\n".join(texts)}
        for page_no, texts in sorted(pages.items())
    ]


def count_chunks(db: Session, doc_id: int) -> int:
    """这份文档被切成了几块。数量太少通常意味着切分配置有问题。"""
    return db.query(Chunk).filter(Chunk.doc_id == doc_id).count()


def document_file(db: Session, doc_id: int) -> Path:
    """取原始文件路径（给「打开原文件」用）。

    单独一个函数而不是让 api 层直接读 doc.stored_path，
    是为了把「文件可能已经不存在」这个判断收敛在一处。
    """
    doc = get_document(db, doc_id)
    path = Path(doc.stored_path)
    if not path.exists():
        raise FileNotFoundError(f"原始文件已丢失：{doc.filename}")
    return path
