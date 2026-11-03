"""离线索引链路：文件 → 文本 → chunk → 向量 → Milvus。

被谁调用：app/api/v1/documents.py 上传成功后，用 BackgroundTasks 丢到后台执行
它调用：loader.load → cleaner.clean_text → splitter.split_text
        → embedding.embed_documents → milvus_store.upsert

流程是直线的（没有「根据结果决定下一步」的分支），所以用普通函数而不是 LangGraph。
幂等：chunk_id 是确定性的，且写入前按 doc_id 删干净，重复索引结果一致。
进度：每步 commit 一次，前端轮询 /api/v1/jobs/{id} 才能看到进度条在动。
"""

from __future__ import annotations

# ---- 标准库 ----
# Path: 处理文件路径
from pathlib import Path

# ---- 项目内部 ----
from app.core.config import settings

# SessionLocal: session 工厂。为什么不用 FastAPI 的 db 依赖？
#   因为这是在**后台任务**里跑的，此时 HTTP 请求早就返回了，
#   请求级的 session 已经被关闭。必须自己开一个，用完自己关。
from app.core.db import SessionLocal

from app.core.logging import get_logger

# embedder: 模块级单例，.embed_documents(texts) 返回向量列表
from app.indexing.embedding import embedder

# ChunkRow: 写入 Milvus 的一行（含 dense 向量）；MilvusStore: 向量库门面
from app.indexing.milvus_store import ChunkRow, MilvusStore

# loader.load(path) 根据文件后缀选解析器，返回 ParseResult（含 pages）
from app.ingestion import loader

# clean_text: 清洗文本（必须在切分之前，否则偏移量全错）
from app.ingestion.cleaner import clean_text

# split_text: 切分，返回 TextChunk 列表（带 char_start/char_end）
from app.ingestion.splitter import split_text

# 三个 ORM 模型：文档、chunk 镜像、索引任务
from app.models.orm import Chunk, Document, IngestionJob

log = get_logger(__name__)

# Milvus 里 content 字段的长度上限（和 indexing/milvus_store.py 的 CONTENT_MAX_LEN 对应）。
# 这里再写一遍是为了在写入前兜底截断，避免一条超长文本让整批写入失败。
MAX_CONTENT_LEN = 4096


def _update(
    db,
    doc: Document,
    job: IngestionJob,
    *,
    status: str | None = None,
    progress: int | None = None,
    stage: str | None = None,
    doc_status: str | None = None,
) -> None:
    """更新任务/文档状态并立刻提交。

    参数里的 `*` 是什么意思？
        它表示「后面的参数必须用关键字传参」，比如必须写 progress=20，
        不能写 _update(db, doc, job, "running", 20)。
        好处：调用处一眼能看出每个数字的含义，不会把 status 和 stage 传反。

    为什么每次都 commit？
        为了让前端轮询能看到进度。见文件头注释。

    参数：
        status      任务状态：pending / running / done / failed
        progress    0~100
        stage       当前阶段的文字描述（前端直接显示）
        doc_status  文档状态（和任务状态不完全一样：
                    任务 done 时文档是 indexed；任务 failed 时文档也是 failed）
    """
    # 更新任务状态,如果传了就更新
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = progress
    if stage is not None:
        job.stage = stage
    if doc_status is not None:
        doc.status = doc_status

    db.commit()

    # 打一条日志。用的是「日志库的 % 风格」，不是 f-string：
    #   %s   占位符，按顺序填后面的参数 → doc.id、进度、阶段
    #   %%   转义，输出一个字面的百分号（写一个 % 会被当成占位符，导致参数对不上）
    # 最终输出： [doc 1] 70% 写入 Milvus
    # 为什么不用 f-string：logging 的规则是「日志等级不够就不拼字符串」，
    # 用 % 风格可以把拼接推迟到真正要输出的时候，省掉无谓开销。
    log.info("[doc %s] %s%% %s", doc.id, job.progress, job.stage)


def ingest_document(doc_id: int, job_id: str) -> None:
    """把一个文档索引进向量库 —— 本模块的主函数。

    谁调用它：api/v1/documents.py 的 BackgroundTasks

    参数：
        doc_id  文档 id（document 表主键）
        job_id  任务 id（用于更新进度）

    返回：
        None。**注意：这里不往外抛异常**。
        所有错误都写进 job.error 和 document.error，让前端能显示原因。
        为什么？因为这是后台任务，异常抛出去只会打印在服务器日志里，
        用户在页面上只会看到进度条永远卡住，完全不知道发生了什么。

    五个阶段（对应下面的 1~5 段注释）：
        1. 解析：文件 → 每页文本
        2. 清洗 + 切分：文本 → chunk（带页码和字符偏移）
        3. 向量化：chunk 文本 → 向量
        4. 写 Milvus：向量 + 正文 + 元信息
        5. 写关系库：chunk 的镜像副本（主数据在这里）
    """
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        job = db.get(IngestionJob, job_id)
        if doc is None or job is None:
            log.error("文档或任务不存在：doc=%s job=%s", doc_id, job_id)
            return

        _update(db, doc, job, status="running", progress=5, stage="解析文件")

        # ==================================================================
        # 1. 解析
        # ==================================================================
        path = Path(doc.stored_path)
        if not path.exists():
            # 典型场景：文件被手工删了，但数据库里还有记录
            raise FileNotFoundError(f"原始文件不见了：{path}")

        result = loader.load(path)
        # 解析器的警告不致命（比如"第 3 页抽不出来"），记日志即可继续
        for warning in result.warnings:
            log.warning("[doc %s] %s", doc.id, warning)

        # 20 个字符都抽不出来，基本可以判定是扫描件（需要 OCR）或者空文件。
        # 这时候继续走下去只会往库里塞垃圾，不如早早失败并给出明确原因。
        if result.text_length < 20:
            raise ValueError(
                f"没有解析出有效文本（共 {result.text_length} 个字符）。"
                "常见原因：扫描件但 OCR 没生效、文件加密、或内容本来就是空的。"
                f"当前解析器是 {result.parser}；扫描件需要配置 MINERU_TOKEN。"
            )

        doc.page_count = result.page_count
        _update(db, doc, job, progress=20, stage="清洗与切分")

        # ==================================================================
        # 2. 清洗 + 切分
        # ==================================================================
        # 元组顺序：(页码, 标题路径, 正文, 起始偏移, 结束偏移)
        drafts: list[tuple[int, str, str, int, int]] = []

        for page in result.pages:
            # 清洗必须在切分之前！
            # 因为 split_text 算出来的偏移是相对「清洗后的文本」的，
            # 如果切完再清洗，偏移量和最终存的正文就对不上了，引用会跳错位置。
            text = clean_text(page.text)
            if not text:
                continue

            # 每页单独切分，这样每个 chunk 天然知道自己在第几页。
            # 代价是跨页的段落会被切开——这是可接受的，
            # 因为"页码准确"对引用的价值，大于"段落完整"。
            for chunk in split_text(
                text,
                max_chars=settings.chunk_size,
                min_chars=settings.chunk_min_size,
                overlap=settings.chunk_overlap,
            ):
                drafts.append(
                    (page.page_no, chunk.title, chunk.content, chunk.char_start, chunk.char_end)
                )

        if not drafts:
            raise ValueError(
                "切分后没有得到任何 chunk：可能文本全是空白，或 CHUNK_SIZE / CHUNK_MIN_SIZE 配置异常。"
                "可以先点文档名看“解析后的文本”确认内容是否正常。"
            )

        log.info("[doc %s] 切出 %d 个 chunk", doc.id, len(drafts))
        _update(db, doc, job, progress=35, stage="向量化")

        # ==================================================================
        # 3. 向量化
        # ==================================================================
        # 只把正文送去算向量（不带标题、不带页码），
        # 因为标题路径里有很多重复词（每个块都有"四、深度学习"），
        # 会让不同块的向量变得过于相似，反而降低区分度。
        vectors = embedder.embed_documents([d[2] for d in drafts])

        _update(db, doc, job, progress=70, stage="写入 Milvus")

        # ==================================================================
        # 4. 写 Milvus
        # ==================================================================
        # 先删旧数据：这是「重新索引」能正确覆盖的关键。
        # 不做这一步的话，文档改版后旧内容会一直能被检索到——
        # 这是 RAG 项目里最常见的脏数据问题，而且很难发现
        # （因为检索结果看起来"有内容"，只是内容是过期的）。
        store = MilvusStore()
        store.delete_by_doc(str(doc.id))

        rows: list[ChunkRow] = []
        # zip(drafts, vectors) 把「chunk 元信息」和「向量」一一配对。
        # enumerate 给 seq，用来生成确定性主键。
        for seq, ((page_no, title, content, s, e), vector) in enumerate(zip(drafts, vectors)):
            # 兜底截断：理论上 splitter 不会产出超长块，
            # 但如果配置被改成 chunk_size=5000，这里能避免整批写入失败
            if len(content) > MAX_CONTENT_LEN:
                content = content[:MAX_CONTENT_LEN]
                e = s + len(content)

            rows.append(
                ChunkRow(
                    # 确定性主键：同一个文档的第 seq 块，永远是同一个 id
                    chunk_id=f"{doc.id}_{seq:05d}",
                    doc_id=str(doc.id),
                    kb_id=str(doc.kb_id),
                    content=content,
                    dense=vector,
                    title=title[:500],  # 和 schema 的 TITLE_MAX_LEN=512 对齐
                    page_no=page_no,
                    seq=seq,
                    char_start=s,
                    char_end=e,
                )
            )
        store.upsert(rows)

        # ==================================================================
        # 5. 写关系库镜像
        # ==================================================================
        # 为什么 Milvus 里存了正文，还要在关系库里再存一份？
        #   ① 主数据必须在关系库：Milvus 只是"可重建的派生索引"，
        #      万一 Milvus 数据丢了，还能靠这里重建
        #   ② 关系库能做 Milvus 做不了的事：按文档统计块数、分页、join 文件名
        #   ③ 前端的引用详情接口直接查这张表，不用每次去问 Milvus
        db.query(Chunk).filter(Chunk.doc_id == doc.id).delete()
        db.add_all(
            [
                Chunk(
                    chunk_id=row.chunk_id,
                    doc_id=doc.id,
                    kb_id=doc.kb_id,
                    seq=row.seq,
                    content=row.content,
                    title=row.title,
                    page_no=row.page_no,
                    char_start=row.char_start,
                    char_end=row.char_end,
                )
                for row in rows
            ]
        )

        doc.status = "indexed"
        doc.error = None
        _update(db, doc, job, status="done", progress=100, stage="完成", doc_status="indexed")
        log.info("[doc %s] 索引完成，共 %d 块", doc.id, len(rows))

    except Exception as exc:  # noqa: BLE001
        # 后台任务绝不能把异常抛出去（抛了也没人接，只会让线程池打一行 traceback）。
        # 正确做法：记日志 + 把错误写进数据库，让前端能展示给用户。
        log.exception("[doc %s] 索引失败", doc_id)
        try:
            doc = db.get(Document, doc_id)
            job = db.get(IngestionJob, job_id)
            if doc:
                doc.status = "failed"
                # 截断到 1000 字：错误信息可能很长（带完整堆栈），
                # 塞进数据库字段和前端提示框都不合适
                doc.error = str(exc)[:1000]
            if job:
                job.status = "failed"
                job.error = str(exc)[:1000]
                job.stage = "失败"
            db.commit()
        except Exception:  # noqa: BLE001
            # 连错误都写不进去（比如数据库连接断了），只能回滚。
            # 这里不能再抛异常，否则会盖住原始错误。
            db.rollback()
    finally:
        # 必须关闭自己开的 session，否则连接池会被慢慢耗尽
        db.close()


def delete_document_vectors(doc_id: int) -> None:
    """删除文档时清掉它的向量。

    顺序很重要：**先删 Milvus，再删关系库**（调用方负责后者）。
    反过来的话，如果 Milvus 删除失败，关系库里已经没有记录了，
    你就再也找不到这些「幽灵 chunk」——它们还会被检索到，
    但点开引用显示不出任何内容。
    """
    MilvusStore().delete_by_doc(str(doc_id))
