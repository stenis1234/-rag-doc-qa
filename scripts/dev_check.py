"""本地链路自检：不花一分钱，把「解析 → 切分 → 写库 → 检索」跑一遍。

它做的是：
    1. 建一个临时 collection（rag_chunks_devcheck）和临时知识库
    2. **把 embedding 换成随机向量**（这样不需要 DASHSCOPE_API_KEY）
    3. 用 data/samples/ai.md 走完整的 ingestion pipeline
    4. 用同一个假 embedding 做一次检索，检查能不能召回
    5. 清理掉临时数据

它验证不了：
    - embedding / rerank / 生成的真实效果（需要 key）
    - 前端交互

跑法：
    .venv\\Scripts\\python.exe scripts\\dev_check.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.db import SessionLocal, init_db  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.indexing import embedding as embedding_module  # noqa: E402
from app.indexing.milvus_store import MilvusStore  # noqa: E402

log = get_logger("dev_check")

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "ai.md"


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """假向量：同样的文本给同样的向量，不同文本给不同向量。

    用文本 hash 当随机种子，保证「同一段文字每次向量一致」——
    虽然语义是假的，但「相同内容能互相匹配」这一点和真实 embedding 一致，
    足够验证链路的正确性。
    """
    vectors = []
    for text in texts:
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        vectors.append([rng.random() for _ in range(settings.embed_dim)])
    return vectors


def main() -> int:
    setup_logging()
    init_db()

    # 用临时 collection，避免污染正式数据
    settings.collection = f"{settings.collection}_devcheck"
    embedding_module.embedder.embed_documents = _fake_embed  # type: ignore[method-assign]
    embedding_module.embedder.embed_query = lambda t: _fake_embed([t])[0]  # type: ignore[method-assign]

    store = MilvusStore()
    store.create(drop=True)

    from app.services import doc_service, kb_service

    db = SessionLocal()
    kb = None
    try:
        kb = kb_service.create_kb(db, "devcheck")
        log.info("1/4 上传样本文档")
        content = SAMPLE.read_bytes()
        doc, job = doc_service.save_upload(db, kb.id, "ai.md", content)

        log.info("2/4 同步执行索引")
        from app.ingestion.pipeline import ingest_document

        ingest_document(doc.id, job.id)
        db.refresh(doc)
        db.refresh(job)
        log.info("   状态=%s 进度=%s%% 文档页数=%d", job.status, job.progress, doc.page_count)
        if job.status != "done":
            log.error("   索引失败：%s", job.error)
            return 1

        from app.models.orm import Chunk

        chunk_count = db.query(Chunk).filter(Chunk.doc_id == doc.id).count()
        log.info("   关系库里有 %d 个 chunk，Milvus 里有 %d 行", chunk_count, store.count())
        if chunk_count == 0:
            log.error("   没有切出任何 chunk")
            return 1

        log.info("3/4 检索「Transformer 的自注意力机制」")
        from app.retrieval.hybrid import retrieve

        hits = retrieve("Transformer 的自注意力机制", [kb.id])
        for hit in hits[:3]:
            log.info("   [%.4f] p%s %s", hit.score, hit.page_no, hit.content[:40].replace("\n", " "))
        if not hits:
            log.error("   没有召回任何内容")
            return 1

        log.info("4/4 清理")
        kb_service.delete_kb(db, kb.id)
        kb = None
        store.drop()

        log.info("[OK] 链路自检通过：解析 / 切分 / 写入 / 检索 / 删除 都正常")
        return 0
    finally:
        if kb is not None:
            try:
                kb_service.delete_kb(db, kb.id)
            except Exception:  # noqa: BLE001
                pass
            store.drop()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
