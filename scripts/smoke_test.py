"""检索层的冒烟测试：不需要任何 API Key，就能验证 Milvus 那一层是通的。

跑法：
    .venv\\Scripts\\python.exe scripts\\smoke_test.py

它会：
    1. 建一个临时 collection（名字带 _smoke 后缀，不影响正式数据）
    2. 塞 3 条假数据（向量是随机数，够用——这里测的是「写得进、搜得出」，不是准不准）
    3. 跑一次 hybrid_search
    4. 删掉临时 collection

如果这一步过不了，先别调 LLM，问题一定在 Milvus 连接 / schema / 版本上。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# 让脚本能直接 python scripts/xxx.py 运行（把项目根加进 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.indexing.milvus_store import ChunkRow, MilvusStore  # noqa: E402

log = get_logger("smoke")


def main() -> int:
    setup_logging()

    collection = f"{settings.collection}_smoke"
    store = MilvusStore(collection=collection)

    log.info("1/4 重建临时 collection: %s", collection)
    store.drop()
    store.create()

    log.info("2/4 写入 3 条假数据")
    rows = [
        ChunkRow(
            chunk_id="1_00000",
            doc_id="1",
            kb_id="1",
            content="Transformer 是一种基于自注意力机制的神经网络结构。",
            dense=[random.random() for _ in range(settings.embed_dim)],
            title="二、模型结构",
            page_no=3,
            seq=0,
            char_start=0,
            char_end=30,
        ),
        ChunkRow(
            chunk_id="1_00001",
            doc_id="1",
            kb_id="1",
            content="AlexNet 在 2012 年的 ImageNet 竞赛中夺冠，标志着深度学习的复兴。",
            dense=[random.random() for _ in range(settings.embed_dim)],
            title="四、深度学习",
            page_no=7,
            seq=1,
            char_start=30,
            char_end=80,
        ),
        ChunkRow(
            chunk_id="2_00000",
            doc_id="2",
            kb_id="2",
            content="BM25 是一种经典的稀疏检索算法，按词频和逆文档频率打分。",
            dense=[random.random() for _ in range(settings.embed_dim)],
            title="三、检索算法",
            page_no=1,
            seq=0,
            char_start=0,
            char_end=40,
        ),
    ]
    store.upsert(rows)

    log.info("3/4 检索（限定 kb_id=1，应该只返回前两条）")
    hits = store.hybrid_search(
        query_text="Transformer 自注意力",
        query_vector=[random.random() for _ in range(settings.embed_dim)],
        kb_ids=["1"],
        top_k_dense=5,
        top_k_sparse=5,
        top_k=5,
    )
    for hit in hits:
        log.info("  hit kb=%s doc=%s page=%s score=%.4f | %s", hit.kb_id, hit.doc_id, hit.page_no, hit.score, hit.content[:30])

    log.info("4/4 清理")
    store.drop()

    ok = len(hits) == 2 and all(h.kb_id == "1" for h in hits)
    if ok:
        log.info("[OK] Milvus 层正常：写入、混合检索、标量过滤都通")
        return 0
    log.error("[FAIL] 结果不符合预期（期望 2 条且都来自 kb_id=1，实际 %d 条）", len(hits))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
