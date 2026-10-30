"""在线检索的总入口：问题 → 一批候选 chunk。

==========================================================================================
这个文件在整个项目里的位置
==========================================================================================
    被谁调用：app/pipeline/rag.py::RAGWorkflow.run_stream() 的第 2、3 步
    它调用谁：
        indexing/embedding.py::embedder.embed_query()      问题 → 1024 维向量
        indexing/milvus_store.py::get_store().hybrid_search()  双路召回 + RRF 融合
        retrieval/rerank.py::rerank()                       精排（可选）

上层只调 `retrieve()` 一个函数，不关心底下用了几路索引、有没有精排。
这样以后想加第三路（比如「按标题匹配」），只要改这个文件，上层不用动。

==========================================================================================
为什么检索要分成「召回」和「精排」两段？
==========================================================================================
    召回（recall）：从几万条里快速挑出几十条「可能相关」的。要求快，允许不准。
    精排（rerank）：对这几十条逐条细算相关性。要求准，允许慢。

这是一个经典的精度/成本权衡：Cross-Encoder 精排一条要几十毫秒，
不可能对全库做；而向量检索快，但只比较「意思像不像」，经常把关键细节漏掉。
所以先用便宜的召回把范围缩小，再用贵的模型在小区间里挑。

==========================================================================================
为什么召回要两路（dense + sparse）？
==========================================================================================
    稠密（dense，向量）：擅长「意思相近但用词不同」。
        问「大模型怎么防止胡说」，能召回「LLM 幻觉抑制」——字面完全不同，但语义对。
    稀疏（sparse，BM25）：擅长「精确的词」。
        问「Transformer 的 QKV 是什么」，能精确命中含 "QKV" 的段落；
        而向量检索可能把 "QKV" 当成噪声，召回一堆泛泛而谈的段落。

两者失败的方式不一样，所以互补。只用其中一路，总有一类问题答不好。
"""

from __future__ import annotations

# ---- 项目内部依赖 ----
# settings：全局配置（TOP_K_*、SCORE_THRESHOLD、MIN_RERANK_SCORE 等都在里面）
from app.core.config import settings

# get_logger：标准库 logging 的封装，名字带模块路径，方便定位日志来源
from app.core.logging import get_logger

# embedder：Embedder 类的模块级单例（见 indexing/embedding.py）。
# 它的 .embed_query(text) 返回 1024 个 float，必须和写入 Milvus 时用的模型完全一致，
# 否则两边向量不在同一个语义空间，检索会「看起来正常但就是不准」。
from app.indexing.embedding import embedder

# ChunkHit：检索结果的 dataclass，字段见 indexing/milvus_store.py
# get_store：延迟创建的 MilvusStore 单例。
#   为什么是函数而不是模块级 `store = MilvusStore()`？
#   因为后者会在 import 时就连接 Milvus，Milvus 没起来的话连单元测试都跑不了。
from app.indexing.milvus_store import ChunkHit, get_store

# rerank：Cross-Encoder 精排函数（调 DashScope 的 gte-rerank-v2）
from app.retrieval.rerank import rerank

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# RRF 相关的两个常量
# ---------------------------------------------------------------------------
# RRF = Reciprocal Rank Fusion（倒数排名融合），是合并多路检索结果的标准做法。
#
# 公式：某条文档的总分 = Σ 1 / (k + 它在第 i 路里的名次)
#   k = 60 是论文里的经验值，作用是「压平」名次差距，避免第 1 名和第 2 名差太远。
#
# 为什么不用「分数加权求和」？
#   因为两路分数根本不是同一个量纲：cosine 相似度在 0~1，BM25 分数可能是 0~20。
#   加权求和要不停调权重，换个数据集就失效；RRF 只看名次，天然免疫量纲问题。
#
# _RRF_MAX 是理论满分：一条文档在两路里都排第 1，得分就是 1/(60+1) + 1/(60+1)。
_RRF_K = 60
_RRF_MAX = 2.0 / (_RRF_K + 1)


def _normalize(hits: list[ChunkHit]) -> None:
    """把 RRF 分数（0.03 量级）线性缩放到 0~1，方便和阈值比较。

    为什么要这个函数：
        `SCORE_THRESHOLD` 的默认值 0.15 是给「rerank 分数」定的（那个天然在 0~1）。
        但关闭重排、或重排失败降级时，score 是 RRF 分数，最大只有 0.0328，
        直接和 0.15 比会永远判成「检索失败」，每次都白跑一次重查。
        归一化之后，两种模式的分数就有了可比性。

    参数：
        hits  原地修改（dataclass 是可变对象，改的是同一个实例）
    返回：
        None（注意：不是返回新列表，调用方别写成 hits = _normalize(hits)）
    """
    for hit in hits:
        hit.score = min(1.0, hit.score / _RRF_MAX)


def retrieve(
    question: str,
    kb_ids: list[int],
    top_k: int | None = None,
) -> list[ChunkHit]:
    """检索与问题最相关的 chunk —— 本模块唯一的对外函数。

    谁调用它：pipeline/rag.py（问答主流程）、scripts/calibrate.py（阈值校准）

    参数：
        question  已经过指代消解的问题。
                  ⚠️ 不要传原始的多轮口头问题（比如「它呢？」），那样检索必然失败，
                  因为「它」没有语义。改写请走 retrieval/rewrite.py。
        kb_ids    允许检索的知识库 id 列表。
                  传空列表会直接返回空——这是有意的「拒绝」而不是「忽略条件」，
                  否则一旦上游算错，就会跨知识库检索，把别人的文档答给用户。
        top_k     最终保留几条；不传则用配置里的 TOP_K_FINAL（默认 5）

    返回：
        ChunkHit 列表，按相关性从高到低。
        每条都带：chunk_id、doc_id、content、title、page_no、char_start、char_end、score

    执行步骤（和下面代码的三段注释对应）：
        1. 问题 → 向量
        2. Milvus 双路召回 + RRF 融合（拿到 top_k_candidate 条候选）
        3. 精排（开启时），并按 min_rerank_score 过滤掉噪声
    """
    # 防御：空列表表示「一个库都不许查」。返回空会让上层走拒答分支，这是正确的行为。
    if not kb_ids:
        return []

    top_k = top_k or settings.top_k_final

    # ---------- 第 1 步：问题向量化 ----------
    # 注意这里用的是 embed_query 而不是 embed_documents：
    # 两者最终调的是同一个接口，分开命名只是为了让「查询」和「文档」在代码里可区分，
    # 将来如果要给查询加前缀（有些模型要求 query 加 "query: " 前缀），只改一处。
    query_vector = embedder.embed_query(question)

    # ---------- 第 2 步：双路召回 + 融合 ----------
    # kb_ids 要转成字符串：Milvus 里 kb_id 字段是 VARCHAR，
    # 而关系库里是 INTEGER。这个转换点必须在这里做，别在 milvus_store 里做——
    # 那里应当假设「传进来的就是 Milvus 认识的类型」。
    hits = get_store().hybrid_search(
        query_text=question,                    # 稀疏那一路直接用原始文本，Milvus 服务端负责分词
        query_vector=query_vector,              # 稠密那一路用向量
        kb_ids=[str(k) for k in kb_ids],
        top_k_dense=settings.top_k_dense,       # 稠密路召回多少（默认 10）
        top_k_sparse=settings.top_k_sparse,     # 稀疏路召回多少（默认 10）
        top_k=settings.top_k_candidate,         # 融合后保留多少（默认 14），要 >= 后面精排的条数
    )
    log.info("召回 %d 条候选（问题：%s）", len(hits), question[:50])

    if not hits:
        return []

    # ---------- 第 3 步：精排 ----------
    if settings.rerank_enabled:
        hits = rerank(question, hits, top_k)

        if hits and hits[0].rerank_score is None:
            # 这种情况说明重排失败了，rerank() 已经退回融合排序。
            # 此时 score 还是 RRF 量纲（0.03 左右），必须归一化，
            # 否则和阈值（0.15）没法比，会导致每次都被判成「检索失败」。
            _normalize(hits)
        else:
            # 重排成功：用 min_rerank_score 做「入场券」过滤。
            # 实测数据：无关问题的最高分约 0.006，同领域问题在 0.30 以上，
            # 所以 0.02 这条线能干净地把「完全无关」滤成空结果，
            # 让上层走拒答分支，而不是把 5 段噪声塞进 prompt 让模型硬编。
            hits = [h for h in hits if h.score >= settings.min_rerank_score]
    else:
        # 关闭精排：直接用融合后的前 top_k 条，同样要归一化
        hits = hits[:top_k]
        _normalize(hits)

    return hits
