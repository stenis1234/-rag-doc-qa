"""Cross-Encoder 精排（调 DashScope gte-rerank-v2）。

==========================================================================================
这个文件在整个项目里的位置
==========================================================================================
    被谁调用：retrieval/hybrid.py::retrieve()（召回之后、返回之前）
    它调用谁：DashScope 的 TextReRank 接口
    输入：问题 + 14 条候选
    输出：重新排序后的前 5 条（score 被替换成 0~1 的相关性分数）

==========================================================================================
为什么需要精排？（召回已经能返回结果了，为什么还要多花一次钱）
==========================================================================================
    召回（向量/BM25）解决的是「别漏」，但排序往往不准。
    因为它比较的是「意思像不像」，而不是「这段话到底能不能回答这个问题」。

    Cross-Encoder 的做法是：把问题和候选文档拼在一起送进模型，
    让模型直接判断「这段文字能不能回答这个问题」。精度高得多。
    代价是慢（每个候选都要跑一次前向计算），
    所以只能对少量候选做，不能对全库做——这就是「两段式」检索的原因。

==========================================================================================
降级策略（这个文件里最重要的设计）
==========================================================================================
    重排接口挂了、额度用完了、超时了 —— 不应该让整个问答失败。
    直接退回融合排序的结果，并在日志里留痕。
    理由：精排是「锦上添花」，不是「雪中送炭」。
    为了一个可选优化让用户完全无法提问，是明显的错误取舍。
"""

from __future__ import annotations

# ---- 第三方库 ----
import dashscope  # 阿里云百炼 SDK，TextReRank.call 是精排接口
from tenacity import retry, stop_after_attempt, wait_exponential

# ---- 项目内部 ----
from app.core.config import settings
from app.core.logging import get_logger

# ChunkHit 只用于类型标注
from app.indexing.milvus_store import ChunkHit

log = get_logger(__name__)


@retry(
    # 精排只重试 2 次（比 embedding 少一次）：
    # 因为它不是关键路径，失败可以降级，没必要让用户多等
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _call_rerank(query: str, documents: list[str], top_n: int):
    """调一次精排接口。

    参数：
        query      用户问题（改写后的）
        documents  候选文档的正文列表
        top_n      要返回多少条

    返回：
        DashScope 的 results 列表，形如
        [{"index": 3, "relevance_score": 0.82}, {"index": 0, "relevance_score": 0.41}, ...]
        注意：index 是「原始候选列表里的下标」，不是排名。
    """
    dashscope.api_key = settings.dashscope_api_key

    # TextReRank.call 的参数名是固定的（不要写成 query_text / docs）
    resp = dashscope.TextReRank.call(
        model=settings.rerank_model,   # gte-rerank-v2
        query=query,
        documents=documents,           # 只传文本，不传对象
        top_n=top_n,
        return_documents=False,        # 不要在响应里带回原文（省流量，原文我们自己有）
    )

    if resp.status_code != 200:
        raise RuntimeError(f"rerank 失败 {resp.status_code}: {resp.message}")

    return resp.output["results"]


def rerank(query: str, hits: list[ChunkHit], top_n: int) -> list[ChunkHit]:
    """按相关性重新排序，返回前 top_n 条。

    谁调用它：retrieval/hybrid.py::retrieve()

    参数：
        query   改写后的问题
        hits    召回结果（未排序或粗排）
        top_n   返回几条

    返回：
        ChunkHit 列表（注意：是**原地修改**了 score 和 rerank_score，
        返回的列表元素和传入的是同一批对象）

    副作用说明：
        hit.score 会被替换成 rerank 分数（0~1）。
        这样上层判断「检索质量够不够」时，两种模式（开/关精排）
        用的是同一个量纲，阈值才好定。
    """
    if not hits:
        return []

    documents = [h.content for h in hits]

    try:
        results = _call_rerank(query, documents, min(top_n, len(documents)))
    except Exception as exc:  # noqa: BLE001 - 这里就是要兜住所有异常
        # 关键：降级而不是抛错。
        # 注意这里返回的 hits 里 rerank_score 全是 None，
        # retrieval/hybrid.py 会据此判断"精排失败了"，然后自己做归一化。
        log.warning("重排失败，退回融合排序：%s", exc)
        return hits[:top_n]

    ordered: list[ChunkHit] = []
    for item in results:
        idx = int(item["index"])

        # 防御：服务端偶尔会返回越界下标，能挡住就挡住
        if 0 <= idx < len(hits):
            hit = hits[idx]
            hit.rerank_score = float(item.get("relevance_score", 0.0))
            # 覆盖 score：之后的逻辑（阈值判断、前端展示）都用这个分
            hit.score = hit.rerank_score
            ordered.append(hit)

    # 理论上服务端已经按分数降序返回了，这里再排一次：
    # 不依赖外部服务的顺序保证，是防御性编程的基本习惯。
    ordered.sort(key=lambda h: h.score, reverse=True)
    return ordered[:top_n]
