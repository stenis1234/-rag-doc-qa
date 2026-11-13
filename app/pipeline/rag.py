"""在线问答主流程：改写 → 检索 → 评估 → 生成 → 引用 → 校验。

被谁调用：app/services/chat_service.py::ask_stream()
它调用：retrieval/rewrite、retrieval/hybrid、generation/prompt、generation/llm、generation/citation

离线建库不在这里（见 app/ingestion/pipeline.py）。两段共用同一个向量模型。
流程只有「分数低就重查一次」这一个循环，所以用 while 而不是 LangGraph。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.logging import get_logger
from app.generation.citation import (
    build_citations,
    extract_keywords,
    extract_markers,
    is_grounded,
)
from app.generation.llm import stream_text
from app.generation.prompt import REFUSAL_TEXT, build_messages
from app.indexing.milvus_store import ChunkHit
from app.models.orm import Document
from app.retrieval.hybrid import retrieve
from app.retrieval.rewrite import condense_question, expand_query

log = get_logger(__name__)


@dataclass(slots=True)
class RagEvent:
    """一个流式事件，API 层会塞进 SSE 的 data 字段。

    type 取值：meta / token / citations / warning / done / error
    default_factory=dict表示默认值是空字典
    """

    type: str
    data: dict = field(default_factory=dict)


class RAGWorkflow:
    """问答流程。无状态，每次问答 new 一个（不要在上面存会话数据）。"""

    def run_stream(
        self,
        question: str,
        kb_ids: list[int],
        history: list[tuple[str, str]] | None = None,
        summary: str = "",
    ) -> Iterator[RagEvent]:
        #Iterator（迭代器）就是“能一个一个往外取东西的对象”
        """执行一次问答，边算边 yield。

        参数：
            question  用户原始问题
            kb_ids    允许检索的知识库（空列表直接拒答，防止跨库泄漏）
            history   最近几轮对话 [(role, content)]

        是生成器：调用不会立刻执行，API 层开始迭代时才真正跑。
        """
        history = history or []
        attempts = 0

        # STEP 1 改写：多轮里的「它」没有语义，直接用去检索必然失准。
        # history 为空时 condense_question 直接返回原问题，不调模型。
        query = condense_question(question, history)

        # STEP 2 检索：转向量 → Milvus 双路召回 → 精排，结果按分数降序
        hits = retrieve(query, kb_ids)

        # STEP 3 评估 + 重查：分数太低说明用词和文档对不上，补同义词再查一次
        while self._is_weak(hits) and attempts < settings.max_retry:
            attempts += 1
            log.info("检索结果偏弱（%d 条），第 %d 次扩写重查", len(hits), attempts)
            query = expand_query(query)
            hits = retrieve(query, kb_ids)

        yield RagEvent(
            "meta",
            {"query": query, "hit_count": len(hits), "retries": attempts},
        )

        # 一条都没召回就拒答。硬把空结果给模型，它一定会用自身知识编。
        if not hits:
            log.info("没有召回任何内容，返回拒答")
            yield RagEvent("token", {"text": REFUSAL_TEXT})
            yield RagEvent("citations", {"items": []})
            yield RagEvent("done", {"grounded": False, "query": query})
            return

        # STEP 4 生成：Milvus 只存 doc_id，先补文件名给 prompt 和前端用
        self._attach_doc_names(hits)
        messages = build_messages(question, hits, history, summary)

        # 用原始 question 而不是改写后的 query：回答要针对用户真正问的那句话
        parts: list[str] = []
        for piece in stream_text(messages):
            parts.append(piece)
            yield RagEvent("token", {"text": piece})

        answer = "".join(parts).strip()

        # STEP 5 引用：把 [1] 翻译成 chunk + 页码 + 字符偏移
        markers = extract_markers(answer)
        citations = build_citations(hits, markers)

        # 给每条引用附上「与回答重合的词」，前端在原文里高亮它们。
        # 用 answer 而不是 query：高亮的应该是「支撑这个回答的原文片段」，
        # 而不是用户问题里碰巧出现的词。
        for citation in citations:
            idx = citation["index"]
            if 1 <= idx <= len(hits):
                citation["keywords"] = extract_keywords(
                    answer[:1200], hits[idx - 1].content
                )

        # STEP 6 校验：没有引用说明模型没按规则回答，只提示不重生成（重生成太贵）
        grounded = is_grounded(answer, citations)
        if not grounded:
            log.warning("回答没有引用任何来源，可能是在自由发挥")
            yield RagEvent(
                "warning",
                {"text": "本次回答没有引用到具体资料，建议核对原文后再采用。"},
            )

        yield RagEvent("citations", {"items": citations})
        yield RagEvent(
            "done",
            {
                "grounded": grounded,
                "query": query,
                "hit_count": len(hits),
                "answer": answer,
            },
        )

    @staticmethod
    def _is_weak(hits: list[ChunkHit]) -> bool:
        """没召回，或最高分低于阈值，就认为检索不够好。

        阈值 0.15 是用 scripts/calibrate.py 实测出来的（同领域 0.30~0.61，无关 0.006）。
        换 rerank 模型后分布会变，要重新量。
        """
        if not hits:
            return True
        return hits[0].score < settings.score_threshold

    @staticmethod
    def _attach_doc_names(hits: list[ChunkHit]) -> None:
        """给检索结果补文件名（原地修改）。

        只读、按主键查、最多 5 个 id，所以直接查一次库，不绕 service 层。
        """
        doc_ids = {int(h.doc_id) for h in hits if h.doc_id.isdigit()}
        if not doc_ids:
            return

        db = SessionLocal()
        try:
            rows = db.query(Document).filter(Document.id.in_(doc_ids)).all()
            names = {str(d.id): d.filename for d in rows}
            for hit in hits:
                hit.doc_name = names.get(hit.doc_id, "")
        finally:
            db.close()
