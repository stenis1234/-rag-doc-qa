"""在线问答的完整流程 —— 建议从这个文件开始读。

==========================================================================================
这个文件在整个项目里的位置
==========================================================================================
它位于「HTTP 接口」和「检索/生成」之间，是问答链路的中枢：

    前端
     ↓  POST /api/v1/chat
    app/api/v1/chat.py                把事件编码成 SSE 帧
     ↓  调用
    app/services/chat_service.py      管会话、存消息
     ↓  调用
【本文件】RAGWorkflow.run_stream()     ★ 编排：改写 → 检索 → 评估 → 生成 → 引用 → 校验
     ↓  调用
    retrieval/rewrite.py              改写问题
    retrieval/hybrid.py               检索
    generation/prompt.py + llm.py     生成
    generation/citation.py            引用映射

==========================================================================================
对应你视频里那张图：本文件管的是「在线阶段」那半边
==========================================================================================
    图上的框                     本文件的 STEP    在干什么
    ---------------------------- ---------------- --------------------------------------
    用户提问                     （api 层）       收到问题
    改写（图上没有）             STEP 1           把「它」换成具体名词，否则检索必然失败
    问题向量化 + 向量检索        STEP 2           去 Milvus 里把可能相关的段落捞出来
    召回知识片段（图上这一格）   STEP 3           捞得不够好就换个说法再捞一次
    拼 prompt + 大模型           STEP 4           拿着这几段资料生成答案
    回答里的出处（图上这一格）   STEP 5           把 [1] 翻译成"第 7 页第 435 个字"
    （图上没有）                 STEP 6           检查它到底有没有用资料，还是在自己发挥

注意：**离线阶段（建库）不在这里**，在 ingestion/pipeline.py。
两段共用同一个向量模型（indexing/embedding.py），这是"能检索到东西"的前提。

==========================================================================================
用保险那个例子走一遍
==========================================================================================
    问：尊享e生2026版免赔额是多少？
    STEP 1  没有历史 → 原样返回（省一次模型调用）
    STEP 2  问题 → 1024 个数 → 去 Milvus 捞 8 条候选
    STEP 3  最高分 0.87 > 阈值 → 不用重查
    STEP 4  5 段资料拼进 prompt → 模型答"免赔额是 1 万元 [1]"
    STEP 5  [1] → chunk "3_00012" → 第 7 页
    STEP 6  有引用 → grounded=true

==========================================================================================
为什么这里没有用 LangGraph？
==========================================================================================
判断标准是：**流程里有没有「根据结果决定下一步」的复杂分支**。
目前只有「分数低就重查一次」这一个简单循环，用 while 写比画图更直观。
等 M3 加上「多轮工具调用 / 自纠错 / 人工审核」时，再整体搬进 LangGraph——
届时把下面每个 STEP 抽成一个节点函数即可，业务代码不用动。
"""

from __future__ import annotations

# ---- 标准库 ----
# collections.abc.Iterator：只用于类型标注，表示本函数是「生成器」
# dataclasses.dataclass / field：用来定义轻量的数据容器 RagEvent（自动生成 __init__）
from collections.abc import Iterator
from dataclasses import dataclass, field

# ---- 项目内部 ----
# settings：全局配置单例，见 app/core/config.py
from app.core.config import settings

# SessionLocal：SQLAlchemy 的 session 工厂（不是请求级的那个）。
# 这里为什么要自己开 session 而不是用 FastAPI 的依赖注入？
#   因为本文件属于 service 层，不应该知道 HTTP 的存在。
#   只在 _attach_doc_names() 里用一次，读取文件名，用完立刻关闭。
from app.core.db import SessionLocal

# get_logger：返回一个标准 logging.Logger，名字带模块路径
from app.core.logging import get_logger

# 引用相关的四个纯函数，见 generation/citation.py
#   build_citations   编号 → 结构化的引用对象（含 chunk_id/页码/偏移）
#   extract_markers   从答案文本里抠出 [1][2] 这类角标
#   is_grounded       判断「这次回答到底有没有依据」
from app.generation.citation import build_citations, extract_markers, is_grounded

# stream_text：流式生成，逐个 yield 文本片段，见 generation/llm.py
from app.generation.llm import stream_text

# REFUSAL_TEXT：资料不足时的标准拒答话术；build_messages：拼最终 prompt
from app.generation.prompt import REFUSAL_TEXT, build_messages

# ChunkHit：检索结果的统一数据结构（dataclass，见 indexing/milvus_store.py）
# 用它做类型标注，好处是编辑器能提示字段，也方便你 Ctrl+点击 跳过去看定义
from app.indexing.milvus_store import ChunkHit

# Document：ORM 模型，对应关系库里的 document 表
from app.models.orm import Document

# retrieve：检索总入口，见 retrieval/hybrid.py
from app.retrieval.hybrid import retrieve

# condense_question：指代消解；expand_query：检索失败后的扩写
from app.retrieval.rewrite import condense_question, expand_query

log = get_logger(__name__)


@dataclass(slots=True)
class RagEvent:
    """一个流式事件。API 层会把它原样塞进 SSE 的 data 字段。

    @dataclass 是标准库提供的装饰器：自动生成 __init__ / __repr__ / __eq__。
    slots=True 表示「只允许这两个属性」，能省一点内存，也能防止手滑写错属性名
    （写错会直接报 AttributeError，而不是悄悄多出一个字段）。

    字段说明：
        type   事件类型，前端按它决定怎么渲染。取值：
                 "meta"       流程元信息（改写后的问题、命中条数、重查次数）
                 "token"      一小段答案文本
                 "citations"  引用列表
                 "warning"    警告（比如回答没有引用）
                 "done"       结束，带 grounded 标志
                 "error"      出错
        data   事件携带的数据，结构随 type 变化（用 dict 是为了让 SSE 能直接 json.dumps）
    """

    type: str
    data: dict = field(default_factory=dict)


class RAGWorkflow:
    """问答流程。

    设计成类而不是一堆函数，是为了将来能注入依赖（比如换一个检索器做 A/B）。
    但它是**无状态**的：不要在实例上存会话数据，每次问答都 new 一个。
    原因：FastAPI 里同一个实例可能被多个请求并发使用，存状态必然串数据。
    """

    def run_stream(
        self,
        question: str,
        kb_ids: list[int],
        history: list[tuple[str, str]] | None = None,
    ) -> Iterator[RagEvent]:
        """执行一次问答，边算边 yield 事件。

        谁调用它：app/services/chat_service.py::ask_stream()
        它调用谁：condense_question / retrieve / stream_text / build_citations

        参数：
            question  用户原始输入，可能是「它的原理是什么」这种指代不明的句子
            kb_ids    允许检索的知识库 id 列表（空列表 → 直接拒答，防止跨库泄漏）
            history   最近几轮对话 [(role, content)]，role 是 "user" / "assistant"

        返回：
            生成器，逐个产出 RagEvent。注意这是「惰性」的：
            调用本函数不会立刻执行，只有 API 层开始迭代它时才真正跑起来。

        为什么用生成器而不是「算完一次性返回」？
            因为要给用户看到字一个个蹦出来。如果用普通函数，用户得等 10 秒才看到全部答案，
            体感差异极大。
        """
        history = history or []
        attempts = 0  # 已经重查了几次

        # ======================================================================
        # STEP 1  改写：把「它」换成具体名词
        # ======================================================================
        # 为什么不直接用 question 去检索？
        #   多轮对话里第二句常常是「它的原理是什么」，这句话没有语义，
        #   拿去做 embedding 只会得到一堆随机结果。
        # condense_question 内部会判断 history 是否为空，为空直接返回原问题（省一次模型调用）
        query = condense_question(question, history)

        # ======================================================================
        # STEP 2  检索
        # ======================================================================
        # retrieve() 内部做了三件事：问题转向量 → Milvus 双路召回 → 精排
        # 返回值已按相关性从高到低排好，score 是 0~1 的分数
        hits = retrieve(query, kb_ids)

        # ======================================================================
        # STEP 3  评估 + 重查
        # ======================================================================
        # 「弱」的定义在文件末尾的 _is_weak()：一条没召回，或最高分低于阈值。
        # 为什么要重查？因为用户的用词可能和文档差很远（比如问「大模型」而文档写「LLM」），
        # 让模型补几个同义词再查一次，经常能救回来。
        #
        # 为什么要有 max_retry 上限？每次重查 = 一次 LLM 调用 + 一次检索，
        # 不设上限遇到「文档里根本没有」的问题会无限重试，又慢又烧钱。
        while self._is_weak(hits) and attempts < settings.max_retry:
            attempts += 1
            log.info("检索结果偏弱（%d 条），第 %d 次扩写重查", len(hits), attempts)
            query = expand_query(query)
            hits = retrieve(query, kb_ids)

        # 先把流程信息推给前端（调试时很有用：能看到改写后的问题和重查次数）
        yield RagEvent(
            "meta",
            {
                "query": query,          # 改写后实际用于检索的问题
                "hit_count": len(hits),  # 最终命中几条
                "retries": attempts,     # 重查了几次（正常应该是 0）
            },
        )

        # 一条都没召回：直接拒答。
        # 这一步很关键——如果硬把空结果塞给模型，它一定会用自身知识编一个答案出来，
        # 而且看起来还很像真的。RAG 的价值就在于这种时候敢说「不知道」。
        if not hits:
            log.info("没有召回任何内容，返回拒答")
            yield RagEvent("token", {"text": REFUSAL_TEXT})
            yield RagEvent("citations", {"items": []})
            yield RagEvent("done", {"grounded": False, "query": query})
            return

        # ======================================================================
        # STEP 4  生成
        # ======================================================================
        # 补文件名：Milvus 里只存了 doc_id，而 prompt 和前端都需要显示「来自哪份文档」
        self._attach_doc_names(hits)

        # 拼消息：system（规则）+ 历史 + human（参考资料 + 问题）
        messages = build_messages(question, hits, history)

        # 流式生成：模型每吐一小段，就立刻 yield 给前端。
        # 注意这里要用**原始 question** 而不是改写后的 query：
        # 改写是为了检索服务的，用户看到的回答必须针对他自己问的那句话。
        parts: list[str] = []
        for piece in stream_text(messages):
            parts.append(piece)
            yield RagEvent("token", {"text": piece})

        # 拼回完整答案，后面解析引用要用
        answer = "".join(parts).strip()

        # ======================================================================
        # STEP 5  引用
        # ======================================================================
        # 模型输出里带 [1][2] 角标，这里把它们解析成结构化的引用对象。
        # 为什么要这一步？因为前端要能「点角标 → 跳原文」，
        # 就必须知道第 1 条引用对应哪个 chunk、哪一页、原文的第几个字。
        markers = extract_markers(answer)
        citations = build_citations(hits, markers)

        # ======================================================================
        # STEP 6  校验
        # ======================================================================
        # 一条引用都没有 = 模型没按规则回答，很可能是它在自由发挥。
        # 这里只打标记 + 给前端一个黄条提示，不重新生成。
        # 为什么先不重新生成？因为「重新生成」要再花一次完整的生成成本，
        # 而且大概率还是同样的结果。先在数据上把这种情况统计出来更重要。
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
                "answer": answer,  # 完整答案，方便调用方一次性拿到（比如存库）
            },
        )

    # ======================================================================
    # 内部辅助方法
    # ======================================================================

    @staticmethod
    def _is_weak(hits: list[ChunkHit]) -> bool:
        """判断「这次检索是不是不够好」，不够好就触发重查。

        规则：
            一条都没召回                                  → 弱
            最高分 < settings.score_threshold（默认 0.15） → 弱

        阈值为什么是 0.15 而不是 0.35？
            是用 scripts/calibrate.py 在真实文档上量出来的：
            同领域问题的最高分在 0.30~0.61，无关问题在 0.006 左右。
            0.35 会把 0.30 这种「其实很相关」的结果误判成弱。
            换模型后这个分布会变，要重新量。

        @staticmethod 表示它不需要访问 self，纯粹是「挂在类下面方便归类」。
        """
        if not hits:
            return True
        return hits[0].score < settings.score_threshold

    @staticmethod
    def _attach_doc_names(hits: list[ChunkHit]) -> None:
        """给每条命中补上文件名（原地修改 hits）。

        为什么要补：Milvus 里只存了 doc_id，文件名存在关系库里。
        prompt 要显示来源，前端引用卡片也要显示文件名。

        为什么不走 service 层：
            只读、只按主键查、量很小（最多 5 个 id），为一个展示字段绕一层反而更难读。
            这是有意的取舍，不是在偷懒。

        SQLAlchemy 用法说明：
            db.query(Document).filter(Document.id.in_(doc_ids)).all()
                → 生成 SELECT * FROM document WHERE id IN (...)
            db.get(Document, pk) 是「按主键查」，比自己拼 filter 更快也更清晰。
        """
        # 只保留能转成整数的 doc_id，防止脏数据把整个问答搞崩
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
            # 必须关闭：这是自己开的 session，不关会占着连接池
            db.close()
