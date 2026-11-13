"""问答服务：会话管理 + 调用 RAG 流程 + 落库。

POST /api/v1/chat → 本文件 ask_stream() → app/pipeline/rag.py::RAGWorkflow
本文件不做检索、不拼 prompt，只管一次问答前后的事务：找会话、读历史、存消息。

SSE 的坑：流式响应要跑好几秒，所有数据库写操作都必须写在生成器内部，
不能写成「先 yield 完、再回头保存」——那时请求级 session 可能已经关闭。

历史只取最近 N 轮：历史越长越贵，而且会让模型忽略当前问题。
"""

from __future__ import annotations

# ---- 标准库 ----
# json: 把引用列表序列化成字符串存进数据库（Text 字段）
import json

# Iterator: 类型标注，表示 ask_stream 是生成器
from collections.abc import Iterator

# ---- 第三方库 ----
# Session: SQLAlchemy 的会话类型，仅用于类型标注
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session

# ---- 项目内部 ----
from app.core.config import settings
from app.core.logging import get_logger

# 三个 ORM 模型：会话、消息
from app.models.orm import Conversation, Message

# RAGWorkflow: 问答流程；RagEvent: 事件类型
from app.pipeline.rag import RAGWorkflow, RagEvent

log = get_logger(__name__)


def _get_or_create_conversation(
    db: Session, kb_id: int, conversation_id: int | None, question: str
) -> Conversation:
    """拿到会话，没有就新建。

    参数：
        conversation_id  前端传的会话 id；None 表示"这是新对话"
        question         用来当会话标题（截前 30 个字）

    返回：
        Conversation 实例（已 commit，id 可用）

    为什么要 commit 而不是 flush？
        因为后面要立刻用 conv.id 往前端推 meta 事件，
        而且浏览器刷新后要能通过这个 id 找回历史——必须真正落库。
    """
    # 如果有会话 id 就用它
    if conversation_id:
        conv = db.get(Conversation, conversation_id)
        if conv is None:
            # 抛 KeyError 而不是 HTTPException：
            # service 层不应该知道 HTTP 状态码的存在（那是 api 层的事）
            raise KeyError(f"会话 {conversation_id} 不存在")
        return conv

    # 如果没有会话 id 就新建一个
    # title 截断：整句话当标题会让左侧列表撑爆
    conv = Conversation(kb_id=kb_id, title=question[:30] or "新会话")
    db.add(conv)
    db.commit()
    # refresh 的作用：把数据库生成的字段（自增 id、created_at）读回对象。
    # 不 refresh 的话 conv.id 可能还是 None。
    db.refresh(conv)
    return conv


def _history(db: Session, conversation_id: int) -> list[tuple[str, str]]:
    """取最近几轮对话，按时间**正序**返回。

    参数：
        conversation_id  会话 id
    返回：
        [(role, content), ...]，role 是 "user" / "assistant"

    实现细节：
        order_by(Message.id.desc()) + limit(N)  → 先取最新的 N 条（倒序）
        reversed(...)                            → 再翻回正序
        为什么不直接正序取？
            正序 + limit 取到的是**最早**的 N 条，那是最没用的（通常是寒暄）。
            对话历史要的是"最近说了什么"。
    """
    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
        .limit(settings.history_turns * 2)  # 一轮 = 用户 + 助手，所以乘 2
        .all()
    )
    return [(m.role, m.content) for m in reversed(rows)]


def _maybe_summarize(db: Session, conv: Conversation) -> str:
    """消息变多时，把早期对话压成摘要存进 conversation.summary。

    触发条件：尚未被摘要的消息条数 > history_turns*2 + 2。
    做法：除最近 history_turns*2 条以外，全部交给模型压成一段摘要，
         覆盖旧摘要并存回数据库（summary_until_id 记录覆盖到哪条）。

    返回：最新的摘要文本（没触发就返回旧摘要）。
    失败不抛异常：摘要只是锦上添花，不能因为它让问答整个挂掉。
    """
    keep = settings.history_turns * 2
    rows = (
        db.query(Message)
        .filter(
            Message.conversation_id == conv.id,
            Message.id > (conv.summary_until_id or 0),
        )
        .order_by(Message.id)
        .all()
    )
    if len(rows) <= keep + 2:
        return conv.summary or ""

    older = rows[:-keep]  # 除最近几轮外，其余压进摘要
    transcript = "\n".join(
        f"{'用户' if m.role == 'user' else '助手'}：{m.content[:300]}" for m in older
    )

    prompt = (
        "把下面的对话压缩成一段中文摘要，供后续对话参考。\n"
        f"要求：不超过 {settings.summary_max_chars} 字；保留用户的目标、已确认的事实或结论、"
        "未完成的事项；不要编造；只输出摘要正文。\n\n"
        f"已有摘要：\n{conv.summary or '（无）'}\n\n"
        f"新增对话：\n{transcript}"
    )

    try:
        # 延迟导入：只有真需要摘要时才加载模型客户端
        from app.generation.llm import complete_text

        new_summary = complete_text(
            [SystemMessage(content="你是一个对话摘要助手，只输出摘要正文。"),
             HumanMessage(content=prompt)],
            max_tokens=600,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("生成对话摘要失败，继续用旧摘要：%s", exc)
        return conv.summary or ""

    if not new_summary:
        return conv.summary or ""

    conv.summary = new_summary[: settings.summary_max_chars * 2]
    conv.summary_until_id = older[-1].id
    db.commit()
    log.info(
        "会话 %s 摘要更新：覆盖到消息 #%s，摘要 %d 字",
        conv.id,
        conv.summary_until_id,
        len(conv.summary),
    )
    return conv.summary


def ask_stream(
    db: Session,
    kb_id: int,
    question: str,
    conversation_id: int | None = None,
) -> Iterator[RagEvent]:
    """流式问答的入口。

    谁调用它：api/v1/chat.py::chat() 里的 event_stream()
    它调用谁：pipeline/rag.py::RAGWorkflow.run_stream()

    参数：
        kb_id            在哪个知识库检索
        question         用户问题
        conversation_id  继续哪个会话

    返回：
        生成器，产出 RagEvent。事件顺序：
            meta(conversation_id) → 流程内部的各种事件 → done
    """
    # ---------- 1. 会话与历史 ----------
    conv = _get_or_create_conversation(db, kb_id, conversation_id, question)
    history = _history(db, conv.id)
    # 早期对话压成摘要（没到阈值就返回已有摘要，几乎没有开销）
    summary = _maybe_summarize(db, conv)

    # ---------- 2. 先把用户这句话存库 ----------
    # 为什么要先存？因为如果生成过程中用户关了页面，
    # 至少还能在历史里看到自己问过什么（否则这句话就"消失"了）。
    # citations="[]" 是因为用户消息没有引用，但字段是 NOT NULL。
    db.add(Message(conversation_id=conv.id, role="user", content=question, citations="[]"))
    db.commit()

    # ---------- 3. 告诉前端会话 id ----------
    # 前端拿到后会保存在内存里，下一轮追问带上它，才能串成多轮对话。
    # 这是 ask_stream 自己加的事件（pipeline 不知道"会话"这个概念）。
    yield RagEvent("meta", {"conversation_id": conv.id})

    # ---------- 4. 跑流程，边跑边转发 ----------
    answer_parts: list[str] = []
    citations: list[dict] = []
    grounded = False

    try:
        workflow = RAGWorkflow()
        # 注意：history 要在"存入本轮问题之前"取，否则会把当前问题也当成历史
        for event in workflow.run_stream(question, [kb_id], history, summary):
            # 顺手收集最终要落库的内容（不能指望后面还能从事件里再拿一遍）
            if event.type == "token":
                answer_parts.append(event.data.get("text", ""))
            elif event.type == "citations":
                citations = event.data.get("items", [])
            elif event.type == "done":
                grounded = bool(event.data.get("grounded"))

            # 原样往上抛，让 API 层编码成 SSE
            yield event

    except Exception as exc:  # noqa: BLE001
        # 流已经开始，HTTP 状态码改不了了，只能发一个 error 事件
        log.exception("问答失败")
        yield RagEvent("error", {"message": str(exc)[:500]})
        return

    # ---------- 5. 存助手回答 ----------
    # 只在有内容时才存：如果生成失败、answer 为空，存一条空消息会污染历史
    answer = "".join(answer_parts).strip()
    if answer:
        db.add(
            Message(
                conversation_id=conv.id,
                role="assistant",
                content=answer,
                # ensure_ascii=False：中文直接存，方便你以后用 sqlite3 命令行看数据
                citations=json.dumps(citations, ensure_ascii=False),
            )
        )
        db.commit()
        log.info("会话 %s 保存回答（引用 %d 条，grounded=%s）", conv.id, len(citations), grounded)


def list_messages(db: Session, conversation_id: int) -> list[Message]:
    """按正序返回一个会话的所有消息。供 api/v1/chat.py 的 GET 接口使用。"""
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id)
        .all()
    )


def list_conversations(db: Session, kb_id: int, limit: int = 50) -> list[dict]:
    """某个知识库下的会话列表，最新的在前。

    每条带消息数和最后一条消息时间，前端用它们显示“几条消息 / 最近什么时候”。
    会话不多时逐条统计就够了；将来量大可以改成一次 group by。
    """
    from sqlalchemy import func

    convs = (
        db.query(Conversation)
        .filter(Conversation.kb_id == kb_id)
        .order_by(Conversation.id.desc())
        .limit(limit)
        .all()
    )

    out: list[dict] = []
    for conv in convs:
        count, last_at = (
            db.query(func.count(Message.id), func.max(Message.created_at))
            .filter(Message.conversation_id == conv.id)
            .one()
        )
        out.append(
            {
                "id": conv.id,
                "kb_id": conv.kb_id,
                "title": conv.title,
                "message_count": count or 0,
                "last_at": last_at,
                "created_at": conv.created_at,
            }
        )
    return out


def delete_conversation(db: Session, conversation_id: int) -> None:
    """删会话（连带消息）。摘要跟着会话一起没。"""
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise KeyError(f"会话 {conversation_id} 不存在")
    db.query(Message).filter(Message.conversation_id == conv.id).delete()
    db.delete(conv)
    db.commit()
    log.info("会话 %s 已删除", conversation_id)
