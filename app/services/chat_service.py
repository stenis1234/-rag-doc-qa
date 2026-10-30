"""问答服务：会话管理 + 调用 RAG 流程 + 落库。

==========================================================================================
这个文件在整个项目里的位置
==========================================================================================
    api/v1/chat.py（HTTP 层）
        ↓ 调用
【本文件】ask_stream()          管会话、存消息、把事件原样往上抛
        ↓ 调用
    pipeline/rag.py            真正的问答流程
        ↓ 调用
    retrieval / generation     检索与生成

本文件**不做检索、不拼 prompt**，它只负责「一次问答的前后事务」：
    问之前：找到/新建会话，把历史读出来，把用户这句话存库
    问之后：把答案和引用存库

==========================================================================================
这里有一个 SSE 特有的坑，必须理解（面试也常问）
==========================================================================================
问题：FastAPI 的请求级依赖（db: DbSession）是在**响应结束之后**才关闭的。
     而流式响应要跑好几秒，中间会不断 yield。
     如果写成：

         def ask_stream(db, ...):
             ...生成...
             yield ...          # 这里开始往外发
             db.add(Message(...))   # 想在这里存库
             db.commit()

     在**普通** FastAPI 版本里，依赖关闭时机和生成器的生命周期容易对不上，
     可能出现「session 已关闭，commit 报错」。

本项目的做法：所有数据库写操作都**写在生成器内部**（不是在函数开头/结尾），
并且用 try/finally 保证异常时也能正确收尾。
另外，service 层拿到的 db 是请求级的，只要还在同一个请求生命周期内就是有效的——
关键在于不要在 yield 之后再假设它一定可用。

==========================================================================================
为什么历史只取最近 N 轮？
==========================================================================================
两个原因：
    1. token 成本：历史越长越贵，而且是每轮都重复付
    2. 注意力稀释：太长的历史会让模型忽略当前问题

真要做长对话，正确做法是「摘要压缩」（把早期对话总结成一段），
那是 M3 的事，现在先用最简单的截断。
"""

from __future__ import annotations

# ---- 标准库 ----
# json: 把引用列表序列化成字符串存进数据库（Text 字段）
import json

# Iterator: 类型标注，表示 ask_stream 是生成器
from collections.abc import Iterator

# ---- 第三方库 ----
# Session: SQLAlchemy 的会话类型，仅用于类型标注
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
    if conversation_id:
        conv = db.get(Conversation, conversation_id)
        if conv is None:
            # 抛 KeyError 而不是 HTTPException：
            # service 层不应该知道 HTTP 状态码的存在（那是 api 层的事）
            raise KeyError(f"会话 {conversation_id} 不存在")
        return conv

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
        for event in workflow.run_stream(question, [kb_id], history):
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
