"""问答接口（SSE 流式）—— HTTP 层，不写业务逻辑。

==========================================================================================
这个文件在整个项目里的位置
==========================================================================================
    POST /api/v1/chat  →  本文件 chat()  →  services/chat_service.py::ask_stream()
                                          →  pipeline/rag.py::RAGWorkflow.run_stream()
                                          →  逐个事件编码成 SSE 帧，推给浏览器

==========================================================================================
为什么用 SSE，而不是普通 JSON 返回？
==========================================================================================
普通 JSON 必须等答案全部生成完才能返回，用户要干等 5~10 秒，
期间页面毫无反应，体感非常差（还会以为程序卡死了）。
SSE（Server-Sent Events）能让服务器边生成边推，用户看到字一个个蹦出来。

为什么不用 WebSocket？
    WebSocket 是双向的，适合聊天室/协同编辑这种"双方都可能随时说话"的场景。
    问答是单向的（客户端说一句，服务端持续返回），SSE 正好，而且：
        - 就是普通 HTTP，不用协议升级，代理/防火墙友好
        - 断线自动重连（用浏览器原生 EventSource 时）

为什么前端不用 EventSource？
    因为 EventSource 只支持 GET，而我们要用 POST 传 JSON body（问题可能很长）。
    所以前端用 fetch + ReadableStream 自己解析 SSE 帧，见 web/app.js::streamChat()。

==========================================================================================
SSE 的报文格式（记不住也没关系，看下面的 _frame 函数）
==========================================================================================
    event: token
    data: {"text": "根据"}

    event: token
    data: {"text": "参考资料"}

    两个换行 = 一帧结束。少一个换行，浏览器就会一直等下一帧，表现为"卡住不输出"。
"""

from __future__ import annotations

# ---- 标准库 ----
# json: 把 dict 序列化成 JSON 字符串（SSE 的 data 必须是文本）
import json

# ---- 第三方库：FastAPI ----
# APIRouter        路由分组，最后在 main.py 里统一挂到 /api/v1 下
# HTTPException    抛 HTTP 错误（这里其实没用到，但保留以便后续加校验）
# StreamingResponse 流式响应：接受一个「生成器」，边产出边发给客户端
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# ---- 项目内部 ----
# DbSession: 类型别名，等价于 Annotated[Session, Depends(get_db)]
# 写起来比 `db: Session = Depends(get_db)` 短，而且改了实现只改一处
from app.api.deps import DbSession

# 接口的请求/响应模型（Pydantic），负责参数校验和文档生成
from app.models.schemas import ChatMessageOut, ChatRequest, Citation

# RagEvent: 流程里产出的事件类型（"token"/"citations"/...）
from app.pipeline.rag import RagEvent

# 业务编排：管会话、存消息、调用流程
from app.services import chat_service

router = APIRouter(prefix="/chat", tags=["问答"])

# ---------------------------------------------------------------------------
# SSE 响应头
# ---------------------------------------------------------------------------
# 这三个头少一个都可能出问题，都是实际踩过的坑：
#   Cache-Control: no-cache     别让浏览器/中间层缓存这个流（缓存了就只有第一帧）
#   Connection: keep-alive      保持长连接（默认就可能断）
#   X-Accel-Buffering: no       关掉 Nginx 的响应缓冲。
#       不加这个，Nginx 会攒够 4KB 才往下发，表现就是"等了 30 秒，答案突然一次性全出来"。
#       这个 bug 非常经典，排查时先看这个头。
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _frame(event: RagEvent) -> str:
    """把一个事件编码成一帧 SSE 文本。

    参数：
        event  RagEvent(type="token", data={"text": "..."})
    返回：
        'event: token\\ndata: {"text": "..."}\\n\\n'

    格式说明：
        event: 后面的名字会变成前端 addEventListener 的事件名
        data:  必须是单行（多行要用多个 data: 前缀）
        末尾必须有两个 \\n：一个是 data 行自己的，一个是帧分隔符
    """
    # ensure_ascii=False：让中文原样输出而不是 \u4f60\u597d。
    # 不影响正确性（两种都能被 JSON.parse 解析），但抓包调试时能直接看懂。
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.type}\ndata: {payload}\n\n"


@router.post("", summary="提问（SSE 流式返回）")
def chat(payload: ChatRequest, db: DbSession) -> StreamingResponse:
    """接收问题，返回一条 SSE 流。

    参数（由 FastAPI 从 JSON body 自动解析，模型见 models/schemas.py::ChatRequest）：
        kb_id            在哪个知识库里检索
        question         问题文本
        conversation_id  继续哪个会话；不传则新建

    返回：
        StreamingResponse，media_type="text/event-stream"

    注意这是个**同步 def**，不是 async def。
        原因：里面的业务（embedding、MSSQL 查询、Milvus 调用）全是同步阻塞的。
        写成 async def 反而会把事件循环卡死（因为阻塞调用不会 yield）。
        用同步 def，FastAPI 会自动把它丢到线程池里跑。
        这是一个很容易搞反的点：**async 里不能有阻塞调用，否则比同步还慢**。
    """

    def event_stream():
        """真正的生成器。StreamingResponse 会不断从它这里取字符串往外发。

        为什么把 try/except 放在这里而不是外面？
            因为流一旦开始（HTTP 200 已经发出去了），就没法再改状态码了。
            后面出的任何错，只能通过发一帧 "error" 事件告诉前端。
        """
        try:
            # 逐个事件转发。注意这里没有做任何转换——
            # API 层只负责"编码成 SSE"，业务逻辑全在 service 和 pipeline 里。
            for event in chat_service.ask_stream(
                db, payload.kb_id, payload.question, payload.conversation_id
            ):
                yield _frame(event)
        except KeyError as exc:
            # 典型场景：传了一个不存在的 conversation_id
            yield _frame(RagEvent("error", {"message": str(exc)}))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",  # 必须写，否则浏览器不会当成 SSE 处理
        headers=SSE_HEADERS,
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=list[ChatMessageOut],
    summary="取会话历史",
)
def list_messages(conversation_id: int, db: DbSession) -> list[ChatMessageOut]:
    """把历史消息读出来，刷新页面时用。

    返回结构里 citations 是**反序列化后的对象列表**，不是字符串。
        → 前端拿到就能直接渲染，不用再 JSON.parse 一次。
        → 代价是这里要做一次 json.loads，并处理脏数据。

    为什么 citations 在库里存成字符串？
        为了兼容 SQLite 和 PostgreSQL：用 Text 存 JSON 是最省事、最不会踩坑的做法
        （PG 的 jsonb 类型在 SQLite 里没有对应物，用 ORM 统一处理反而更麻烦）。
    """
    rows = chat_service.list_messages(db, conversation_id)

    result: list[ChatMessageOut] = []
    for row in rows:
        try:
            # 注意 json.loads 的结果是 dict 列表，要转成 Citation 对象才能进 Pydantic 模型
            citations = [Citation(**c) for c in json.loads(row.citations or "[]")]
        except (ValueError, TypeError):
            # 历史数据格式不对不应该让整个接口 500，退化成"没有引用"即可
            citations = []

        result.append(
            ChatMessageOut(
                id=row.id,
                role=row.role,
                content=row.content,
                citations=citations,
                created_at=row.created_at,
            )
        )
    return result
