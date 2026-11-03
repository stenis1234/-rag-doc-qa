"""模型客户端（LangChain 1.x）：连上模型、吐出文字。

被谁调用：app/pipeline/rag.py → stream_text()（生成答案）
          app/retrieval/rewrite.py → complete_text()（改写问题）

用 init_chat_model 而不是直接 new ChatDeepSeek：换模型只改 .env 的 LLM_MODEL，
接自建网关时填 LLM_BASE_URL 就走 OpenAI 兼容协议，代码不用动。

stream 边生成边返回（用于答案），invoke 等生成完一次性返回（用于改写，代码更简单）。
"""

from __future__ import annotations

# ---- 标准库 ----
# Iterator: 只用于类型标注，表示"这是个生成器"
from collections.abc import Iterator

# ---- 第三方库：LangChain ----
# init_chat_model 来自 langchain.chat_models（LangChain 1.x 的入口函数）。
# 它根据「模型名 + provider」自动构造对应的 ChatModel 实例，
# 比如 model="deepseek-chat", model_provider="deepseek" → ChatDeepSeek
from langchain.chat_models import init_chat_model

# BaseMessage 来自 langchain_core.messages，是所有消息类型的基类（只用于标注）
from langchain_core.messages import BaseMessage

# ---- 项目内部 ----
from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# 模块级缓存：模型对象内部持有 HTTP 连接池，反复创建会浪费连接、拖慢首次响应。
# 用全局变量 + 懒加载，而不是在 import 时就创建——否则没配 key 的话连 import 都会失败。
_model = None


def get_chat_model():
    """返回 ChatModel 实例（进程内复用同一个）。

    两条分支，对应两种部署方式：

    分支 A：配置了 LLM_BASE_URL → 走 OpenAI 兼容协议（ChatOpenAI）
        适用于：公司内网网关、one-api / new-api 这类中转服务、本地 vLLM。
        这类服务的共同点是"接口长得像 OpenAI"，所以用 ChatOpenAI 就能连。

    分支 B：没配 base_url → 用 LangChain 官方的 deepseek provider（ChatDeepSeek）
        它会自动读取环境变量 DEEPSEEK_API_KEY，并请求 api.deepseek.com。

    返回：
        ChatModel 实例，支持 .invoke() / .stream()
    """
    global _model
    if _model is not None:
        return _model

    if settings.llm_base_url:
        # 延迟导入：只有走中转时才需要 langchain_openai，
        # 放在模块顶部会让每次 import 都加载它（多几十毫秒启动时间）
        from langchain_openai import ChatOpenAI

        _model = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,  # 关键：指向你的网关，而不是 api.openai.com
            # api_key 是必填参数，但很多内网网关不校验，给个占位字符串即可
            api_key=settings.llm_api_key or "not-needed",
            # temperature 越低越"稳定保守"，RAG 场景建议 0~0.3：
            # 我们要的是"忠实复述资料"，不是"发挥创造力"
            temperature=settings.llm_temperature,
        )
    else:
        kwargs: dict = {"temperature": settings.llm_temperature}
        # 只有在 .env 里显式配了 key 才传，否则让 LangChain 自己去读环境变量
        # （两条路都支持，是为了兼容"用 .env"和"用系统环境变量"两种习惯）
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        _model = init_chat_model(
            settings.llm_model,
            model_provider="deepseek",
            **kwargs,
        )

    log.info("使用模型：%s（base_url=%s）", settings.llm_model, settings.llm_base_url or "默认")
    return _model


def _text_of(message) -> str:
    """从一条模型返回里取出纯文本。

    为什么需要这个函数？
        LangChain 1.x 里，模型的 content 可能是：
            ① 一个字符串                      → 最常见
            ② 一个 list，里面是内容块（dict）  → 推理模型/多模态模型会这样，
                                                 比如 [{"type":"text","text":"..."}]
        上层不想关心这些差异，所以在这里统一成字符串。

    参数：
        message  一个 ChatModel 的返回对象（AIMessage 或其中的 chunk）
    返回：
        纯文本（可能是空字符串，比如只返回了推理过程、没有正文）
    """
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                # 只取 type=text 的块，忽略 reasoning/thinking 等其他类型
                parts.append(block.get("text", ""))
        return "".join(parts)

    return ""


def stream_text(messages: list[BaseMessage]) -> Iterator[str]:
    """流式生成：模型吐一点，这里 yield 一点。

    谁调用它：pipeline/rag.py（第 4 步）
    它调用谁：ChatModel.stream()

    参数：
        messages  build_messages() 拼好的消息列表

    返回：
        生成器，逐个产出文本片段（可能为空字符串，所以下面要判断）
    """
    # .stream() 是 LangChain ChatModel 的标准方法，返回一个迭代器，
    # 每次产出一个 AIMessageChunk（可以理解为"答案的一个字/几个字"）
    for chunk in get_chat_model().stream(messages):
        text = _text_of(chunk)
        if text:  # 跳过空片段：有些模型会先吐一个空 chunk 表示"开始"
            yield text


def complete_text(messages: list[BaseMessage], max_tokens: int | None = None) -> str:
    """一次性生成（非流式），用于改写、扩写这类短输出。

    谁调用它：retrieval/rewrite.py

    参数：
        messages    消息列表
        max_tokens  输出上限。改写任务 80~120 足够；
                    给大了模型反而会开始"解释自己为什么这么改写"，
                    我们还得再写代码去清理这些废话。

    返回：
        去掉首尾空白的结果字符串
    """
    kwargs = {"max_tokens": max_tokens} if max_tokens else {}
    # .invoke() 会等模型生成完再返回，返回一个 AIMessage
    response = get_chat_model().invoke(messages, **kwargs)
    return _text_of(response).strip()
