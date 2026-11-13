"""所有 prompt 集中在这里 —— 全项目唯一的 prompt 来源。

被谁用：app/pipeline/rag.py 的第 4 步「生成」，产出的消息列表直接喂给 generation/llm.py。

RAG prompt 的三个要点：
    1. 明确「只依据资料」，并给模型一个「资料不足」的出口，否则它一定编
    2. 明确用 [1][2] 标来源，后端才能解析；让模型输出 JSON 更容易编字段名
    3. 明确不要照抄原文，否则答案是复制粘贴
"""

from __future__ import annotations

# ---- 第三方库：langchain_core ----
# 这三个消息类来自 langchain_core.messages（LangChain 的核心消息抽象）：
#   SystemMessage  系统提示，相当于"角色设定/规则"，模型会当作最高优先级指令
#   HumanMessage   用户这一轮说的话
#   BaseMessage    上面两者的共同基类，只用于类型标注
# 它们最终会被模型客户端翻译成 OpenAI 协议的 role=system / role=user 消息。
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# ---- 项目内部 ----
# settings: 取 history_turns（带几轮历史）
from app.core.config import settings

# ChunkHit: 检索结果的类型，仅用于类型标注
from app.indexing.milvus_store import ChunkHit

# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------
# 每一条规则都对应一个真实踩过的坑：
#   第 1 条（只依据资料）  → 不加这句，模型会用训练数据里的知识补充，答案看似正确但无法溯源
#   第 2 条（标注来源）    → 不加这句，答案里一个角标都没有，前端引用卡片全空
#   第 3 条（资料不足时）  → 只写"不许编"不够，要给模型一个"体面退出"的方式
#   第 4 条（中文+不照抄）→ 不加这句，中文文档会被用英文总结，或者整段复制
#   第 5 条（矛盾要指出）  → 多文档场景必然出现互相矛盾的说法，让模型挑一个会误导用户
# 单条资料进 prompt 的最大字符数：块本身约 900 字，全塞进去 5 条就 4500 字，
# 而回答通常只用得上开头和关键句。
CONTEXT_CHARS_PER_HIT = 400
# 历史消息单条最大字符数：助手回答常有七八百字，全量带进 prompt 非常贵。
HISTORY_CHARS_PER_MSG = 200


SYSTEM_PROMPT = """你是一个严谨的文档问答助手。请严格遵守以下规则：
1. 只能依据「参考资料」回答问题，不要使用你自己的知识补充事实。
2. 每个结论后面必须标注来源编号，格式为 [1]、[2]，编号对应参考资料。
3. 如果参考资料中没有足够信息，直接回答「根据现有资料，我无法回答这个问题」，
   并说明缺少什么信息。不要编造，不要猜测。
4. 用中文回答，条理清晰，必要时用分点陈述；不要整段照抄原文。
5. 如果不同资料之间存在矛盾，请指出矛盾之处，并分别标注来源。
6. 只回答「问题」这一条；历史对话仅供参考，不要复述、追加或纠正你之前的回答。"""


def format_context(hits: list[ChunkHit]) -> str:
    """把检索结果拼成 prompt 里的「参考资料」段落。

    谁调用它：build_messages()
    输出长这样：

        [1] 来源：ai.md 第 3 页（四、深度学习 > 4.2 CNN）
        2012 年，AlexNet 在 ImageNet 竞赛中夺冠……

        [2] 来源：其他.pdf 第 7 页
        ……

    为什么要带「来源 / 页码 / 标题」？
        ① 模型能写出「根据《xxx》第 3 页」这种话，答案更可信
        ② 这些元信息本身也会被前端用来做引用卡片，格式统一省得二次加工
        ③ 标题路径给了模型上下文：同样一段话，在"实验设置"和"结论"下含义完全不同

    参数：
        hits  已排好序的检索结果（第 1 条就是最相关的）
    返回：
        拼好的字符串；没有命中时返回 "（无）"
    """
    if not hits:
        return "（无）"

    blocks: list[str] = []
    # enumerate(..., start=1) 让编号从 1 开始，和答案里的 [1] 对应。
    # 这个对应关系是整个引用机制的基础：prompt 里第几条 = 答案里的 [几]。
    for i, hit in enumerate(hits, start=1):
        # 文件名可能为空（数据脏了），给个兜底，避免 prompt 里出现 "来源： 第 页" 这种残句
        source = hit.doc_name or f"文档{hit.doc_id}"
        # page_no=0 表示这份文档没有页码概念（Markdown / docx），此时不显示页码
        page = f" 第 {hit.page_no} 页" if hit.page_no else ""
        title = f"（{hit.title}）" if hit.title else ""

        # .strip() 去掉正文首尾空白：prompt 里的多余空行会浪费 token
        body = hit.content.strip()
        # 单段资料截断：900 字的块全塞进去，5 条就 4500 字，
        # 而回答通常只用得上开头和关键句，截到 400 字能省一半以上
        if len(body) > CONTEXT_CHARS_PER_HIT:
            body = body[:CONTEXT_CHARS_PER_HIT] + "…"
        blocks.append(f"[{i}] 来源：{source}{page}{title}\n{body}")

    # 用两个换行分隔，比单个换行更容易让模型分清「这是一条新资料」
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    hits: list[ChunkHit],
    history: list[tuple[str, str]] | None = None,
    summary: str = "",
) -> list[BaseMessage]:
    """组装最终送给模型的消息列表。

    谁调用它：pipeline/rag.py（第 4 步）
    它调用谁：format_context()

    参数：
        question  用户**原始**问题（不是改写后的）
                  为什么？改写是为了检索服务的，回答必须针对用户真正问的那句话。
        hits      检索结果
        history   最近几轮对话 [(role, content)]，role 是 "user"/"assistant"

    返回：
        [SystemMessage, ...历史消息..., HumanMessage(参考资料 + 问题)]
    """
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    # 早期对话的摘要（超过窗口的部分被压成这一段），放在历史之前
    if summary:
        messages.append(SystemMessage(content=f"（早前对话的摘要）{summary}"))

    # ---- 历史对话 ----
    # 只取最近 history_turns*2 条（一轮 = 用户一条 + 助手一条）。
    # 为什么要截断？历史越长，token 越贵，而且模型注意力会被早期内容分散。
    for role, content in (history or [])[-settings.history_turns * 2 :]:
        # 历史只保留语义线索，不需要全文——单条截到 200 字
        clipped = content if len(content) <= HISTORY_CHARS_PER_MSG else content[:HISTORY_CHARS_PER_MSG] + "…"
        if role == "user":
            messages.append(HumanMessage(content=clipped))
        else:
            # 助手的历史回答必须用 AIMessage：
            # 用 SystemMessage 会让模型把旧答案当成"系统指令/既定事实"，
            # 结果就是新问题它不答，转而去复述或纠正上一轮的答案。
            messages.append(AIMessage(content=clipped))

    # ---- 本轮 ----
    # 把参考资料放在问题**前面**，是经过考虑的：
    # 模型是按顺序读的，先看到资料再看问题，回答时会自然地"引用上面的内容"；
    # 反过来先问问题再看资料，模型容易先形成一个答案，然后拿资料去凑。
    user_prompt = (
        f"参考资料：\n{format_context(hits)}\n\n"
        f"问题：{question}\n\n"
        "请依据上述资料回答，并在每个结论后标注来源编号。"
    )
    messages.append(HumanMessage(content=user_prompt))
    return messages


# ---------------------------------------------------------------------------
# 拒答话术
# ---------------------------------------------------------------------------
# 为什么不在 prompt 里让模型生成，而是写死一段话？
#   因为「检索到 0 条」这种情况根本不需要调模型——
#   调了它只会用自身知识编一个答案出来，白花钱还误导用户。
#   直接返回这段固定文本，既快又诚实。
#
# 注意它还要"有用"：不只说"我不知道"，还告诉用户下一步能做什么
# （换说法 / 传文档），这比一句冷冰冰的"无法回答"体验好很多。
REFUSAL_TEXT = (
    "根据现有资料，我无法回答这个问题。\n\n"
    "可能的原因：知识库里没有相关内容，或者问题里的关键词与文档用词差异太大。"
    "你可以换个说法，或者先上传相关文档。"
)
