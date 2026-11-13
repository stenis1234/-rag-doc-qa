"""多轮对话下的 query 改写。

解决两个具体问题：

1. **指代消解**
   用户第二句问「它的原理是什么」，直接拿去检索什么都搜不到，
   因为「它」没有语义。要先结合上一轮，改写成「Transformer 的原理是什么」。

2. **检索失败后的扩写**
   第一次检索分数太低，说明用词和文档对不上。
   让模型补几个同义词/相关术语再查一次，比直接放弃强。

这两个都用「让模型做一次短输出」实现，不复杂，但效果很明显。
注意 max_tokens 给得很小（100 左右）——这类任务不需要长篇大论，
给多了模型反而会开始解释自己为什么这么改写。
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.generation.llm import complete_text
from app.core.logging import get_logger

log = get_logger(__name__)

_CONDENSE_PROMPT = """把用户的最新问题改写成一个可以独立检索的问题。

要求：
1. 结合对话历史，把「它」「这个」「上面说的」这类指代替换成语义明确的名词
2. 保留原始问题的关键信息，不要改变意图
3. 只输出改写后的问题本身，不要解释，不要加引号
4. 如果最新问题本身已经足够独立，原样输出

对话历史：
{history}

最新问题：{question}

改写后的问题："""

_EXPAND_PROMPT = """用户问题在文档库里检索不到足够相关的内容，请生成一个更适合检索的查询。

要求：
1. 补充 2-3 个同义词、近义词或更通用的说法
2. 保留原问题的核心意图
3. 只输出一行查询语句，用空格分隔关键词，不要解释

原问题：{question}

检索用查询："""


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "（无）"
    lines = []
    for role, content in history:
        speaker = "用户" if role == "user" else "助手"
        # 历史太长会撑爆 prompt，截断即可：改写只需要语义线索，不需要全文
        lines.append(f"{speaker}：{content[:200]}")
    return "\n".join(lines)


# 出现这些词，才说明问题依赖上文（"它""这个""上面说的"）
_ANAPHORA = re.compile(
    r"它|他|她|这个|那个|这些|那些|上面|上述|刚才|前面|这两|这三|该(?:方案|功能|方法|产品)"
)


def needs_rewrite(question: str) -> bool:
    """问题是否依赖上文。不依赖就不必花一次模型调用去改写。"""
    return bool(_ANAPHORA.search(question or ""))


def condense_question(question: str, history: list[tuple[str, str]]) -> str:
    """把多轮问题改写成独立可检索的问题。

    多轮里常见「它有什么好处」这种单独检索没有意义的问题，需要补全主语。

    两个提前返回，都是为了省一次模型调用：
        - 没有历史：本来就是独立问题
        - 问题里没有指代词：已经能独立检索，改写大概率原样返回
    """
    if not history or not needs_rewrite(question):
        return question

    try:
        rewritten = complete_text(
            [
                SystemMessage(content="你是一个查询改写助手，只输出改写后的问题。"),
                HumanMessage(
                    content=_CONDENSE_PROMPT.format(
                        history=_format_history(history), question=question
                    )
                ),
            ],
            max_tokens=120,
        )
    except Exception as exc:  # noqa: BLE001 - 改写失败不该让问答失败
        log.warning("query 改写失败，使用原问题：%s", exc)
        return question

    rewritten = (rewritten or "").strip().strip('"').strip()
    # 防御：模型偶尔会返回空或者一大段解释，超过 200 字基本就是跑偏了
    if not rewritten or len(rewritten) > 200:
        return question
    log.info("query 改写：%s -> %s", question, rewritten)
    return rewritten


def expand_query(question: str) -> str:
    """检索失败后扩写查询。失败就返回原问题。"""
    try:
        expanded = complete_text(
            [
                SystemMessage(content="你是一个检索查询优化助手，只输出一行查询语句。"),
                HumanMessage(content=_EXPAND_PROMPT.format(question=question)),
            ],
            max_tokens=80,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("query 扩写失败：%s", exc)
        return question

    expanded = (expanded or "").strip()
    if not expanded or len(expanded) > 200:
        return question
    log.info("query 扩写：%s -> %s", question, expanded)
    return expanded
