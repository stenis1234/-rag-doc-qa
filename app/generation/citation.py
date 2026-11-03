"""引用解析：把答案里的 [1][2] 映射回 chunk。

被谁调用：app/pipeline/rag.py 的第 5、6 步
输入：答案文本 + 检索结果；输出：结构化引用（chunk_id / 页码 / 字符偏移）

为什么不让模型直接输出 JSON：它会编字段名和 id，而 [n] 是最不容易写错的格式。
本文件是纯函数、不依赖第三方库，可以直接单测。
"""

from __future__ import annotations

# ---- 标准库 ----
# re: 正则表达式，用来从答案文本里找 [1] [12] 这类角标
import re

# ChunkHit: 只用于类型标注，运行时不需要它也能跑
# （但为了让编辑器能跳转、能提示字段，还是导入它）
from app.indexing.milvus_store import ChunkHit

# ---------------------------------------------------------------------------
# 角标正则
# ---------------------------------------------------------------------------
# \[  匹配左方括号。方括号在正则是特殊字符（表示字符组），所以要转义
# (\d{1,2})  捕获组，1~2 位数字。为什么限制两位？
#            引用条数最多 5 条（TOP_K_FINAL），两位足够了；
#            不限长度的话，模型偶尔写个 [2024] 会被误认成引用编号。
# \]  匹配右方括号
#
# 注意：这个正则**不匹配** [1,2] 这种写法。模型偶尔会这么写，
# 我们宁可漏掉（那条引用不展示），也不要错认（把 12 当成编号 1 和 2）。
_MARKER = re.compile(r"\[(\d{1,2})\]")


def extract_markers(answer: str) -> list[int]:
    """按出现顺序取出引用编号，并去重。

    谁调用它：pipeline/rag.py（第 5 步）

    参数：
        answer  模型输出的完整答案文本

    返回：
        编号列表，例如 [1, 3]。顺序 = 在答案里第一次出现的顺序，
        这样前端展示的引用卡片顺序和正文角标顺序一致，用户不会觉得乱。

    示例：
        >>> extract_markers("结论 A [2]，结论 B [1]，又提到 [2]。")
        [2, 1]
    """
    seen: list[int] = []
    # finditer 返回所有匹配（而不是 findall 只返回字符串），
    # 因为我们需要 group(1) 拿到括号里的数字。
    for matched in _MARKER.finditer(answer or ""):
        idx = int(matched.group(1))
        # 去重但保留首次出现的顺序。
        # 用 list 而不是 set，就是因为 set 不保证顺序——
        # 而"引用卡片的顺序"对用户是有意义的。
        if idx not in seen:
            seen.append(idx)
    return seen


def build_citations(
    hits: list[ChunkHit],
    markers: list[int] | None = None,
) -> list[dict]:
    """把编号映射回 chunk 元信息。

    谁调用它：pipeline/rag.py（第 5 步）

    参数：
        hits     检索结果（顺序必须和 prompt 里给模型的编号一致！）
        markers  extract_markers() 的结果；None 或空列表表示"模型没引用任何来源"

    返回：
        dict 列表，每个 dict 就是接口里的 Citation 结构

    ⚠️ 关键设计：markers 为空时返回**空列表**，不要退化成"返回全部命中"。
        模型没引用任何来源，说明它这次不是基于资料回答的。
        这时候给用户展示 5 张"引用卡片"反而是误导——
        用户会以为这些话有出处，实际上没有。
        宁可让前端显示"无引用"的黄条警告。
    """
    if not hits or not markers:
        return []

    citations: list[dict] = []
    for idx in markers:
        # 编号从 1 开始，所以要减 1 才能当列表下标。
        # 越界说明模型在瞎编编号（比如只有 3 条资料却写了 [7]），直接丢弃。
        if 1 <= idx <= len(hits):
            # to_citation() 定义在 indexing/milvus_store.py 的 ChunkHit 里，
            # 负责把检索结果转成接口用的结构（含截断后的 snippet）
            citations.append(hits[idx - 1].to_citation(index=idx))
    return citations


def is_grounded(answer: str, citations: list[dict]) -> bool:
    """判断这次回答是否「有据可依」。

    谁调用它：pipeline/rag.py（第 6 步）

    规则（简单到有点粗暴）：
        有至少一条合法引用，且答案非空 → 有据可依

    为什么先用这么简单的规则？
        便宜、快、可解释。真正严格的做法是让另一个模型判断
        「answer 的每个论断是否真的被 citations 支撑」，但那要多花一次调用，
        而且会引入新的不确定性。先把简单规则跑起来，
        等有了评测集，再决定这额外的成本值不值（M3 的事）。
    """
    return bool(citations) and len(answer.strip()) > 0
