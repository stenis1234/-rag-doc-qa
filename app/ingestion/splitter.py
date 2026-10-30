"""把解析出来的页面文本切成 chunk（切片）。

==========================================================================================
这个文件在整个项目里的位置
==========================================================================================
    被谁调用：ingestion/pipeline.py::ingest_document()（第 2 步「清洗与切分」）
    它调用谁：（无）——只依赖 Python 标准库
    输入：一页清洗过的纯文本
    输出：TextChunk 列表，每个都带「在原文里的起止位置」

==========================================================================================
为什么不用 langchain_text_splitters 的 RecursiveCharacterTextSplitter？
==========================================================================================
因为**它不告诉你每个 chunk 在原文里的位置**（char_start / char_end）。
而「点引用角标 → 跳到原文第几个字」必须要这个偏移量。

LangChain 的切分器里有个 `add_start_index=True` 能给出起始位置，
但：① 它在结构感知（按标题分节）时行为和你想的不一样；
    ② 它给的是相对整篇文档的偏移，而我们按「页」切分，需要页内偏移；
    ③ 自己实现这 100 行，你能完全看懂每一处边界条件。

所以这里手写了一个，顺便把「标题路径」和「偏移计算」放在同一遍扫描里完成。

==========================================================================================
算法思路（一句话版）
==========================================================================================
按行扫描，维护一个「标题栈」得到标题路径；遇到新标题就结束当前块；
块超过 max_chars 就在句子边界切开；最后把过短的块合并进前一块。

三个必须守住的不变量：
    ★ 对每个 chunk：text[char_start:char_end] == content
      （tests/test_splitter.py 里有专门的测试守着这条）
    ★ 每个 chunk 的长度 <= max_chars
    ★ 一个 chunk 不跨越标题边界（保证标题路径是准确的）

==========================================================================================
先认识三个概念
==========================================================================================
    偏移量（offset）：字符在字符串里的下标，从 0 开始。中文一个字算 1 个字符。
        text = "你好世界"，"世界" 的 char_start=2, char_end=4（左闭右开）
    左闭右开：包含起点、不包含终点。Python 切片 s[a:b] 就是这个语义，
        所以 content == text[char_start:char_end] 能严格成立。
    标题栈：像 ["一、总览", "1.1 细节"] 这样的列表，
        表示「当前这段文字属于哪一级哪一节的标题之下」。
"""

from __future__ import annotations

# ---- 标准库 ----
# re：正则表达式。用来识别 Markdown 标题、找句子边界。
import re

# dataclass：自动生成 __init__ 等，用来定义 TextChunk
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 两个正则
# ---------------------------------------------------------------------------
# 句子结束符：中文的 。！？； 英文的 !?; 以及换行。
# 用「字符组」[...] 表示其中任意一个；这里的反斜杠是为了转义正则里的特殊含义。
# 注意没有包含「，」和「、」——它们只是句内停顿，在这里切开会把一句话劈成两半。
_SENTENCE_END = re.compile(r"[。！？；!?;\n]")

# Markdown 标题：
#   ^(#{1,6})   行首 1~6 个 #，用捕获组 () 记下有几个（级别）
#   \s+         后面至少一个空格
#   (.*?)       标题文字（非贪婪）
#   \s*#*\s*$   允许结尾带 ### 和空格（有些文档写 "## 标题 ##"）
#   re.MULTILINE 不需要，因为我们是逐行匹配的
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass(slots=True)
class TextChunk:
    """一个切片。

    字段说明：
        content     切出来的正文（已经去掉首尾空白）
        title       标题路径，例如 "四、深度学习 > 4.2 CNN"；没有标题就是空字符串
        char_start  在**本页清洗后文本**里的起始偏移
        char_end    结束偏移（左闭右开）

    ⚠️ 偏移是「页内」的，不是「整篇文档」的。
    因为解析器按页返回文本（PDF 的每一页），每页单独切分，页码则由调用方记在 chunk 上。
    这样做的好处：引用能精确定位到「第 7 页第 123 个字」。
    """

    content: str
    title: str
    char_start: int
    char_end: int


def _heading_of(line: str) -> tuple[int, str] | None:
    """判断一行是不是 Markdown 标题。

    参数：
        line  一行文本（可以带换行符，内部会 strip）

    返回：
        (级别, 标题文字)，例如 (2, "四、深度学习")
        不是标题则返回 None

    返回类型 `tuple[int, str] | None` 是 Python 3.10+ 的写法，等价于
    Union[tuple[int, str], None]，意思是「要么是二元组，要么是 None」。
    """
    # strip() 去掉首尾空白：很多 PDF 转出来的 markdown 标题前面有空格
    matched = _HEADING.match(line.strip())
    if not matched:
        return None

    # group(1) 是第一个捕获组（# 的个数），group(2) 是标题文字
    level = len(matched.group(1))
    text = matched.group(2).strip()

    # 空标题（比如只有 "## "）当普通行处理，否则标题路径里会出现空字符串
    return (level, text) if text else None


def _split_long_text(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """把一段超长文本切成若干不超过 max_chars 的小段。

    什么时候需要它：
        一整段没有任何换行和句号（代码块、表格、从 PDF 里抽出来的连续文本），
        按行扫描根本切不开，只能在这里硬切。

    参数：
        text       全文（用绝对下标，这样算出来的偏移天然正确）
        start      这一段在 text 里的起点
        end        这一段在 text 里的终点（左闭右开）
        max_chars  每段上限

    返回：
        [(s1,e1), (s2,e2), ...]，这些区间拼起来正好覆盖 [start, end) 且互不重叠
    """
    spans: list[tuple[int, int]] = []
    cursor = start

    while cursor < end:
        # 剩下的还不够一段，直接收尾
        if end - cursor <= max_chars:
            spans.append((cursor, end))
            break

        # 取一个「窗口」，在窗口里找最后一个句子结束符
        window = text[cursor : cursor + max_chars]
        cut = -1
        for m in _SENTENCE_END.finditer(window):
            cut = m.end()  # end() 是「结束符之后」的位置，正好是下一句的开头

        # 找不到断点，或断点太靠前（会让块小到没有意义）→ 硬切
        # 为什么用 max_chars // 3 这个阈值？
        #   如果窗口前 1/3 就出现了句号，在那儿切会得到一堆 30 字的碎片，
        #   上下文严重缺失。宁可硬切在 900 字处，也不要碎成渣。
        if cut <= max_chars // 3:
            cut = max_chars

        spans.append((cursor, cursor + cut))
        cursor += cut

    return spans


def _trim(text: str, start: int, end: int) -> tuple[str, int, int]:
    """去掉首尾空白，并**同步修正偏移**。

    为什么必须做：
        块开头如果是一堆空行，既浪费 token，也让前端高亮很难看。
        但去掉空白后，char_start 必须跟着往后挪，否则
        「text[char_start:char_end] == content」这条不变量就不成立了。

    参数：
        text   全文
        start  候选起点
        end    候选终点

    返回：
        (去空白后的内容, 修正后的起点, 修正后的终点)
        如果内容全是空白，返回 ("", start, start)
    """
    raw = text[start:end]
    stripped = raw.strip()
    if not stripped:
        return "", start, start

    # 左边被去掉了几个字符，起点就要往后挪几个
    lead = len(raw) - len(raw.lstrip())

    # 这里用 raw.lstrip() 的长度差而不是重新 search，是因为 strip 的规则
    # 和 lstrip 一致，算出来最准。
    new_start = start + lead
    return stripped, new_start, new_start + len(stripped)


def split_text(
    text: str,
    max_chars: int = 900,
    min_chars: int = 120,
) -> list[TextChunk]:
    """主入口：把一页文本切成 chunk 列表。

    谁调用它：ingestion/pipeline.py（每个 page 调一次）

    参数：
        text       清洗过的页面文本。
                   ⚠️ 偏移量是相对这个字符串算的，所以**切分之后不能再改文本**
                   （比如再 replace 一下），否则所有引用都会错位。
                   这也是为什么清洗必须在切分之前一次性做完（见 cleaner.py）。
        max_chars  单块目标上限（字符数）。调用方传 settings.chunk_size。
        min_chars  单块下限。太碎的块会在末尾被合并进前一块。

    返回：
        list[TextChunk]，顺序与原文一致

    实现分三步：
        1. 按行扫描，遇到标题就收尾、更新标题栈
        2. 块超长就在句子边界切开
        3. 第二遍扫描，把过短的块合并（避免出现"目录"这种一行一块的碎片）
    """
    # 空文本直接返回空列表，避免后面 splitlines 之类的操作浪费时间
    if not text.strip():
        return []

    # splitlines(keepends=True) 保留每行末尾的 \n。
    # 保留的好处：把所有行拼起来正好等于原文，偏移量可以直接累加。
    lines = text.splitlines(keepends=True)

    chunks: list[TextChunk] = []
    title_stack: list[tuple[int, str]] = []  # [(级别, 标题), ...]
    buf_start: int | None = None  # 当前块在原文中的起点；None 表示"还没开始攒"
    buf_end = 0                   # 当前块在原文中的终点
    pos = 0                       # 扫描游标（当前行的起始下标）

    def current_title() -> str:
        """把标题栈拼成 "一级 > 二级" 的形式。

        这是闭包（nested function），能直接读写外层的 title_stack。
        用闭包而不是传参，是因为它在循环里会被调用很多次，传参会让代码更啰嗦。
        """
        return " > ".join(t for _, t in title_stack)

    def flush() -> None:
        """把「当前攒着的这块」收尾，追加到 chunks 里。

        nonlocal buf_start：声明我要修改外层函数的变量，而不是新建一个局部变量。
        这是 Python 闭包写法的必要声明，漏了会导致「改了但没生效」这种诡异现象。
        """
        nonlocal buf_start
        if buf_start is None:
            return
        content, s, e = _trim(text, buf_start, buf_end)
        if content:
            chunks.append(
                TextChunk(content=content, title=current_title(), char_start=s, char_end=e)
            )
        buf_start = None

    for line in lines:
        line_start = pos
        line_end = pos + len(line)
        pos = line_end

        heading = _heading_of(line)

        if heading:
            # ---- 遇到标题 ----
            # 先把上一节的内容收尾（保证一个块不跨小节）
            flush()

            level, title = heading
            # 维护标题栈：把「级别 >= 当前级别」的都弹掉。
            # 例：栈是 [（1,"总览"）,（2,"细节"）]，来了一个 (2, "另一节")，
            #     就要弹掉旧的 (2,"细节")，变成 [（1,"总览"）,（2,"另一节"）]。
            while title_stack and title_stack[-1][0] >= level:
                title_stack.pop()
            title_stack.append((level, title))

            # 标题本身也作为块的起点（留在正文里），这样模型能看到
            # "## 四、深度学习" 这行，理解这段话属于哪个主题。
            buf_start = line_start
            buf_end = line_end
            continue

        if buf_start is None:
            # 块还没开始（文件开头，或上一次刚好 flush 完）
            if not line.strip():
                continue  # 跳过开头的空行，避免块以空行起始
            buf_start = line_start

        # ---- 会超长吗？----
        if line_end - buf_start > max_chars:
            # 把已有内容按句子边界切开，最后一段留给后续行继续拼接
            spans = _split_long_text(text, buf_start, line_end, max_chars)
            for s, e in spans[:-1]:  # 注意是 [:-1]：最后一段不在这里收尾
                content, ss, ee = _trim(text, s, e)
                if content:
                    chunks.append(
                        TextChunk(content=content, title=current_title(), char_start=ss, char_end=ee)
                    )
            buf_start = spans[-1][0]  # 最后一段作为新块的开头，继续往后攒

        buf_end = line_end

    flush()  # 循环结束，把最后一块收尾

    # ------------------------------------------------------------------
    # 第三遍：合并过短的块
    # ------------------------------------------------------------------
    # 什么情况会产生碎片？
    #   只写了一行的段落、表格的小尾巴、被标题隔开的很短的一节。
    # 这些块单独送进向量库，检索出来毫无上下文，回答质量会很差。
    if min_chars > 0 and len(chunks) > 1:
        merged: list[TextChunk] = [chunks[0]]
        for chunk in chunks[1:]:
            prev = merged[-1]
            # 三个条件同时满足才合并：
            #   1. 前一块太短
            #   2. 两块属于同一小节（标题相同），否则合并会破坏标题路径的准确性
            #   3. 合并后不会超出 max_chars 太多（1.3 倍是留的余量）
            if (
                len(prev.content) < min_chars
                and prev.title == chunk.title
                and len(prev.content) + len(chunk.content) <= max_chars * 1.3
            ):
                # 注意：这里必须从原文重新切片，不能写成 prev.content + "\n" + chunk.content。
                # 因为两个块在原文里中间可能还隔着空行/标题行，
                # 用拼接的字符串会让 content != text[char_start:char_end]，
                # 引用跳转就会偏移几个字符——这种小偏差肉眼根本看不出来。
                merged[-1] = TextChunk(
                    content=text[prev.char_start : chunk.char_end],
                    title=prev.title,
                    char_start=prev.char_start,  # 起点沿用前一块
                    char_end=chunk.char_end,     # 终点用后一块
                )
            else:
                merged.append(chunk)
        chunks = merged

    return chunks
