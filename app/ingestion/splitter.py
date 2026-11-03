"""把一页文本切成 chunk，并记下每块在原文里的起止位置。

切分本身用 LangChain 官方的 RecursiveCharacterTextSplitter（中文分隔符 + overlap），
它负责「怎么把长文本切成接近 max_chars 的块」这件事，比手写更成熟。

官方没提供的三样东西，在这里补齐：
    1. 按 Markdown 标题分节 —— 官方切分器不认标题层级
    2. 标题路径 —— 每块标注 "一级 > 二级"，供引用卡片展示
    3. 短块合并 —— 官方没有 min_chars，会切出 10 来个字的碎片

不变量：text[char_start:char_end] == content（tests/test_splitter.py 守着这条）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Markdown 标题：行首 1~6 个 #，后面至少一个空格；允许结尾带 "###"
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

# 切分优先级：先按空行（段落），再按换行，再按中文句读，最后才硬切。
# keep_separator="end" 让标点留在前一块末尾，块内容才是连续的原文切片。
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


@dataclass(slots=True)
class TextChunk:
    """一个切片。content 严格等于 text[char_start:char_end]。"""

    content: str
    title: str
    char_start: int
    char_end: int


def _heading_of(line: str) -> tuple[int, str] | None:
    """是标题行就返回 (级别, 标题文字)，否则 None。"""
    matched = _HEADING.match(line.strip())
    if not matched:
        return None
    text = matched.group(2).strip()
    return (len(matched.group(1)), text) if text else None


def _sections(text: str) -> list[tuple[str, int, int]]:
    """按标题把全文切成节，返回 [(标题路径, 起点, 终点), ...]。

    标题栈维护「当前在哪一节」，遇到同级或更深的标题就把它们弹掉。
    这样 900 字的切分只发生在节内部，一块不会跨两个标题。
    """
    sections: list[tuple[str, int, int]] = []
    stack: list[tuple[int, str]] = []
    start = 0
    pos = 0

    for line in text.splitlines(keepends=True):
        heading = _heading_of(line)
        if heading:
            # 上一节到此结束
            if pos > start:
                sections.append((" > ".join(t for _, t in stack), start, pos))

            level, title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            start = pos
        pos += len(line)

    if pos > start:
        sections.append((" > ".join(t for _, t in stack), start, pos))
    return sections


def _trim(text: str, start: int, end: int) -> tuple[str, int, int]:
    """去掉首尾空白，并同步修正偏移，保证 text[s:e] == content。"""
    raw = text[start:end]
    stripped = raw.strip()
    if not stripped:
        return "", start, start
    lead = len(raw) - len(raw.lstrip())
    new_start = start + lead
    return stripped, new_start, new_start + len(stripped)


def _merge_short(
    text: str, chunks: list[TextChunk], max_chars: int, min_chars: int
) -> list[TextChunk]:
    """把过短的块合并进前一块。

    官方切分器没有下限，短段落会单独成为一块（10 来字），送去向量库毫无上下文。
    合并的三个条件：前一块太短、两块同一小节、合并后不超出上限太多。
    合并时从原文重新切片，不能拼接字符串，否则偏移会对不上。
    """
    if min_chars <= 0 or len(chunks) <= 1:
        return chunks

    merged: list[TextChunk] = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        if (
            len(prev.content) < min_chars
            and prev.title == chunk.title
            and len(prev.content) + len(chunk.content) <= max_chars * 1.3
        ):
            merged[-1] = TextChunk(
                content=text[prev.char_start : chunk.char_end],
                title=prev.title,
                char_start=prev.char_start,
                char_end=chunk.char_end,
            )
        else:
            merged.append(chunk)
    return merged


def split_text(
    text: str,
    max_chars: int = 900,
    min_chars: int = 120,
    overlap: int = 100,
) -> list[TextChunk]:
    """把一页文本切成 chunk 列表。

    参数：
        text       清洗过的页面文本；切完不能再改它，否则偏移全废
        max_chars  单块目标上限（settings.chunk_size）
        min_chars  单块下限，过短的块会被合并（settings.chunk_min_size）
        overlap    相邻块的重叠字数（settings.chunk_overlap），只在同一小节内部重叠

    返回：
        list[TextChunk]，顺序与原文一致
    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        keep_separator="end",
        add_start_index=True,  # 让每块带上在输入文本中的起始下标
    )

    chunks: list[TextChunk] = []
    for title, start, end in _sections(text):
        piece = text[start:end]
        # 只在当前小节内部切，避免一块跨两个标题
        for doc in splitter.create_documents([piece]):
            offset = start + doc.metadata["start_index"]
            content, s, e = _trim(text, offset, offset + len(doc.page_content))
            if content:
                chunks.append(
                    TextChunk(content=content, title=title, char_start=s, char_end=e)
                )

    return _merge_short(text, chunks, max_chars, min_chars)
