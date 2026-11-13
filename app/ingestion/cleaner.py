"""文本清洗。

放在切分**之前**：切分算出来的 char_start/char_end 是相对清洗后的文本的，
所以清洗必须只在入口做一次，后面任何环节都不能再改动正文，
否则偏移全废、引用会跳错位置。
"""

from __future__ import annotations

import re

# 零宽字符、软连字符：PDF 里常见，肉眼看不见但会污染分词和检索
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]")
# 连续 3 个以上换行压成 2 个（保留段落感，去掉大片空白）
_MANY_NEWLINES = re.compile(r"\n{3,}")

# ---- 网页/维基百科另存为文本时残留的标记 ----
# 不清掉的话，引用弹窗里全是 [[...]]、{{...}}、**...**，检索也会被噪声干扰
_WIKI_ANCHOR = re.compile(r"\[\[#[^\]]*\]\]")                  # [[#sup-1]]
_WIKI_LINK = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")        # [[页面|显示文字]]
_WIKI_LINK_PLAIN = re.compile(r"\[\[([^\]]+)\]\]")             # [[页面]]
_WIKI_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")                 # {{模板}}
_WIKI_BOLD = re.compile(r"'{3,5}(.+?)'{3,5}")                  # '''粗体'''
_WIKI_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*=+\s*$", re.MULTILINE)  # == 标题 ==
_EXT_LINK = re.compile(r"\[\s*(?:https?://|//|/reference)[^\]]*\]")      # [url 文字]
_MD_LOCAL_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")                   # [文字](链接) → 文字
_ESCAPED_REF = re.compile(r"\\\[[^\]]*\]\**")                            # \[1-2]**** 这类残留脚注
_WIKI_ITALIC = re.compile(r"''(.+?)''")                                  # ''斜体''


def _clean_wiki(text: str) -> str:
    """清理维基/网页残留标记。只处理明确的语法，不碰普通标点。"""
    text = _WIKI_ANCHOR.sub("", text)
    text = _WIKI_LINK.sub(r"\2", text)
    text = _WIKI_LINK_PLAIN.sub(r"\1", text)
    # 模板可能嵌套，跑两遍基本能清干净
    for _ in range(2):
        text = _WIKI_TEMPLATE.sub("", text)
    text = _WIKI_BOLD.sub(r"\1", text)
    text = _WIKI_ITALIC.sub(r"\1", text)
    text = _WIKI_HEADING.sub(lambda m: "#" * (len(m.group(1)) - 1) + " " + m.group(2), text)
    text = _MD_LOCAL_LINK.sub(r"\1", text)
    text = _ESCAPED_REF.sub("", text)
    text = _EXT_LINK.sub("", text)
    return text


def clean_text(text: str) -> str:
    """基础清洗。宁可少做，也不要过度处理——过度清洗会改变语义。"""
    if not text:
        return ""

    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 全角空格转半角，避免切分和检索时被当成「空行」
    text = text.replace("\u3000", " ")
    text = _INVISIBLE.sub("", text)
    text = _clean_wiki(text)
    # 行尾空白
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)

    return text.strip()
