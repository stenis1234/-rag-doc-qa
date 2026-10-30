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


def clean_text(text: str) -> str:
    """基础清洗。宁可少做，也不要过度处理——过度清洗会改变语义。"""
    if not text:
        return ""

    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 全角空格转半角，避免切分和检索时被当成「空行」
    text = text.replace("\u3000", " ")
    text = _INVISIBLE.sub("", text)
    # 行尾空白
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)

    return text.strip()
