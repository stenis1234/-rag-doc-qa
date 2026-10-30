"""切分器的单元测试。

跑法（不需要任何第三方包）：
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from app.ingestion.splitter import split_text


class TestSplitter(unittest.TestCase):
    def test_offsets_match_content(self):
        """最重要的一条不变量：content 必须严格等于原文的对应区间。

        引用跳转、原文高亮全靠它。这条挂了，前端点角标就会跳到错误位置。
        """
        text = (
            "# 一、总览\n\n"
            "这是第一段内容，用来测试偏移是否正确。\n\n"
            "## 1.1 细节\n\n"
            "这是第二段内容，稍微长一点点，看看切分和偏移是否还对齐。\n"
        )
        chunks = split_text(text, max_chars=200, min_chars=0)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(text[chunk.char_start : chunk.char_end], chunk.content)

    def test_title_path(self):
        """标题路径要能表达层级，而不是只留最后一级。"""
        text = "## 四、深度学习\n\n内容 A\n\n### 4.2 CNN\n\n内容 B\n"
        chunks = split_text(text, max_chars=200, min_chars=0)
        titles = [c.title for c in chunks]
        self.assertIn("四、深度学习", titles)
        self.assertIn("四、深度学习 > 4.2 CNN", titles)

    def test_long_paragraph_is_split(self):
        """超长段落必须被切开，且每块不超过上限。"""
        text = "。".join(["这是一句测试用的中文内容"] * 200) + "。"
        chunks = split_text(text, max_chars=300, min_chars=0)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), 300)

    def test_short_chunk_merged(self):
        """过短的块会被合并，避免出现只有一行字的碎片。

        合并之后偏移仍然必须覆盖完整的原文区间——这里特意再断言一次，
        因为「合并」是最容易破坏 char_start/char_end 不变量的地方。
        """
        text = "## 标题\n\n短。\n\n另一段也不长，但是内容稍微多一点点。\n"
        chunks = split_text(text, max_chars=500, min_chars=100)
        self.assertEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(text[chunk.char_start : chunk.char_end], chunk.content)

    def test_empty(self):
        self.assertEqual(split_text("   \n\n  "), [])


if __name__ == "__main__":
    unittest.main()
