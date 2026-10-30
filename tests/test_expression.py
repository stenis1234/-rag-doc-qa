"""Milvus 过滤表达式拼接的单元测试（纯字符串，不需要连数据库）。"""

from __future__ import annotations

import unittest

from app.indexing.expression import build_filter, quote


class TestExpression(unittest.TestCase):
    def test_empty_means_no_filter(self):
        self.assertEqual(build_filter(), "")

    def test_kb_filter(self):
        self.assertEqual(build_filter(kb_ids=["1", "2"]), 'kb_id in ["1", "2"]')

    def test_and_conditions(self):
        expr = build_filter(kb_ids=["1"], doc_ids=["7"])
        self.assertEqual(expr, 'kb_id in ["1"] and doc_id in ["7"]')

    def test_empty_list_is_always_false(self):
        """空列表必须是恒假，不能退化成「不过滤」——否则会跨库检索。"""
        self.assertEqual(build_filter(kb_ids=[]), "1 == 0")

    def test_quote_escapes(self):
        self.assertEqual(quote('a"b'), '"a\\"b"')
        self.assertEqual(quote("a\\b"), '"a\\\\b"')


if __name__ == "__main__":
    unittest.main()
