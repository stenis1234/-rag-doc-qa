"""检索层：query → 候选 chunk。

    hybrid.py   双路召回 + 融合（调 milvus_store）
    rerank.py   精排（调 DashScope）
    rewrite.py  多轮对话下的指代消解
"""
