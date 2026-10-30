"""用真实问题校准检索阈值（SCORE_THRESHOLD）。

背景：`SCORE_THRESHOLD` 是「检索结果够不够好」的判断线，它决定要不要多花一次
LLM 调用去扩写重查。这个值**不能拍脑袋**——不同 rerank 模型的分数分布差别很大，
拍高了每次都在重查（浪费钱），拍低了该重查的时候不重查（答非所问）。

这个脚本对每个问题只做「检索 + 重排」，**不生成答案**，所以：
    - 不消耗生成模型的 token
    - 几秒钟出结果
    - 每个问题只花一次 embedding + 一次 rerank

跑法：
    .venv\\Scripts\\python.exe scripts\\calibrate.py

看什么：
    「同领域」问题和「无关」问题的最高分之间有没有明显间隔。
    有明显的间隔 → 阈值取间隔中间值。
    没有间隔 → 说明这个 rerank 模型区分度不够，或者阈值这条路本身不靠谱，
              该改用「条数 + 相对分差」来判断。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.models.orm import KnowledgeBase  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402

log = get_logger("calibrate")

# 同领域问题：这些应该拿到高分
IN_DOMAIN = [
    "Transformer 的核心机制是什么？",
    "AlexNet 是哪一年拿的冠军？",
    "深度学习爆发的推动因素有哪些？",
    "RAG 有什么价值？",
    "达特茅斯会议提出了什么？",
]

# 无关问题：这些应该拿到低分（如果分也高，说明阈值判断不可靠）
OUT_OF_DOMAIN = [
    "红烧肉怎么做才不腻？",
    "2024 年欧洲杯冠军是谁？",
    "如何用 Rust 写一个 HTTP 服务器？",
]


def score_of(question: str, kb_ids: list[int]) -> tuple[float, float, int]:
    """返回 (最高分, 平均分, 条数)。"""
    hits = retrieve(question, kb_ids)
    if not hits:
        return 0.0, 0.0, 0
    scores = [h.score for h in hits]
    return max(scores), sum(scores) / len(scores), len(hits)


def main() -> int:
    setup_logging()
    db = SessionLocal()
    try:
        kbs = db.query(KnowledgeBase).all()
        if not kbs:
            log.error("还没有知识库。先启动服务、建库并上传文档。")
            return 1
        kb = kbs[0]
        kb_ids = [kb.id]
        log.info("使用知识库「%s」(id=%s)，当前阈值 %.2f", kb.name, kb.id, settings.score_threshold)
    finally:
        db.close()

    log.info("")
    log.info("%-38s %8s %8s %4s", "问题", "最高分", "平均分", "条数")
    log.info("-" * 64)

    in_scores: list[float] = []
    for q in IN_DOMAIN:
        top, avg, n = score_of(q, kb_ids)
        in_scores.append(top)
        log.info("%-38s %8.4f %8.4f %4d", q, top, avg, n)

    log.info("")
    out_scores: list[float] = []
    for q in OUT_OF_DOMAIN:
        top, avg, n = score_of(q, kb_ids)
        out_scores.append(top)
        log.info("%-38s %8.4f %8.4f %4d", q, top, avg, n)

    log.info("-" * 64)
    log.info("同领域最高分：min=%.4f max=%.4f", min(in_scores), max(in_scores))
    log.info("无关问题最高分：min=%.4f max=%.4f", min(out_scores), max(out_scores))

    gap_low, gap_high = max(out_scores), min(in_scores)
    if gap_high > gap_low:
        suggested = round((gap_low + gap_high) / 2, 2)
        log.info("两类之间有间隔，建议 SCORE_THRESHOLD = %.2f", suggested)
    else:
        log.info("两类分数有重叠（无关问题也能拿高分），说明阈值不可靠：")
        log.info("  建议改用「条数 + 相对分差」判断，或换更强的 rerank 模型")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
