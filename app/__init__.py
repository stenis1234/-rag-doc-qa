"""RAG 文档问答项目。

分层约定（从外到内）：
    api       → 只做参数校验 / 调 service / 返回响应，不写业务逻辑
    services  → 业务编排，跨模块组合
    pipeline  → 一次问答的完整流程（可读性最高的地方，建议从这里开始读）
    retrieval / generation / indexing / ingestion → 各自领域的实现
    models / core → 数据模型与基础设施
"""
