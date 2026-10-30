"""接口的请求/响应模型。

和 orm.py 分开是有意的：
    orm      → 数据库长什么样（可以随便改字段）
    schemas  → 接口长什么样（改了就是破坏性变更，前端要跟着改）
混在一起的话，某天你给表加个内部字段，接口就莫名其妙多返回一个字段。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class KBCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="知识库名称，唯一")
    description: str = Field("", max_length=255)


class KBOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    created_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kb_id: int
    filename: str
    size_bytes: int
    page_count: int
    status: str
    error: str | None = None
    created_at: datetime


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    doc_id: int
    status: str
    progress: int
    stage: str
    error: str | None = None


class UploadResult(BaseModel):
    """上传接口立刻返回的东西：文档 id + 任务 id（进度要另查）。"""

    doc_id: int
    job_id: str
    filename: str
    status: str


class DocumentPage(BaseModel):
    """一页的文本。page_no=0 表示这份文档没有页码概念（md / docx）。"""

    page_no: int
    text: str


class DocumentDetail(BaseModel):
    """文档详情：点文档名时用它把内容展示出来。

    pages 是从 chunk 反向拼回来的「按页文本」，不是原始文件字节。
    为什么不直接返回原始文件？
        因为用户想看的往往是「系统到底读到了什么」——
        PDF 解析出乱码时，看原文是看不出来的，看这里才知道。
        要看原始文件，另有 /api/v1/documents/{id}/file。
    """

    id: int
    filename: str
    status: str
    page_count: int
    chunk_count: int
    size_bytes: int
    error: str | None = None
    pages: list[DocumentPage] = []


class Citation(BaseModel):
    """一条引用。前端点角标时要靠 doc_id/chunk_id 去取原文。"""

    index: int              # 答案里的角标编号，从 1 开始
    chunk_id: str
    doc_id: int
    doc_name: str
    title: str
    page_no: int
    char_start: int
    char_end: int
    snippet: str            # 截断后的正文，列表里直接展示
    score: float            # 重排分数（没开重排就是融合分数）


class ChatRequest(BaseModel):
    kb_id: int
    question: str = Field(..., min_length=1, max_length=2000)
    # 传了就是继续这个会话；不传就新建一个
    conversation_id: int | None = None


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    citations: list[Citation] = []
    created_at: datetime
