"""应用入口。

启动顺序（lifespan 里）：
    1. 建数据目录
    2. 建表（SQLite 第一次跑会自动生成 data/rag.db）
    3. 探一下 Milvus，连不上只警告不退出——这样你还能打开页面看错误提示，
       而不是对着一个起不来的进程猜原因

跑起来：
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000
然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1 import chat, chunks, documents, jobs, kb
from app.core.config import settings
from app.core.db import init_db
from app.core.logging import get_logger, setup_logging
from app.core.milvus import ping
from app.indexing.milvus_store import MilvusStore

setup_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("启动 %s", settings.app_name)
    settings.ensure_dirs()
    init_db()

    if ping():
        store = MilvusStore()
        if not store.exists():
            log.warning(
                "collection「%s」不存在，请先运行：python scripts/init_milvus.py",
                settings.collection,
            )
        else:
            log.info("collection「%s」共 %d 条", settings.collection, store.count())
    else:
        log.error(
            "Milvus 连不上（%s），检索和上传后的索引都会失败。"
            "请确认 Milvus 已启动；地址在 .env 的 MILVUS_URI 或 config.py 默认值里。",
            settings.milvus_uri,
        )

    yield
    log.info("退出")


app = FastAPI(
    title=settings.app_name,
    description="文档问答 RAG：FastAPI + Milvus + LangChain",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 所有业务接口统一挂在 /api/v1 下，方便以后加版本
for router in (kb.router, documents.router, jobs.router, chat.router, chunks.router):
    app.include_router(router, prefix="/api/v1")


@app.get("/api/health", tags=["运维"], summary="健康检查")
def health() -> dict:
    """给运维/前端用的探活。Milvus 单独一个字段，方便定位是谁挂了。"""
    milvus_ok = ping()
    collection_ok = False
    if milvus_ok:
        try:
            collection_ok = MilvusStore().exists()
        except Exception:  # noqa: BLE001
            collection_ok = False
    return {
        "app": "ok",
        "milvus": "ok" if milvus_ok else "down",
        "collection": "ok" if collection_ok else "missing",
        "collection_name": settings.collection,
    }


# 前端是纯静态文件，挂到根路径。
# 必须放在所有路由之后：mount 会兜住所有没匹配上的路径。
if settings.web_dir.exists():
    app.mount("/", StaticFiles(directory=str(settings.web_dir), html=True), name="web")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """给前端静态资源加 no-store。

    开发时文件频繁改，但 StaticFiles 的缓存头很宽松，浏览器会直接拿缓存，
    结果就是「代码改了、页面还是旧样子」。这里禁掉缓存，刷新即生效。
    """
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/index.html") or path.endswith((".js", ".css")):
        response.headers["Cache-Control"] = "no-store"
    return response
