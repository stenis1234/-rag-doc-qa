"""文本向量化（DashScope text-embedding-v3）。

写入：app/ingestion/pipeline.py → embed_documents()
检索：app/retrieval/hybrid.py   → embed_query()

两个坑：
    1. 文档侧和查询侧必须同一个模型、同一个维度，否则两边不在一个语义空间，
       表现是「检索有结果但都不相关」，且不报错。所以下面加了维度断言。
    2. DashScope 单次请求条数有上限（约 10 条），必须分批。
"""

from __future__ import annotations

# ---- 第三方库：dashscope ----
# dashscope 是阿里云百炼（通义千问）的官方 Python SDK。
# 用到的入口：
#   dashscope.api_key            全局设置密钥
#   dashscope.TextEmbedding.call(...)   文本向量化，等价于 HTTP POST /embeddings
# 返回的是 DashScope 自己的响应对象，不是 OpenAI 格式，所以下面要手动取 output["embeddings"]
import dashscope

# ---- 第三方库：tenacity ----
# tenacity 是重试库。装饰器写法：
#   @retry(stop=..., wait=...)
# 作用：函数抛异常时自动重试，次数和间隔由参数控制。
# 为什么需要它？调外部 API 一定会遇到网络抖动、限流（429），
# 不重试的话用户会看到莫名其妙的失败，而你重试一次通常就好了。
from tenacity import retry, stop_after_attempt, wait_exponential

# ---- 项目内部 ----
from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


class EmbeddingError(RuntimeError):
    """向量化失败。

    单独定义一个异常类，而不是直接抛 RuntimeError，是为了让上层能精确区分
    「模型服务出问题了」和「我们自己的代码有 bug」：
        except EmbeddingError  → 提示用户检查 key / 配额
        except Exception       → 真的去看日志
    """


def _setup_key() -> None:
    """把 key 塞给 dashscope 的全局变量。

    为什么不在模块顶部设置？
        因为 import 本模块时 .env 可能还没加载完；而且没配 key 时
        应该报一个「人话错误」，而不是在 import 阶段就崩掉。
    """
    if not settings.dashscope_api_key:
        raise EmbeddingError(
            "缺少 DASHSCOPE_API_KEY。请在项目根目录的 .env 里填上（参考 env.example），"
            "或直接设置同名环境变量，然后重启服务。"
        )
    dashscope.api_key = settings.dashscope_api_key


@retry(
    # 最多试 3 次（第 1 次 + 重试 2 次）
    stop=stop_after_attempt(3),
    # 指数退避：1s → 2s → 4s（上限 8s）。
    # 为什么不固定间隔？因为限流通常是"短时间内请求太多"，
    # 固定 1 秒重试很容易继续撞墙；退避能让服务端缓过来。
    wait=wait_exponential(multiplier=1, min=1, max=8),
    # reraise=True：重试 3 次还失败，就把原始异常抛出去（而不是抛 tenacity 的包装异常）。
    # 这样上层 except EmbeddingError 才能正确捕获。
    reraise=True,
)
def _call_batch(texts: list[str]) -> list[list[float]]:
    """调一次接口，转一批文本。

    参数：
        texts  一批文本（条数不能超过 settings.embed_batch_size）
    返回：
        向量列表，顺序与输入严格一致

    这个函数**不应该被外部直接调用**，统一走下面的 Embedder 类。
    """
    _setup_key()

    # TextEmbedding.call 是同步阻塞调用（内部用 requests），
    # 所以本函数在 FastAPI 里运行时要注意：它是跑在线程池里的（后台任务/普通同步接口），
    # 不会阻塞事件循环。如果哪天改成 async 接口直接调它，会卡住整个服务。
    resp = dashscope.TextEmbedding.call(
        model=settings.embed_model,
        input=texts,                    # 传 list 就是批量，传 str 就是单条
        dimension=settings.embed_dim,   # 显式指定维度，避免模型升级后默认值变化
    )

    # DashScope 的错误不是抛异常，而是返回非 200（是 DashScope 返回的响应状态码，和 HTTP 状态码一个含义：200 表示成功，其他都是失败）
    # 的 status_code，所以必须手动检查
    if resp.status_code != 200:
        raise EmbeddingError(f"DashScope embedding 失败 {resp.status_code}: {resp.message}")

    embeddings = resp.output["embeddings"]
    # embeddings中每个元素是一个字典，包含两个字段：
    # [
    #   {"text_index": 0, "embedding": [-0.0458, 0.0816, -0.0366, ...]},   # 1024 个数
    #   {"text_index": 1, "embedding": [-0.0312, 0.0554, ...]},      ]      # 1024 个数
    # 服务端返回的顺序**不保证**和输入一致（虽然通常一致），
    # 所以按 text_index 排回去。这一步不做的话，
    # 可能出现"第 1 条文本配了第 2 条的向量"——数据全错但完全不报错。
    embeddings = sorted(embeddings, key=lambda x: x.get("text_index", 0))
    return [item["embedding"] for item in embeddings]


class Embedder:
    """向量化的入口类。

    为什么用类而不是两个裸函数？
        需要记住 batch_size 这个状态。用类可以：
            embedder = Embedder(batch_size=25)   # 测试时改成小批量
        以后还能自然地在类里加缓存、限流、统计。
    """

    def __init__(self, batch_size: int | None = None):
        # 不传就用配置里的值（默认 10）,batch_size 是“每次调用 API 送多少条文本”。
        self.batch_size = batch_size or settings.embed_batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把一批 chunk 正文转向量。返回顺序与输入严格一致。

        谁调用它：ingestion/pipeline.py

        参数：
            texts  切分后的正文列表（可能几百条）
        返回：
            list[list[float]]，和 texts 一一对应
        """
        if not texts:
            return []

        vectors: list[list[float]] = []

        # range(0, n, step) 是 Python 里分批的惯用写法：
        #   texts[0:10], texts[10:20], ...
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs = _call_batch(batch)

            # 维度检查放在这里而不是 Milvus 那边，是为了给出更有用的错误信息。
            # 如果等到写 Milvus 才报错，你只会看到一句
            # "vector dimension mismatch"，还得自己去猜是哪个配置错了。
            for vec in vecs:
                if len(vec) != settings.embed_dim:
                    raise EmbeddingError(
                        f"向量维度对不上：模型 {settings.embed_model} 返回了 {len(vec)} 维，"
                        f"但配置的 EMBED_DIM 是 {settings.embed_dim}。\n"
                        "两项配置在 .env 或 app/core/config.py 的默认值里，改完需要：\n"
                        "  1) 重建 Milvus collection：python scripts/init_milvus.py --drop\n"
                        "  2) 重新上传文档（旧向量是按旧维度算的）"
                    )

            vectors.extend(vecs)
            # 用 debug 级别记录进度：生产环境不会刷屏，排查时可以调成 INFO
            log.debug(
                "embedding 进度 %d/%d", min(start + self.batch_size, len(texts)), len(texts)
            )

        return vectors

    def embed_query(self, text: str) -> list[float]:
        """把问题转向量。

        虽然只有一条，也走同一个 _call_batch，保证和文档侧用的是同一个模型、
        同一套参数。**不要**为了省一次函数调用而另写一个实现。
        """
        vectors = _call_batch([text])
        return vectors[0]


# 模块级单例：绝大多数地方直接 `from app.indexing.embedding import embedder` 即可。
# 这里可以安全地在 import 时创建，因为它不会发起任何网络请求（不发请求就不会失败）。
embedder = Embedder()
