"""全局配置 —— 全项目唯一读环境变量的地方。

所有模块都 `from app.core.config import settings`。
值来自 .env，优先级：真实环境变量 > .env > 字段默认值。
字段名自动映射成大写环境变量（milvus_uri ← MILVUS_URI），有类型标注，非法值启动就报错。

⚠️ 改完 .env 必须重启进程；uvicorn --reload 只监听 .py，不会因 .env 变化而重启。
"""

from __future__ import annotations

# ---- 标准库 ----
# Path: 处理文件路径。用它可以跨平台拼接、判断存在性，比字符串拼接安全。
from pathlib import Path

# ---- 第三方库：pydantic-settings ----
# BaseSettings: 带"从环境变量/文件读取"能力的 Pydantic 模型
# SettingsConfigDict: 配置这个模型的行为（比如指定 .env 路径）
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：本文件位于 <root>/app/core/config.py
#   parents[0] = app/core
#   parents[1] = app
#   parents[2] = <root>
# 用绝对路径是为了避免"在哪个目录启动 uvicorn"影响配置和文件路径。
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """所有配置项。字段名 = 配置名（大小写不敏感）。"""

    # model_config 是 pydantic v2 的写法（v1 叫 class Config）。
    # env_file 指向项目根目录的 .env：
    #   用绝对路径，避免"从别的目录启动服务时找不到 .env"这种坑。
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",  # 不写这个，Windows 上可能用 GBK 读，中文注释会乱码
        extra="ignore",
    )

    # ======================================================================
    # 应用
    # ======================================================================
    app_name: str = "RAG 文档问答"
    debug: bool = True  # True 会把日志级别设成 DEBUG（见 core/logging.py）
    # 前端就是 web/ 下的静态文件，同源访问其实不需要 CORS。
    # 放开是为了方便你用别的端口/工具调试接口。
    cors_origins: list[str] = ["*"]

    # ======================================================================
    # 路径
    # ======================================================================
    # 这三个都在项目根下，跟着代码走，不会因为工作目录改变而失效
    data_dir: Path = BASE_DIR / "data"
    upload_dir: Path = BASE_DIR / "data" / "files"  # 原始文件落盘位置
    web_dir: Path = BASE_DIR / "web"                # 前端静态文件

    # ======================================================================
    # 业务库（关系型，存知识库/文档/会话/任务）
    # ======================================================================
    # 默认 SQLite：零配置，第一次启动自动创建文件。
    # as_posix() 把 Windows 的反斜杠转成正斜杠，否则 SQLAlchemy 的 URL 解析会出错。
    # 换 PostgreSQL 只需在 .env 里改：DATABASE_URL=postgresql+psycopg://...
    database_url: str = f"sqlite:///{(BASE_DIR / 'data' / 'rag.db').as_posix()}"

    # ======================================================================
    # Milvus（向量库）
    # ======================================================================
    milvus_uri: str = "http://127.0.0.1:19530"  # 19530 是 gRPC 端口，不是网页
    milvus_token: str = ""                       # 无鉴权时留空
    collection: str = "rag_chunks"               # collection 名（相当于表名）

    # ======================================================================
    # 向量化（DashScope）
    # ======================================================================
    dashscope_api_key: str = ""
    embed_model: str = "text-embedding-v3"
    # ⚠️ 这个值必须和 Milvus schema 里的 dim 一致。
    #    改了它就要重建 collection（scripts/init_milvus.py --drop），
    #    否则插入时会报 dimension mismatch。
    embed_dim: int = 1024
    # 单次请求的文本条数。DashScope 有限制，别调太大；
    # 调小只是慢一点，调大会直接报错。
    embed_batch_size: int = 10

    # ======================================================================
    # 精排（DashScope）
    # ======================================================================
    rerank_enabled: bool = True   # 关掉会快一点、省一次调用，但准确率下降
    rerank_model: str = "gte-rerank-v2"

    # ======================================================================
    # 生成（DeepSeek）
    # ======================================================================
    llm_model: str = "deepseek-flash"
    llm_api_key: str = ""   # 可选：显式指定；留空则用 DEEPSEEK_API_KEY
    # 直接读 .env 里的 DEEPSEEK_API_KEY（pydantic 不会把它自动塞进环境变量，
    # 而 LangChain 只认环境变量，所以这里显式接一下）
    deepseek_api_key: str = ""
    llm_base_url: str = ""  # 走自建网关/中转时填（OpenAI 兼容协议），见 generation/llm.py
    llm_temperature: float = 0.2  # 低温度 = 更忠实复述资料，少发挥
    # DeepSeek 的思考开关："" 用服务端默认；"disabled" 关闭（省 token，RAG 推荐）；
    # "enabled" 显式开启。关掉思考后首字更快，也不会出现大段推理消耗。
    llm_thinking: str = "disabled"

    # ======================================================================
    # 切分
    # ======================================================================
    # 900 字是怎么定的？
    #   太小（200）：一句话被切成两半，检索到半句，答不出完整意思
    #   太大（2000）：一个块里混了多个主题，向量被"平均"掉，检索精度下降
    #   900 左右是中文技术文档的经验值，你可以按自己文档的特点调
    chunk_size: int = 900
    # 低于这个长度的块会和相邻块合并，避免"目录行"这种碎片
    chunk_min_size: int = 120
    # 相邻块的重叠字数：一句话被切断时，前后两块都能看到上下文
    chunk_overlap: int = 100

    # ======================================================================
    # 检索
    # ======================================================================
    top_k_dense: int = 8     # 稠密路召回条数
    top_k_sparse: int = 8    # 稀疏（BM25）路召回条数
    top_k_candidate: int = 10 # 两路融合后、进精排前的候选数（精排最慢，候选少一半也更快）
    top_k_final: int = 4      # 最终进 prompt 的条数（5→4：省 20% 上下文，召回影响很小）

    # ---- 两个阈值，很容易混，这里说清楚 ----
    # min_rerank_score  单条候选的「入场券」：低于它直接丢掉，不进 prompt
    # score_threshold   「这次检索够不够好」的判断线：低于它触发扩写重查
    #
    # 这两个值不是拍脑袋定的，是用 scripts/calibrate.py 在真实数据上量出来的：
    #   同领域问题最高分 0.30~0.61，无关问题 0.006，中间空白很大
    # 换 embedding/rerank 模型后分布会变，要重新跑 calibrate.py
    min_rerank_score: float = 0.02
    score_threshold: float = 0.15
    max_retry: int = 1  # 最多重查几次（每次 = 一次 LLM 调用 + 一次检索，别设太大）

    # ======================================================================
    # 会话
    # ======================================================================
    # 带几轮历史给「改写」节点用。太大不仅费钱，还会让模型被旧话题带偏。
    history_turns: int = 3
    # 早期对话压缩成摘要：超过 history_turns*2 条后触发，摘要长度上限
    summary_max_chars: int = 400

    # ======================================================================
    # PDF 解析
    # ======================================================================
    # 填了就走 MinerU（版面还原 + OCR，质量高但丢页码）；
    # 留空则用 pypdf（页码准，但版面信息基本没有）。见 ingestion/parsers/pdf.py
    mineru_token: str = ""

    def ensure_dirs(self) -> None:
        """确保数据目录存在。

        谁调用它：app/main.py 启动时、core/db.py::init_db()、几个脚本
        为什么需要：第一次运行时 data/ 目录还不存在，
                    上传文件时会因为"目录不存在"直接 500。
                    parents=True 表示"父目录也一起建"，exist_ok 表示"已存在不报错"。
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
# 整个进程只实例化一次。在 import 时创建是安全的：读配置不涉及网络，
# 即使配置有错也会在启动阶段立刻暴露（而不是等到第一个请求才炸）。
settings = Settings()
