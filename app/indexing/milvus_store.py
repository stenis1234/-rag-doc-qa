"""Milvus 的 collection 定义与读写。

写入：app/ingestion/pipeline.py → upsert()
检索：app/retrieval/hybrid.py  → hybrid_search()
删除：app/services/*            → delete_by_doc / delete_by_kb

schema 的四个要点：
    1. 主键用 "12_00003" 这种确定性字符串，重复索引就是覆盖写，不会产生脏数据
    2. 带 doc_id / kb_id / seq / page_no / char_start / char_end，引用才能跳原文
    3. dense 由客户端算，保证文档侧和查询侧用同一个模型
    4. sparse 交给服务端 BM25 Function 生成，客户端插入时不要传
"""

from __future__ import annotations

# ---- 标准库 ----
# dataclass: 自动生成 __init__/__repr__，用来定义 ChunkHit / ChunkRow 这种纯数据容器
# field: 用来给 dataclass 字段设默认值（可变默认值必须用它，否则所有实例会共享同一个 list）
from dataclasses import dataclass, field

# ---- 第三方库：pymilvus ----
# 这几个类都来自 pymilvus（Milvus 官方 Python SDK），安装后在 site-packages/pymilvus 下：
#   MilvusClient        高层客户端，封装了 collection 的增删改查（底层走 gRPC 到 19530 端口）
#   DataType            字段类型枚举：VARCHAR / INT32 / FLOAT_VECTOR / SPARSE_FLOAT_VECTOR ...
#   Function            「函数」的定义，用来让服务端在写入时自动生成字段（BM25 就是靠它）
#   FunctionType        函数类型枚举：BM25 / TEXTEMBEDDING / RERANK
#   AnnSearchRequest    一次「单路检索请求」，hybrid_search 要把多路请求打包传进去
#   RRFRanker           倒数排名融合（Reciprocal Rank Fusion）的实现，见 hybrid.py 里的解释
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

# ---- 项目内部 ----
# settings: 全局配置（collection 名、向量维度都在这里）
from app.core.config import settings

# get_logger: 带模块名的 logger
from app.core.logging import get_logger

# get_client: 拿到 MilvusClient 单例；这里只负责「连接」，不负责建表
from app.core.milvus import get_client

# build_filter: 拼 Milvus 的过滤表达式（`kb_id in ["1"]` 这种），纯字符串函数，可单测
from app.indexing.expression import build_filter

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# 长度常量
# ---------------------------------------------------------------------------
# Milvus 的 VARCHAR 必须显式给 max_length，而且**按字符数**算（不是字节）。
# content 给 4096 是留余量：切分时按 900 字左右切，中文标点/换行会占额外空间，
# 而且有些表格行会被合并成一个长块。给太小会在插入时报「length exceeds max length」，
# 那种错发生在半夜的批量导入里，非常难查，所以宁可给大。
CONTENT_MAX_LEN = 4096
TITLE_MAX_LEN = 512
ID_MAX_LEN = 64

# 检索时要从 Milvus 带回来的标量字段（正文 + 引用定位信息）。
# 注意：不在这里的字段，检索结果里就没有，前端也就拿不到。
OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "kb_id",
    "content",
    "title",
    "page_no",
    "char_start",
    "char_end",
]


@dataclass(slots=True)
class ChunkHit:
    """一条检索结果。

    字段名和 Milvus schema 一一对应，方便互转；doc_name 是例外——
    它不在 Milvus 里，由 pipeline/rag.py 查关系库补上。

    为什么 score 和 rerank_score 要分开存？
        因为要知道「这条是融合分还是精排分」，排查问题时才能判断是哪一层出了偏差。
    """

    chunk_id: str            # 主键，形如 "12_00003"
    doc_id: str              # 所属文档 id（字符串形式，和 Milvus 的 VARCHAR 对齐）
    kb_id: str               # 所属知识库 id
    content: str             # 正文
    title: str = ""          # 标题路径，例如 "四、深度学习 > 4.2 CNN"
    page_no: int = 0         # 页码，从 1 开始；0 表示该文档没有页码概念
    char_start: int = 0      # 在「该页文本」里的起始字符偏移
    char_end: int = 0        # 结束偏移（左闭右开）
    score: float = 0.0       # 当前排序分（融合分或精排分，取决于是否开了 rerank）
    rerank_score: float | None = None  # 精排分；None 表示没做精排或精排失败
    doc_name: str = ""       # 文件名，Milvus 里没有，检索后补齐

    def to_citation(self, index: int, snippet_len: int = 160) -> dict:
        """转成给前端用的引用结构。

        谁调用它：generation/citation.py::build_citations()

        参数：
            index        答案里的角标编号（从 1 开始），前端用它和正文 [n] 对应
            snippet_len  摘要截断长度。为什么要截？
                         前端引用卡片是小卡片，塞 900 字会撑爆布局；
                         用户真要看全文，点开时会用 chunk_id 去 /api/v1/chunks/{id} 取。

        返回：
            dict，字段和 models/schemas.py 里的 Citation 对齐
        """
        # 把连续空白（含换行）压成单个空格：前端用 <pre> 之外的容器展示，
        # 留着换行会让卡片高度忽高忽低，很难看。
        text = " ".join(self.content.split())
        snippet = text[:snippet_len] + ("…" if len(text) > snippet_len else "")
        return {
            "index": index,
            "chunk_id": self.chunk_id,
            # doc_id 在 Milvus 里是字符串，转成 int 才符合接口定义；
            # 转不了就退化成 0（脏数据不应该让整个响应 500）
            "doc_id": int(self.doc_id) if self.doc_id.isdigit() else 0,
            "doc_name": self.doc_name,
            "title": self.title,
            "page_no": self.page_no,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "snippet": snippet,
            # 保留 4 位小数够用了，太长只会让前端卡片显示不下
            "score": round(float(self.score), 4),
        }


@dataclass(slots=True)
class ChunkRow:
    """写 Milvus 时的一行。

    和 ChunkHit 分开定义，是因为写入时多一个 dense 向量，而检索结果里不需要它。
    硬要塞进同一个类，会出现「检索结果里 dense 永远是空」这种莫名其妙的字段。
    """

    chunk_id: str
    doc_id: str
    kb_id: str
    content: str
    dense: list[float]       # 1024 个 float
    title: str = ""
    page_no: int = 0
    seq: int = 0             # 块在文档内的顺序号，将来做「前后文扩展」要用
    char_start: int = 0
    char_end: int = 0
    extra: dict = field(default_factory=dict)  # 预留：以后想加自定义字段

    def to_dict(self) -> dict:
        """转成 pymilvus 的 upsert 期望的 dict 格式。

        ⚠️ 关键：**不要**在返回的 dict 里放 sparse。
        sparse 由服务端的 BM25 Function 在写入时自动生成，
        客户端手动传会和 Function 冲突，报「field sparse is generated by function」。
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "kb_id": self.kb_id,
            "content": self.content,
            "dense": self.dense,
            "title": self.title,
            "page_no": self.page_no,
            "seq": self.seq,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


class MilvusStore:
    """collection 的门面（facade）。

    所有对 Milvus 的直接操作都收敛在这个类里。好处：
        1. 想换向量库（Milvus → Qdrant/Milvus Lite），只改这一个文件
        2. 想给所有操作加日志/埋点/重试，也只改这里
        3. 单元测试可以用假 client 替换（构造函数接受 client 参数就是为这个）
    """

    def __init__(self, client: MilvusClient | None = None, collection: str | None = None):
        """参数都给了默认值，所以正常业务代码 `MilvusStore()` 就够了。

        显式传参的场景：
            scripts/smoke_test.py 里传 collection="rag_chunks_smoke"，避免污染正式数据
        """
        self.client = client or get_client()
        self.collection = collection or settings.collection

    # ======================================================================
    # 建表
    # ======================================================================
    def exists(self) -> bool:
        """collection 是否存在。对应 pymilvus 的 MilvusClient.has_collection()。"""
        return self.client.has_collection(self.collection)

    def create(self, drop: bool = False) -> None:
        """建 collection（表结构 + 索引 + BM25 函数）。

        谁调用它：scripts/init_milvus.py（部署时跑一次）、scripts/dev_check.py

        参数：
            drop=False  已存在就跳过（安全，可重复执行）
            drop=True   先删再建（**会丢数据**，只在改了 schema 时用）

        一遍建表包含五件事，下面按顺序标注：
            1. 定义 schema（字段）
            2. 定义 BM25 Function（服务端自动生成 sparse）
            3. 定义索引（dense / sparse / 标量）
            4. 创建 collection
            5. load 到内存（不 load 无法检索）
        """
        if self.exists():
            if not drop:
                log.info("collection %s 已存在，跳过创建", self.collection)
                return
            log.warning("删除并重建 collection: %s", self.collection)
            self.client.drop_collection(self.collection)

        # ----- 1. schema -----
        # create_schema() 来自 pymilvus 的 MilvusClient，返回一个 CollectionSchema 对象
        # auto_id=False            主键由我们自己给（因为要用确定性 ID "12_00003"）
        # enable_dynamic_field=False  只允许写入声明过的字段。
        #   这个选项很重要：开了的话，写错字段名（比如 title 写成 tittle）不会报错，
        #   而是被当成一个「动态字段」悄悄存下来，检索时永远取不到，排查起来很痛苦。
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        # --- 主键 ---
        # DataType.VARCHAR 对应 Milvus 的字符串类型，必须给 max_length
        schema.add_field(
            "chunk_id", DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LEN
        )

        # --- 过滤 / 溯源字段 ---
        # kb_id 用于「只搜这个知识库」；doc_id 用于「删这个文档的所有块」
        schema.add_field("doc_id", DataType.VARCHAR, max_length=ID_MAX_LEN)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=ID_MAX_LEN)
        # seq 是块在文档内的序号，将来做「取前后各一段」的上下文扩展要用
        schema.add_field("seq", DataType.INT32)

        # --- 正文 ---
        # enable_analyzer=True + type="chinese" 让 Milvus 用中文分词器处理 content，
        # 这是 BM25 Function 能正常工作的前提（不分词的话整句会被当成一个词）。
        schema.add_field(
            "content",
            DataType.VARCHAR,
            max_length=CONTENT_MAX_LEN,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        # title 存完整的标题路径（"四、深度学习 > 4.2 CNN"），比只存 h2 好：
        # 同名的二级标题在不同章节里不会混在一起。
        schema.add_field("title", DataType.VARCHAR, max_length=TITLE_MAX_LEN)
        # 引用定位三件套：第几页、页内第几个字到第几个字
        schema.add_field("page_no", DataType.INT32)
        schema.add_field("char_start", DataType.INT32)
        schema.add_field("char_end", DataType.INT32)

        # --- 向量 ---
        # FLOAT_VECTOR：普通稠密向量，维度必须和 embedding 模型输出一致（1024）
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.embed_dim)
        # SPARSE_FLOAT_VECTOR：稀疏向量，只有少数维度有值，由下面 BM25 自动生成
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        # ----- 2. BM25 Function -----
        # 作用：插入数据时，Milvus 服务端自动做三件事
        #   1) 用中文分词器把 content 切成词
        #   2) 用 BM25 算法算出每个词的权重
        #   3) 把结果写进 sparse 字段
        # 所以客户端插入时只给 content，不要给 sparse。
        schema.add_function(
            Function(
                name="content_bm25",                 # 函数名，collection 内唯一
                function_type=FunctionType.BM25,     # 固定值：BM25 全文检索
                input_field_names=["content"],       # 读哪个字段
                output_field_names=["sparse"],       # 写哪个字段
            )
        )

        # ----- 3. 索引 -----
        # prepare_index_params() 返回 IndexParams 对象，用 add_index 逐个添加
        index_params = self.client.prepare_index_params()
        # dense 用 AUTOINDEX：Milvus 会按数据量自动选 HNSW/IVF，省去调参
        # COSINE：余弦相似度，只看向量方向不看长度，是文本 embedding 的常用选择
        index_params.add_index(
            field_name="dense", index_type="AUTOINDEX", metric_type="COSINE"
        )
        # sparse 只能配倒排索引；metric_type 必须是 BM25（和上面的 Function 配套）
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        # 标量字段建 INVERTED 索引：让 kb_id / doc_id 的过滤和删除走索引而不是全表扫
        # 数据量小的时候感觉不出来，上万条以后差别很明显
        for field_name in ("kb_id", "doc_id"):
            index_params.add_index(field_name=field_name, index_type="INVERTED")

        # ----- 4. 创建 -----
        # 把 schema 和 index_params 一起传进去，Milvus 会一次把表和索引都建好
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

        # ----- 5. load -----
        # Milvus 是「存算分离」的：数据在磁盘/ObjectStorage，检索时必须先加载到查询节点的内存。
        # 不 load 就检索会报 "collection not loaded"。通常创建后会自动 load，
        # 但显式调一次更稳（重启 Milvus 后需要重新 load）。
        self.client.load_collection(self.collection)
        log.info("collection 创建完成: %s", self.collection)

    def drop(self) -> None:
        """删掉整个 collection。脚本和测试里用，业务代码不要调。"""
        if self.exists():
            self.client.drop_collection(self.collection)

    # ======================================================================
    # 写入
    # ======================================================================
    def upsert(self, rows: list[ChunkRow], batch_size: int = 200) -> int:
        """批量写入，主键已存在则覆盖。

        谁调用它：ingestion/pipeline.py::ingest_document()

        参数：
            rows        要写的行
            batch_size  每批多少行。为什么要分批？
                        一次塞几千行会让单次请求过大，超时后你无法判断
                        「到底写进去没有」；分批 + 确定性主键 = 重试永远安全。

        返回：
            实际写入的行数

        关于 upsert 语义：
            它 = 「存在就更新，不存在就插入」，按主键判断。
            因为 chunk_id 是确定性的（doc_id_序号），重复索引同一个文件
            会覆盖旧行，而不是产生重复。
        """
        if not rows:
            return 0

        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            # client.upsert 来自 pymilvus 的 MilvusClient
            # data 传 list[dict]，每个 dict 的 key 必须和 schema 字段名一致
            self.client.upsert(
                collection_name=self.collection,
                data=[r.to_dict() for r in batch],
            )
            total += len(batch)

        # flush 让数据立刻可被检索。
        # 不 flush 的话要等 Milvus 自己的刷盘周期（可能几十秒），
        # 表现就是「上传完成了，但提问搜不到」。
        self.client.flush(self.collection)
        log.info("写入 Milvus %d 行", total)
        return total

    # ======================================================================
    # 删除
    # ======================================================================
    def delete_by_doc(self, doc_id: str) -> None:
        """删掉某个文档的所有块。

        filter 语法是 Milvus 的表达式（不是 SQL），字符串必须用双引号。
        """
        self.client.delete(
            collection_name=self.collection, filter=f'doc_id == "{doc_id}"'
        )

    def delete_by_kb(self, kb_id: str) -> None:
        """删掉某个知识库的所有块（删库时用）。"""
        self.client.delete(
            collection_name=self.collection, filter=f'kb_id == "{kb_id}"'
        )

    def count(self) -> int:
        """当前行数。注意：Milvus 的 row_count 是统计值，可能有延迟。"""
        if not self.exists():
            return 0
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    # ======================================================================
    # 检索
    # ======================================================================
    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        kb_ids: list[str],
        top_k_dense: int,
        top_k_sparse: int,
        top_k: int,
    ) -> list[ChunkHit]:
        """稠密 + 稀疏双路召回，再用 RRF 融合。

        谁调用它：retrieval/hybrid.py::retrieve()

        参数：
            query_text    原始问题文本。稀疏那一路直接用它，Milvus 服务端会分词算 BM25
            query_vector  问题的稠密向量
            kb_ids        限定的知识库 id（字符串），会被拼成过滤表达式
            top_k_dense   稠密路取多少条
            top_k_sparse  稀疏路取多少条
            top_k         融合后返回多少条（进精排前的候选数）

        返回：
            ChunkHit 列表，score 是 RRF 融合分（0.03 量级，不是 0~1）

        为什么两路的 limit 可以不一样？
            可以按数据特点调：文档里专有名词多，就把稀疏路调大一点；
            口语化提问多，就把稠密路调大一点。
        """
        # build_filter 是纯字符串函数（见 indexing/expression.py），
        # 产出形如：kb_id in ["1", "2"]
        # 传 None 而不是 "" 给 pymilvus，表示「不加过滤条件」
        expr = build_filter(kb_ids=kb_ids) or None

        # AnnSearchRequest 来自 pymilvus，代表「一路检索请求」
        # 字段说明：
        #   data       检索用的输入。稠密传向量、稀疏传**文本**（让服务端算 BM25）
        #   anns_field 搜哪个向量字段
        #   param      检索参数；AUTOINDEX 场景留空即可，Milvus 会用索引建好时的参数
        #   limit      这一路返回多少条
        #   expr       过滤条件（注意：过滤是**先于**检索生效的，能显著减少搜索范围）
        reqs = [
            AnnSearchRequest(
                data=[query_vector],
                anns_field="dense",
                param={},
                limit=top_k_dense,
                expr=expr,
            ),
            AnnSearchRequest(
                data=[query_text],
                anns_field="sparse",
                param={},
                limit=top_k_sparse,
                expr=expr,
            ),
        ]

        # RRFRanker 来自 pymilvus，k=60 是论文默认值，一般不用改。
        # 换成 WeightedRanker(0.7, 0.3) 就是「按分数加权」，但那要自己调权重，
        # 且两路分数量纲不同，很容易调崩——这是选 RRF 的原因。
        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=reqs,
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
        )

        # 返回结构（pymilvus 的规定）：
        #   results = [ [hit1, hit2, ...] ]   外层是「每个 query 一组」，我们只传了一个 query
        #   hit = {"id":..., "distance":..., "entity": {"content": ..., "page_no": ...}}
        # 注意：distance 这里是 RRF 融合分，意义和「余弦相似度」完全不同。
        hits: list[ChunkHit] = []
        for hit in results[0] if results else []:
            # hit["entity"] 里放的是 output_fields 指定的标量字段
            entity = hit.get("entity", {})
            hits.append(
                ChunkHit(
                    chunk_id=str(entity.get("chunk_id", hit.get("id", ""))),
                    doc_id=str(entity.get("doc_id", "")),
                    kb_id=str(entity.get("kb_id", "")),
                    content=entity.get("content", ""),
                    title=entity.get("title", ""),
                    # 用 int(x or 0) 兜底：Milvus 对 INT32 字段可能返回 None
                    page_no=int(entity.get("page_no", 0) or 0),
                    char_start=int(entity.get("char_start", 0) or 0),
                    char_end=int(entity.get("char_end", 0) or 0),
                    score=float(hit.get("distance", 0.0)),
                )
            )
        return hits


# ---------------------------------------------------------------------------
# 模块级单例（延迟创建）
# ---------------------------------------------------------------------------
# 为什么是函数而不是 `store = MilvusStore()`？
#   模块级的 MilvusStore() 会在 **import 时**就连接 Milvus。
#   一旦 Milvus 没起来，所有 import 这个模块的地方（包括单元测试、脚本）
#   全部直接炸掉，你会看到一堆和 Milvus 无关的 ImportError。
_store: MilvusStore | None = None


def get_store() -> MilvusStore:
    """拿默认 store。第一次调用时才真正连接 Milvus。"""
    global _store
    if _store is None:
        _store = MilvusStore()
    return _store
