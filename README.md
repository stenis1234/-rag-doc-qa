# RAG 文档问答

一个可以跑起来的文档问答系统：**FastAPI + Milvus + LangChain 1.x**。
重点在「流程」和「分层」，不在前端花活。

当前完成度：**M0 + M1**（单库问答全链路 + 混合检索 + 重排 + 流式 + 引用回溯）。
M2/M3（Celery 异步、LangGraph 自纠错、RAGAS 评测）见文末路线图。

> **第一次读代码，先看 [docs/代码导读.md](docs/代码导读.md)**：
> 里面有两条链路的调用图、推荐阅读顺序、术语表，以及「哪个文件读完能回答什么问题」。

---

## 一、快速开始

### 1. 准备环境

需要一个跑着的 Milvus（本机 19530）。如果你之前的学习环境已经起过，直接跳过。

```bash
# 依赖（已装好可跳过）
uv pip install -r pyproject.toml
```

### 2. 配置

```bash
copy env.example .env      # Windows
# cp env.example .env      # macOS / Linux
```

然后编辑 `.env`，至少填两个 key：

```ini
DASHSCOPE_API_KEY=sk-...   # embedding + rerank
DEEPSEEK_API_KEY=sk-...    # 生成答案
```

模型名和你学习笔记里的一致（`text-embedding-v3` / `gte-rerank-v2` / `deepseek-flash`），
换成你环境里实际可用的即可。

### 3. 建 collection 和表

```bash
.venv\Scripts\python.exe scripts\init_milvus.py
.venv\Scripts\python.exe scripts\init_db.py
```

### 4. 先自检（不花钱）

```bash
.venv\Scripts\python.exe scripts\smoke_test.py   # 只测 Milvus 层
.venv\Scripts\python.exe scripts\dev_check.py    # 整条链路，embedding 用假向量
```

两个都输出 `[OK]` 再往下走。这一步能把 90% 的环境问题挡在模型调用之前。

### 5. 启动

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

打开 <http://127.0.0.1:8000>：

1. 左侧新建知识库 → 上传 `data/samples/ai.md` → 等进度条走完
2. 提问「Transformer 的核心机制是什么？」
3. 点答案里的 `[1]` 角标，看原文片段

接口文档在 <http://127.0.0.1:8000/docs>。

---

## 二、目录结构

```
app/
├── main.py            入口：装配路由、静态文件、生命周期
├── api/v1/            HTTP 层，只做「校验 → 调 service → 返回」
├── services/          业务编排：跨模块的顺序问题在这里解决
├── pipeline/rag.py    ★ 一次问答的完整流程（建议从这里开始读）
├── ingestion/         离线链路：文件 → 文本 → chunk
│   ├── loader.py      按扩展名分发
│   ├── parsers/       pdf(MinerU/pypdf) · docx · markdown
│   ├── cleaner.py     清洗（必须在切分之前，否则偏移全废）
│   ├── splitter.py    结构感知切分 + 字符偏移（纯标准库，可单测）
│   └── pipeline.py    串起来，写 Milvus + 关系库
├── indexing/          写入与召回
│   ├── milvus_store.py  ★ collection schema / upsert / hybrid_search
│   ├── embedding.py     DashScope 向量化
│   └── expression.py    过滤表达式（纯函数，可单测）
├── retrieval/         在线检索
│   ├── hybrid.py       双路召回 + RRF + 门面
│   ├── rerank.py       Cross-Encoder 精排（失败自动降级）
│   └── rewrite.py      多轮指代消解 / 检索失败后的扩写
├── generation/        生成
│   ├── prompt.py       全项目唯一的 prompt 来源
│   ├── llm.py          模型客户端（LangChain 1.x）
│   └── citation.py     [n] → 引用结构（纯函数，可单测）
├── models/            orm.py（表） + schemas.py（接口模型）
└── core/              config / db / milvus / logging

web/                   单页前端（原生 JS，无构建）
scripts/               初始化与自检脚本
deploy/                Milvus standalone 的 compose（换机器时用）
tests/                 单元测试
```

**推荐阅读顺序**：`app/pipeline/rag.py` → `app/indexing/milvus_store.py` → `app/ingestion/splitter.py`
→ `app/retrieval/hybrid.py` → `app/generation/prompt.py`。看完这五个就掌握了全貌。

---

## 三、一次问答发生了什么

先记住两段式——这就是所有 RAG 流程图讲的那两段：

**离线阶段（上传文档时跑一次）**

```
保险条款.pdf → 解析 → 清洗 → 切分(900字/块，记下页码和字符偏移) → 向量化 → 存进 Milvus
```

**在线阶段（每次提问时）**

```
你的问题
   ├─ 改写      有历史对话才做，把「它」换成具体名词
   ├─ 向量化    和离线用的是同一个模型（这点错了会"检索得到但全不相关"）
   ├─ 检索      向量 + 关键词两路各捞 10 条，合并
   ├─ 精排      逐条判断"能不能回答这个问题"，留 5 条
   ├─ 拼 prompt 5 段资料 + 「只能照资料说」的规矩 + 你的问题
   ├─ 生成      大模型流式输出，一个字一个字往外吐
   └─ 引用      把答案里的 [1] 翻译成"哪份文件第几页第几个字"
```

用保险那个例子：问「尊享e生2026版免赔额是多少？」→ 检索到"年度免赔额 1 万元"那一段（第 7 页）
→ 拼进 prompt → 模型答「免赔额是 1 万元 [1]」→ 点 [1] 弹出第 7 页原文。

图上没有、但代码里多出来的四步（每一步都在补一个真实漏洞）：

| 多出来的 | 不做会怎样 |
|---|---|
| 关键词检索（BM25） | 问「QKV」「2026版」这类精确词会漏 |
| 精排 | 向量检索只管"意思像"，前 5 条里混进无关段落 |
| 改写 | 第二句问「它呢？」，这句话没有语义，检索必然失败 |
| 引用解析 | 用户没法验证真假，也跳不回原文 |

细节和代码对应关系见 [docs/代码导读.md](docs/代码导读.md)。

---

## 四、几个关键设计决定

| 决定 | 为什么 |
|---|---|
| **chunk 在关系库和 Milvus 各存一份** | Milvus 是「可重建的派生索引」，关系库才是主数据。这样换库/重建索引不会丢内容 |
| **chunk_id = `{doc_id}_{seq:05d}`** | 确定性 ID，重复索引 = 覆盖写，天然幂等 |
| **dense 客户端算、sparse 交给 Milvus BM25 Function** | 文档和查询走同一套 embedding 代码，避免「两个模型」这种最难查的错；分词交给 Milvus |
| **索引用函数编排，检索用显式步骤** | 索引是线性 DAG，检索才有分支。没有分支的地方硬套 LangGraph 只会更难读 |
| **删除顺序：先向量后记录** | 反过来一旦 Milvus 删除失败，就产生永远找不到的「幽灵 chunk」 |
| **SSE 而不是 WebSocket** | 问答是单向流；SSE 不需要心跳和重连逻辑 |

---

## 五、API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（含 Milvus / collection 状态） |
| POST | `/api/v1/kb` | 新建知识库 |
| GET | `/api/v1/kb` | 知识库列表 |
| DELETE | `/api/v1/kb/{kb_id}` | 删库（连带向量与文件） |
| POST | `/api/v1/kb/{kb_id}/documents` | 上传（返回 job_id，后台索引） |
| GET | `/api/v1/kb/{kb_id}/documents` | 文档列表 |
| DELETE | `/api/v1/kb/documents/{doc_id}` | 删文档 |
| GET | `/api/v1/jobs/{job_id}` | 索引进度 |
| POST | `/api/v1/chat` | 提问（SSE 流） |
| GET | `/api/v1/chat/{conversation_id}/messages` | 会话历史 |
| GET | `/api/v1/chunks/{chunk_id}` | 取 chunk 原文 |

SSE 事件类型：`meta`（含 conversation_id）、`token`、`citations`、`warning`、`done`、`error`。

---

## 六、排错

| 现象 | 原因 |
|---|---|
| 启动日志 `collection ... 不存在` | 没跑 `scripts/init_milvus.py` |
| 上传后进度卡在 5% 然后失败 | 看 `/api/v1/jobs/{id}` 的 `error`；多半是解析不出文本（扫描件） |
| `缺少 DASHSCOPE_API_KEY` | `.env` 没填或没重启进程 |
| 检索总是召回 0 条 | `EMBED_DIM` 和建 collection 时不一致；改完要 `init_milvus.py --drop` 重建 |
| 答案里没有引用角标 | 模型没按格式标；检查 `app/generation/prompt.py` 的规则，或换更强的模型 |
| 前端一直转圈 | 代理缓冲了 SSE；本项目已设 `X-Accel-Buffering: no`，自己加 Nginx 时注意别覆盖 |
| 总在「重查」或总在拒答 | 两个阈值没校准，跑 `scripts/calibrate.py` 看真实分数分布 |

---

## 七、参数是怎么定的（别拍脑袋）

`SCORE_THRESHOLD` 和 `MIN_RERANK_SCORE` 这两个数不是随手写的，是用
`scripts/calibrate.py` 在真实数据上量出来的：

```
同领域问题最高分   0.2992 ~ 0.6082
无关问题最高分     0.0060 ~ 0.0061
```

两类之间有巨大的空白带，所以：

| 参数 | 值 | 作用 |
|---|---|---|
| `MIN_RERANK_SCORE` | 0.02 | 单条候选的入场券，低于它不进 prompt（把无关问题噪成空结果 → 走拒答） |
| `SCORE_THRESHOLD` | 0.15 | 「这次检索够不够好」，低于它才多花一次 LLM 调用去扩写重查 |
| `TOP_K_FINAL` | 5 | 上限，实际条数由 `MIN_RERANK_SCORE` 卡 |

实测效果（同一个问题）：

- 阈值 0.35 时：相关度 0.2992 的问题被误判成「检索失败」，白跑一次改写 + 重查
- 阈值 0.15 时：`retries=0`，且 5 条候选里只有 2 条过线，prompt 从「整篇文档」缩到「两段相关文字」

**换 embedding / rerank 模型后一定要重跑一次 calibrate.py**，分数分布会变。

---

## 八、路线图

- **M2**：Celery 替掉 BackgroundTasks（进程重启不丢任务）、任务并发限流、增量更新（文档改名/改内容）、前后文扩展（用 `seq`）
- **M3**：把 `app/pipeline/rag.py` 搬进 LangGraph（加上 grade 回退、LLM 校验、人工审核 middleware）、RAGAS 评测集与回归
- **M4**：按用户隔离、引用定位到 PDF 高亮、多模态（图片/表格理解）

## 九、已知限制

- 解析层没做页眉页脚识别，长 PDF 会有重复文本（靠 chunk 内容去重可以缓解，尚未实现）
- 检索只做了单轮，没有「子问题拆解」
- 没有鉴权：接口裸奔，只适合内网/本机
- 会话历史只按条数截断，没有做摘要压缩
- 「删文档」和「正在索引」同时发生时存在竞态：极端情况下可能留下孤儿向量
  （正常操作不会触发；要彻底解决需要加软删除标记，属于 M2）
