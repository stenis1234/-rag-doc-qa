# RAG 文档问答系统

> 把 PDF / Word / Markdown 丢进去，用自然语言提问；答案逐字流式输出，每个结论都能点回原文的「第几页、第几个字」。

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-2.5%2B-00A1EA)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

---

## 一、项目简介

这是一个**可以完整跑起来**的文档问答（RAG）系统：后端 FastAPI，向量库 Milvus，模型编排用 LangChain 1.x，生成用 DeepSeek，向量化与精排用 DashScope。

市面上 RAG 的流程图都长一个样，难的是把图上的每个方块变成能跑的代码，并且知道**为什么这么写**。这个项目的重点不是前端花活，而是：

- **两段式链路完整闭环**：离线建库（解析 → 清洗 → 切分 → 向量化 → 入库）和在线问答（改写 → 双路召回 → 融合 → 精排 → 拼 prompt → 流式生成 → 引用回溯）。
- **每一步都有存在理由**：混合检索、精排、问题改写、引用解析、检索失败重查——每加一步，都是因为它补上了一个真实漏洞（见[第五节](#五一次问答到底发生了什么)）。
- **参数是量出来的，不是拍脑袋**：`SCORE_THRESHOLD`、`MIN_RERANK_SCORE` 由 `scripts/calibrate.py` 在真实文档上实测分数分布后确定（见[第八节](#八参数是怎么定的别拍脑袋)）。
- **工程细节写进了注释**：为什么先删向量再删记录、为什么 dense 客户端算而 sparse 交给 Milvus、为什么用 SSE 不用 WebSocket——代码里都有答案。

**适合谁看**：想搞懂 RAG 内部到底发生了什么、并且想照着搭一套自己知识库的开发者。代码按「一层只做一件事」组织，配合 [docs/代码导读.md](docs/代码导读.md) 可以从一次问答的入口一路读到底层。

**当前状态**：核心链路已可用（多知识库、混合检索、重排、流式、引用回溯、会话历史）。异步任务、自纠错、评测体系在路线图里，见[第十二节](#十二路线图)。

---

## 二、功能特性

**文档接入**

- 支持 `md` / `markdown` / `txt` / `pdf` / `docx`；配了 MinerU 之后还能吃 `doc` / `ppt(x)` / `xls(x)` / 图片 / `html`
- PDF 双通道：配 `MINERU_TOKEN` 走 MinerU（版面还原 + OCR，扫描件也能读），否则回退 pypdf（页码更准）
- 清洗在切分之前做——顺序反了，字符偏移和页码就全废了
- 结构感知切分：按中文标点递归切，900 字/块、相邻重叠 100 字，每块记录**页码 + 字符偏移**
- 上传即返回 `job_id`，前端轮询进度条；解析失败的原因写进 job 记录，不用翻日志

**检索**

- 稠密 + 稀疏双路召回：向量路 8 条，BM25 关键词路 8 条（走 Milvus 内置 BM25 Function，分词交给它）
- RRF（Reciprocal Rank Fusion，k=60）融合两路，只按名次算分，不用调权重
- `gte-rerank-v2` Cross-Encoder 精排，最终 4 条进 prompt；精排调用失败自动降级为 RRF 分数，不会整条链路挂掉
- 分数统一归一化到 0~1，让「关掉精排」和「精排成功」两种情况的阈值都说得通
- 检索偏弱时自动扩写查询重查一次（有次数上限，避免无限烧 token）

**生成与引用**

- DeepSeek 流式输出，SSE 推送 `token` 事件，首字延迟低
- prompt 全项目只有一处（`app/generation/prompt.py`）：4 段资料 + 「只能照资料说」的规矩 + 用户问题
- 引用解析：把答案里的 `[1]` 翻译成「哪份文件、第几页、第几个字」，并提取与回答重合的关键词供前端高亮
- 兜底逻辑：一条都没召回就直接拒答，不把空结果丢给模型去编；答案没有引用角标时返回 `warning`
- 多轮对话：取最近 3 轮做指代消解（「它」→ 具体名词），更早的历史压成摘要，控制上下文成本

**工程化**

- 分层的目录结构：`api`（校验 → 调 service → 返回）/ `services`（跨模块编排）/ `pipeline`（一次问答的流程）/ 各领域模块
- 纯函数能单测的都拆出来了（切分、过滤表达式已有测试，引用解析同样是纯函数），`pytest` 直接跑，不需要 Milvus 和大模型
- 两个自检脚本：`smoke_test.py` 只测 Milvus 层，`dev_check.py` 跑整条链路但用假向量——**不花钱**就能挡掉 90% 的环境问题
- SQLite 零配置起步，改一行 `DATABASE_URL` 就能切 PostgreSQL；Milvus 用 `deploy/docker-compose.yml` 一键起

---

## 三、技术栈

| 层次 | 选型 | 说明 |
|---|---|---|
| Web | FastAPI + Uvicorn | 异步、自动生成 OpenAPI 文档 |
| 前端 | 原生 HTML/CSS/JS | 单页、无构建步骤，改完刷新即生效 |
| 业务库 | SQLAlchemy 2.x + SQLite（可换 PostgreSQL） | 存知识库、文档、chunk 主数据、会话、任务 |
| 向量库 | Milvus 2.5 | 需要 2.5+ 才有 BM25 Function 与 `hybrid_search` |
| 向量化 | DashScope `text-embedding-v3`（1024 维） | 离线与在线共用同一个模型 |
| 精排 | DashScope `gte-rerank-v2` | Cross-Encoder，失败自动降级 |
| 生成 | DeepSeek（`deepseek-flash`）+ LangChain 1.x | 流式输出，温度 0.2，默认关闭思考省 token |
| 文档解析 | MinerU / pypdf / python-docx | 按格式与是否配 token 自动选择 |
| 切分 | LangChain `RecursiveCharacterTextSplitter` | 按中文标点递归，保留偏移 |

---

## 四、系统架构

```
浏览器  web/  单页前端（原生 JS，无构建）
   │  HTTP / SSE
   ▼
FastAPI  app/main.py
   ├─ api/v1/      kb · documents · jobs · chat · chunks
   ├─ services/    业务编排（上传、删除、问答、会话）
   └─ pipeline/    rag.py：一次问答的完整流程
   │
   ├── 离线：上传文档时
   │     ingestion/   解析 → 清洗 → 切分（带页码与字符偏移）
   │     indexing/    embedding（1024 维）→ Milvus 写入
   │     SQLite/PG    文档 · chunk 原文 · 任务状态（主数据）
   │
   └── 在线：每次提问
         retrieval/   rewrite 改写 → hybrid 双路召回 + RRF → rerank 精排
         generation/  prompt 拼装 → llm 流式生成 → citation 引用解析
         前端         答案逐字输出，[n] 角标点回原文并高亮
```

关键分工：**关系库是主数据，Milvus 是可重建的派生索引**。chunk 原文两份都存，向量库删了可以重灌，内容不会丢。

---

## 五、一次问答到底发生了什么

先记住两段式——所有 RAG 流程图讲的就是这两段。

**离线阶段（上传文档时跑一次）**

```
文档 → 解析 → 清洗 → 切分（900 字/块，记下页码和字符偏移） → 向量化 → 写入 Milvus
```

**在线阶段（每次提问时）**

```
你的问题
   ├─ 改写      有历史对话才做，把「它」换成具体名词
   ├─ 向量化    和离线用的是同一个模型（这点错了会「检索得到但全不相关」）
   ├─ 检索      向量路 8 条 + 关键词路 8 条 → RRF 融合成 10 条候选
   ├─ 精排      逐条判断「能不能回答这个问题」，留 4 条
   ├─ 拼 prompt 4 段资料 +「只能照资料说」的规矩 + 你的问题
   ├─ 生成      大模型流式输出，一个字一个字往外吐
   └─ 引用      把答案里的 [1] 翻译成「哪份文件第几页第几个字」
```

用保险条款举例：问「尊享 e 生 2026 版免赔额是多少？」→ 检索到「年度免赔额 1 万元」那一段（第 7 页）→ 拼进 prompt → 模型答「免赔额是 1 万元 [1]」→ 点 `[1]` 弹出第 7 页原文并高亮关键词。

图上没有、但代码里多出来的四步，每一步都在补一个真实漏洞：

| 多出来的 | 不做会怎样 |
|---|---|
| 关键词检索（BM25） | 问「QKV」「2026 版」这类精确词会漏 |
| 精排 | 向量检索只管「意思像」，候选里混进无关段落 |
| 改写 | 第二句问「它呢？」，这句话没有语义，检索必然失败 |
| 引用解析 | 用户没法验证真假，也跳不回原文 |

细节和代码对应关系见 [docs/代码导读.md](docs/代码导读.md)。

---

## 六、快速开始

### 1. 准备环境

- Python 3.11+
- 一个跑着的 Milvus（默认 `http://127.0.0.1:19530`）。没有的话用仓库里的 compose 起一个：

```bash
docker compose -f deploy/docker-compose.yml up -d
```

```bash
# 依赖（已装好可跳过；用 uv，没有 uv 也可以用 pip）
uv pip install -r pyproject.toml
```

### 2. 配置

```bash
copy env.example .env      # Windows
cp env.example .env        # macOS / Linux
```

编辑 `.env`，至少填两个 key：

```ini
DASHSCOPE_API_KEY=sk-...   # embedding + rerank
DEEPSEEK_API_KEY=sk-...    # 生成答案
```

模型默认是 `text-embedding-v3` / `gte-rerank-v2` / `deepseek-flash`，按自己环境可用的改。走自建网关时填 `LLM_BASE_URL`（OpenAI 兼容协议）。

> `EMBED_DIM` 必须和建 collection 时的维度一致。改了它要重建 collection，否则插入会报 dimension mismatch。

### 3. 建 collection 和表

```bash
python scripts/init_milvus.py
python scripts/init_db.py        # 用 .venv 时：.venv\Scripts\python.exe scripts\init_db.py
```

### 4. 先自检（不花钱）

```bash
python scripts/smoke_test.py     # 只测 Milvus 层
python scripts/dev_check.py      # 整条链路，embedding 用假向量
```

两个都输出 `[OK]` 再往下走。这一步能把 90% 的环境问题挡在模型调用之前。

### 5. 启动

```bash
python -m uvicorn app.main:app --reload --port 8000
```

打开 <http://127.0.0.1:8000>：

1. 左侧新建知识库 → 上传一份自己的 PDF / Word / Markdown → 等进度条走完
2. 提问文档里的内容
3. 点答案里的 `[1]` 角标，看原文片段与高亮

接口文档在 <http://127.0.0.1:8000/docs>，健康检查在 <http://127.0.0.1:8000/api/health>。

---

## 七、目录结构

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
scripts/               初始化、自检、阈值校准
deploy/                Milvus standalone 的 compose（换机器时用）
tests/                 单元测试
docs/                  代码导读
```

**推荐阅读顺序**：`app/pipeline/rag.py` → `app/indexing/milvus_store.py` → `app/ingestion/splitter.py` → `app/retrieval/hybrid.py` → `app/generation/prompt.py`。看完这五个就掌握了全貌。

---

## 八、参数是怎么定的（别拍脑袋）

`SCORE_THRESHOLD` 和 `MIN_RERANK_SCORE` 这两个数不是随手写的，是用 `scripts/calibrate.py` 在真实数据上量出来的：

```
同领域问题最高分   0.2992 ~ 0.6082
无关问题最高分     0.0060 ~ 0.0061
```

两类之间有巨大的空白带，所以：

| 参数 | 值 | 作用 |
|---|---|---|
| `MIN_RERANK_SCORE` | 0.02 | 单条候选的入场券，低于它不进 prompt（把无关问题噪成空结果 → 走拒答） |
| `SCORE_THRESHOLD` | 0.15 | 「这次检索够不够好」，低于它才多花一次 LLM 调用去扩写重查 |
| `TOP_K_FINAL` | 4 | 上限，实际条数由 `MIN_RERANK_SCORE` 卡 |
| `MAX_RETRY` | 1 | 最多重查一次，每次都是一轮 LLM + 一次检索 |

实测效果（同一个问题）：

- 阈值 0.35 时：相关度 0.2992 的问题被误判成「检索失败」，白跑一次改写 + 重查
- 阈值 0.15 时：`retries=0`，候选里只有 2 条过线，prompt 从「整篇文档」缩到「两段相关文字」

**换 embedding / rerank 模型后一定要重跑一次 `calibrate.py`**，分数分布会变。

---

## 九、API

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

## 十、几个关键设计决定

| 决定 | 为什么 |
|---|---|
| **chunk 在关系库和 Milvus 各存一份** | Milvus 是「可重建的派生索引」，关系库才是主数据。这样换库/重建索引不会丢内容 |
| **chunk_id = `{doc_id}_{seq:05d}`** | 确定性 ID，重复索引 = 覆盖写，天然幂等 |
| **dense 客户端算、sparse 交给 Milvus BM25 Function** | 文档和查询走同一套 embedding 代码，避免「两个模型」这种最难查的错；分词交给 Milvus |
| **索引用函数编排，检索用显式步骤** | 索引是线性 DAG，检索才有分支。没有分支的地方硬套 LangGraph 只会更难读 |
| **删除顺序：先向量后记录** | 反过来一旦 Milvus 删除失败，就产生永远找不到的「幽灵 chunk」 |
| **SSE 而不是 WebSocket** | 问答是单向流；SSE 不需要心跳和重连逻辑 |
| **检索失败才重查，且上限 1 次** | 重查是一轮 LLM + 一次检索的成本，收益递减，必须有上限 |

---

## 十一、排错

| 现象 | 原因 |
|---|---|
| 启动日志 `collection ... 不存在` | 没跑 `scripts/init_milvus.py` |
| 上传后进度卡在 5% 然后失败 | 看 `/api/v1/jobs/{id}` 的 `error`；多半是解析不出文本（扫描件且没配 MinerU） |
| `缺少 DASHSCOPE_API_KEY` | `.env` 没填或没重启进程（改 `.env` 不会触发热重载） |
| 检索总是召回 0 条 | `EMBED_DIM` 和建 collection 时不一致；改完要 `init_milvus.py --drop` 重建 |
| 答案里没有引用角标 | 模型没按格式标；检查 `app/generation/prompt.py` 的规则，或换更强的模型 |
| 前端一直转圈 | 代理缓冲了 SSE；本项目已设 `X-Accel-Buffering: no`，自己加 Nginx 时注意别覆盖 |
| 总在「重查」或总在拒答 | 两个阈值没校准，跑 `scripts/calibrate.py` 看真实分数分布 |

---

## 十二、路线图

- **M2**：Celery 替掉 BackgroundTasks（进程重启不丢任务）、任务并发限流、增量更新（文档改名/改内容）、前后文扩展（用 `seq`）
- **M3**：把 `app/pipeline/rag.py` 搬进 LangGraph（加上 grade 回退、LLM 校验、人工审核 middleware）、RAGAS 评测集与回归
- **M4**：按用户隔离、引用定位到 PDF 高亮、多模态（图片/表格理解）

## 十三、已知限制

- 解析层没做页眉页脚识别，长 PDF 会有重复文本（靠 chunk 内容去重可以缓解，尚未实现）
- 检索只做了单轮，没有「子问题拆解」
- 没有鉴权：接口裸奔，只适合内网/本机
- 会话历史：超出 `HISTORY_TURNS` 的部分会被压成摘要（`conversation.summary`），原文细节会丢
- 没有长期记忆：跨会话不共享用户偏好/事实（需要单独做记忆库）
- 「删文档」和「正在索引」同时发生时存在竞态：极端情况下可能留下孤儿向量（正常操作不会触发；要彻底解决需要加软删除标记，属于 M2）
- 索引任务跑在进程内的 `BackgroundTasks` 里，服务重启会中断（M2 换成 Celery）

## 十四、相关文档

- [docs/代码导读.md](docs/代码导读.md)：两条链路的调用图、推荐阅读顺序、术语表，以及「哪个文件读完能回答什么问题」

## 十五、许可证

仓库暂未附带 LICENSE 文件。用于学习交流没有问题；如果打算公开或复用，建议补一个 [MIT](https://choosealicense.com/licenses/mit/) 许可证。
