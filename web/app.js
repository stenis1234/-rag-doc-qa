/* 前端逻辑：没有框架、没有构建步骤，一个文件搞定。
 *
 * 三个功能：
 *   1. 左侧：知识库选择 / 新建、上传文档（轮询索引进度）、文档列表
 *   2. 右侧：提问，解析 SSE 流式答案
 *   3. 引用：点击角标或用例卡片，弹窗显示原文片段
 */

const $ = (sel) => document.querySelector(sel);

const state = {
  kbId: null,
  conversationId: null,
  busy: false,
};

/* ------------------------------------------------------------------ 工具 */

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    // FastAPI 的错误体是 {"detail": "..."}，直接展示出来比 "HTTP 400" 有用得多
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* 忽略解析失败 */ }
    throw new Error(detail);
  }
  return resp;
}

/* ------------------------------------------------------------------ 知识库 */

async function loadKbs() {
  const resp = await api("/api/v1/kb");
  const kbs = await resp.json();
  const select = $("#kb-select");
  select.innerHTML = "";

  if (kbs.length === 0) {
    select.innerHTML = '<option value="">（还没有知识库）</option>';
    state.kbId = null;
    renderDocs([]);
    return;
  }

  for (const kb of kbs) {
    const opt = document.createElement("option");
    opt.value = kb.id;
    opt.textContent = kb.name;
    select.appendChild(opt);
  }

  // 尽量保持当前选中的库不变
  const keep = kbs.some((k) => String(k.id) === String(state.kbId)) ? state.kbId : kbs[0].id;
  state.kbId = Number(keep);
  select.value = String(keep);
  persist();
  await loadDocs();
}

/* ---------- 会话状态持久化（刷新页面后能恢复） ---------- */

const STORAGE_KEY = "rag-chat-state";

function persist() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ kbId: state.kbId, conversationId: state.conversationId })
    );
  } catch (_) {
    /* 隐私模式下可能禁用 localStorage，忽略即可 */
  }
}

function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (saved.kbId) state.kbId = Number(saved.kbId);
    if (saved.conversationId) state.conversationId = Number(saved.conversationId);
  } catch (_) {
    /* 存的内容坏了就当没有 */
  }
}

// 拉历史消息并渲染。会话不存在（被删了）就静默开新会话。
async function loadHistory() {
  if (!state.conversationId) return;
  try {
    const resp = await api(`/api/v1/chat/${state.conversationId}/messages`);
    const rows = await resp.json();
    if (!rows.length) return;

    $("#messages").innerHTML = "";
    for (const msg of rows) {
      const bubble = appendMessage(msg.role, msg.content);
      if (msg.role === "assistant") {
        renderAnswer(bubble, msg.content, msg.citations || []);
      }
    }
  } catch (err) {
    console.warn("加载历史失败", err);
    state.conversationId = null;
    persist();
  }
}

function resetChat() {
  state.conversationId = null;
  persist();
  $("#messages").innerHTML =
    '<div class="empty" id="empty-hint"><h3>新会话</h3><p>直接提问即可。</p></div>';
}

$("#kb-select").addEventListener("change", async (e) => {
  state.kbId = Number(e.target.value) || null;
  state.conversationId = null; // 换库就开新会话，避免上下文串库
  persist();
  resetChat();
  await loadDocs();
  await loadConversations();
});

$("#new-chat").addEventListener("click", resetChat);

$("#kb-create").addEventListener("click", async () => {
  const input = $("#kb-name");
  const name = input.value.trim();
  if (!name) return;
  try {
    await api("/api/v1/kb", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description: "" }),
    });
    input.value = "";
    await loadKbs();
  } catch (err) {
    alert("新建失败：" + err.message);
  }
});

$("#kb-refresh").addEventListener("click", () => loadKbs().catch((e) => alert(e.message)));

/* ------------------------------------------------------------------ 文档 */

function renderDocs(docs) {
  const list = $("#doc-list");
  list.innerHTML = "";
  $("#doc-count").textContent = docs.length ? `(${docs.length})` : "";

  for (const doc of docs) {
    const li = document.createElement("li");

    const name = document.createElement("span");
    name.className = "name clickable";
    name.textContent = doc.filename;
    name.title = "点击查看解析结果";  // 鼠标悬停提示，告诉用户"能点"
    name.addEventListener("click", () => openDoc(doc.id));
    li.appendChild(name);

    const tag = document.createElement("span");
    tag.className = "tag " + doc.status;
    tag.textContent = {
      indexed: "已索引",
      failed: "失败",
      pending: "等待",
      parsing: "解析中",
      indexing: "索引中",
    }[doc.status] || doc.status;
    li.appendChild(tag);

    const del = document.createElement("button");
    del.className = "icon";
    del.textContent = "✕";
    del.title = "删除";
    del.addEventListener("click", async () => {
      if (!confirm(`删除「${doc.filename}」？相关向量也会一起删掉。`)) return;
      try {
        await api(`/api/v1/kb/documents/${doc.id}`, { method: "DELETE" });
        await loadDocs();
      } catch (err) {
        alert("删除失败：" + err.message);
      }
    });
    li.appendChild(del);

    list.appendChild(li);
  }
}

async function loadDocs() {
  if (!state.kbId) {
    renderDocs([]);
    return;
  }
  const resp = await api(`/api/v1/kb/${state.kbId}/documents`);
  renderDocs(await resp.json());
}

/* ---------- 历史会话 ---------- */

async function loadConversations() {
  if (!state.kbId) {
    renderConversations([]);
    return;
  }
  try {
    const resp = await api(`/api/v1/chat/conversations?kb_id=${state.kbId}`);
    renderConversations(await resp.json());
  } catch (err) {
    console.warn("加载会话列表失败", err);
  }
}

function renderConversations(list) {
  const ul = $("#conv-list");
  ul.innerHTML = "";
  $("#conv-count").textContent = list.length ? `(${list.length})` : "";

  for (const conv of list) {
    const li = document.createElement("li");
    const active = String(conv.id) === String(state.conversationId);
    li.className = "conv-item" + (active ? " active" : "");

    const name = document.createElement("span");
    name.className = "name clickable";
    name.textContent = conv.title || `会话 #${conv.id}`;
    name.title = `${conv.message_count} 条消息`;
    name.addEventListener("click", () => openConversation(conv.id));
    li.appendChild(name);

    const count = document.createElement("span");
    count.className = "tl";
    count.textContent = `${conv.message_count}条`;
    li.appendChild(count);

    const del = document.createElement("button");
    del.className = "icon";
    del.textContent = "✕";
    del.title = "删除会话";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`删除会话「${conv.title}」？`)) return;
      try {
        await api(`/api/v1/chat/conversations/${conv.id}`, { method: "DELETE" });
        if (String(state.conversationId) === String(conv.id)) resetChat();
        await loadConversations();
      } catch (err) {
        alert("删除失败：" + err.message);
      }
    });
    li.appendChild(del);

    ul.appendChild(li);
  }
}

async function openConversation(id) {
  state.conversationId = id;
  persist();
  $("#messages").innerHTML = "";
  await loadHistory();
  await loadConversations(); // 刷新高亮
}

/* ---------- 上传 + 进度轮询 ---------- */
//
// 三种入口最后都汇到 uploadFile()：
//   1. 点击虚线框选文件
//   2. 把文件拖进虚线框
//   3. 直接 Ctrl+V 粘贴（文字会自动存成 .md 再上传）

// 后端真正能解析的格式（和 app/ingestion/loader.py 的 PARSERS 保持一致）。
// 注意这里不再用 <input accept> 过滤，而是在选中后校验——
// 否则文件对话框会把不匹配的文件藏起来，用户只会看到"无与搜索匹配的项"。
// 能上传的格式（和 app/ingestion/loader.py 的 supported_extensions() 保持一致）。
// 其中 ppt/xlsx/doc/图片 依赖 MINERU_TOKEN，没配 token 时后端会返回 400。
const ACCEPTED = [
  ".pdf", ".docx", ".doc", ".md", ".markdown", ".txt",
  ".pptx", ".ppt", ".xlsx", ".xls",
  ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff",
  ".html", ".htm",
];

function fileExt(name) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

async function uploadFile(file) {
  if (!state.kbId) {
    alert("请先选择或新建一个知识库");
    return;
  }
  if (!file) return;

  const ext = fileExt(file.name);
  if (!ACCEPTED.includes(ext)) {
    alert(
      `不支持「${ext || "无扩展名"}」格式。\n\n` +
        `目前支持：${ACCEPTED.join("、")}\n\n` +
        `如果是 .doc / .xlsx / .pptx，需要先配置 MINERU_TOKEN 才能解析。\n` +
        `如果是图片（.png / .jpg），同样需要 MINERU_TOKEN 走 OCR。`
    );
    return;
  }

  const form = new FormData();
  form.append("file", file, file.name);

  const progress = $("#job-progress");
  progress.classList.remove("hidden");
  $("#job-bar").style.width = "0%";
  $("#job-stage").textContent = `上传中…（${file.name}）`;

  try {
    const resp = await api(`/api/v1/kb/${state.kbId}/documents`, {
      method: "POST",
      body: form,
    });
    const result = await resp.json();
    await loadDocs();
    await pollJob(result.job_id);
  } catch (err) {
    $("#job-stage").textContent = "上传失败：" + err.message;
  }
}

$("#file-input").addEventListener("change", async (e) => {
  const files = Array.from(e.target.files || []);
  e.target.value = ""; // 清空，允许连续选同一批文件
  // 逐个上传：一个传完再传下一个，避免多个进度条互相覆盖
  for (const file of files) {
    await uploadFile(file);
  }
});

/* ---------- 拖拽上传 ---------- */

const dropzone = $("#dropzone");

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault(); // 不阻止的话浏览器会直接打开文件
    dropzone.classList.add("dragging");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragging");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  const files = Array.from(e.dataTransfer?.files || []);
  (async () => {
    for (const f of files) await uploadFile(f);
  })();
});

/* ---------- 粘贴上传 ----------
 *
 * 场景：从网页/Word 里复制一段文字，直接 Ctrl+V 就存成文档，
 * 省掉"新建 txt → 粘贴 → 保存 → 选文件"这一串动作。
 *
 * 注意两件事：
 *   1. 如果焦点在输入框里（用户想往问题框里粘文字），必须放过，不能劫持
 *   2. 粘贴板里可能有文件（截图），也可能只有纯文本
 */
window.addEventListener("paste", (e) => {
  const tag = e.target?.tagName;
  if (tag === "TEXTAREA" || tag === "INPUT") return; // 正常粘贴文字，不拦截

  const files = Array.from(e.clipboardData?.files || []);
  if (files.length) {
    const file = files[0];
    if (file.type.startsWith("image/")) {
      alert(
        "暂不支持直接粘贴图片。\n\n" +
          "图片里没有可检索的文字，需要 OCR。\n" +
          "你可以：把图片插入 Word 另存为 PDF 上传，或直接用文字复制。"
      );
      return;
    }
    uploadFile(file);
    return;
  }

  const text = e.clipboardData?.getData("text/plain");
  if (text && text.trim()) {
    // 用一个带时间戳的文件名，避免两次粘贴互相覆盖
    const stamp = new Date()
      .toISOString()
      .slice(0, 19)
      .replace(/[:T]/g, "-");
    const file = new File([text], `粘贴内容-${stamp}.md`, { type: "text/markdown" });
    uploadFile(file);
  }
});

async function pollJob(jobId) {
  for (let i = 0; i < 600; i++) {
    const resp = await api(`/api/v1/jobs/${jobId}`);
    const job = await resp.json();
    $("#job-bar").style.width = `${job.progress}%`;
    $("#job-stage").textContent = `${job.stage}（${job.progress}%）`;

    if (job.status === "done" || job.status === "failed") {
      if (job.status === "failed") {
        $("#job-stage").textContent = "索引失败：" + (job.error || "未知原因");
      }
      await loadDocs();
      setTimeout(() => $("#job-progress").classList.add("hidden"), 2500);
      return;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

/* ------------------------------------------------------------------ 对话 */

function appendMessage(role, text) {
  $("#empty-hint")?.remove();

  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "我" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrap.appendChild(who);
  wrap.appendChild(bubble);
  $("#messages").appendChild(wrap);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return bubble;
}

/** 把答案里的 [1][2] 变成可点击的角标。 */
function renderAnswer(bubble, text, citations) {
  const html = escapeHtml(text)
    // 模型爱用 markdown 的 **粗体**，不转换的话会原样显示星号
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(
      /\[(\d{1,2})\]/g,
      (match, idx) => `<span class="cu" data-idx="${idx}">[${idx}]</span>`
    );
  bubble.innerHTML = html;

  if (citations?.length) {
    const box = document.createElement("div");
    box.className = "citations";
    for (const c of citations) {
      const chip = document.createElement("button");
      chip.className = "cite-chip";
      chip.innerHTML = `<b>[${c.index}]</b>${escapeHtml(c.doc_name || "文档")}${
        c.page_no ? " · 第" + c.page_no + "页" : ""
      }`;
      chip.title = c.title || "";
      // 把关键词挂在卡片上，点开原文时用来高亮
      chip.dataset.keywords = JSON.stringify(c.keywords || []);
      chip.addEventListener("click", () =>
        openChunk(c.chunk_id, JSON.parse(chip.dataset.keywords || "[]"))
      );
      box.appendChild(chip);
    }
    bubble.appendChild(box);
  }
}

const dialog = $("#chunk-dialog");
$("#dialog-close").addEventListener("click", () => dialog.close());

/* ---------- 文档预览 ----------
 *
 * 为什么要有这个功能：
 *   上传完一个 PDF，如果检索效果不好，第一个要回答的问题是
 *   「系统到底从文件里读到了什么」。
 *   看原始 PDF 是看不出来的（原文件永远是对的），
 *   只有看「解析后的文本」才知道是解析出乱码了、还是表格没读出来、还是切分切坏了。
 *
 * 数据来源：GET /api/v1/documents/{id}
 *   pages 是从 chunk 按页拼回来的，不是原始文件字节。
 */

const docDialog = $("#doc-dialog");
$("#doc-close").addEventListener("click", () => docDialog.close());

async function openDoc(docId) {
  let doc;
  try {
    // 注意路径前缀是 /kb/documents/...，和删除接口保持一致
    const resp = await api(`/api/v1/kb/documents/${docId}`);
    doc = await resp.json();
  } catch (err) {
    alert("打开失败：" + err.message);
    return;
  }

  const statusText =
    { indexed: "已索引", failed: "索引失败", pending: "等待中", parsing: "解析中", indexing: "索引中" }[
      doc.status
    ] || doc.status;

  $("#doc-title").textContent = doc.filename;
  $("#doc-meta").textContent = [
    statusText,
    doc.page_count ? `共 ${doc.page_count} 页` : "",
    `切成 ${doc.chunk_count} 块`,
    doc.size_bytes ? `${(doc.size_bytes / 1024).toFixed(1)} KB` : "",
  ]
    .filter(Boolean)
    .join(" · ");

  // 「打开原文件」用新标签页直接怼后端接口，浏览器会内联预览 PDF / 显示文本
  $("#doc-open-file").onclick = () =>
    window.open(`/api/v1/kb/documents/${docId}/file`, "_blank");

  const errorBox = $("#doc-error");
  if (doc.error) {
    errorBox.textContent = "索引失败：" + doc.error;
    errorBox.classList.remove("hidden");
  } else {
    errorBox.classList.add("hidden");
  }

  const body = $("#doc-body");
  body.innerHTML = "";

  if (!doc.pages.length) {
    const tip = document.createElement("p");
    tip.className = "muted";
    tip.textContent =
      doc.status === "indexed"
        ? "没有可显示的文本（可能是空文档）"
        : "文档还没解析完成，稍后再点一次";
    body.appendChild(tip);
  } else {
    for (const page of doc.pages) {
      const section = document.createElement("section");
      section.className = "doc-page";
      // page_no=0 表示没有页码概念（md/docx），此时不显示"第 0 页"
      if (page.page_no) {
        const title = document.createElement("h4");
        title.textContent = `第 ${page.page_no} 页`;
        section.appendChild(title);
      }
      const pre = document.createElement("pre");
      pre.textContent = page.text;
      section.appendChild(pre);
      body.appendChild(section);
    }
  }

  docDialog.showModal();
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** 把原文里的关键词包成 <mark>，让用户一眼看到「哪几个词是我问的」。 */
function highlightText(text, keywords) {
  let html = escapeHtml(text || "");
  const terms = (keywords || [])
    .filter((k) => k && k.length >= 2)
    .sort((a, b) => b.length - a.length);

  for (const term of terms) {
    const pattern = new RegExp(escapeRegExp(escapeHtml(term)), "g");
    // 已经包过 <mark> 的跳过，避免嵌套
    html = html.replace(pattern, (m) => `<mark>${m}</mark>`);
  }
  return html;
}

async function openChunk(chunkId, keywords = []) {
  try {
    const resp = await api(`/api/v1/chunks/${encodeURIComponent(chunkId)}`);
    const chunk = await resp.json();
    $("#chunk-title").textContent = chunk.title || chunk.doc_name || "原文片段";
    $("#chunk-meta").textContent = [
      chunk.doc_name,
      chunk.page_no ? `第 ${chunk.page_no} 页` : "",
      `字符 ${chunk.char_start}–${chunk.char_end}`,
    ]
      .filter(Boolean)
      .join(" · ");
    // 用 innerHTML + <mark> 高亮，文本已做过 HTML 转义
    $("#chunk-content").innerHTML = highlightText(chunk.content, keywords);
    dialog.showModal();
  } catch (err) {
    alert("取原文失败：" + err.message);
  }
}

/** 点击答案里的角标 → 找到对应 citation → 打开原文 */
$("#messages").addEventListener("click", (e) => {
  const el = e.target.closest(".cu");
  if (!el) return;
  const idx = el.dataset.idx;
  const chips = el.closest(".bubble")?.querySelectorAll(".cite-chip");
  // 角标和下方卡片是一一对应的，按编号找比按顺序找稳
  for (const chip of chips || []) {
    if (chip.textContent.startsWith(`[${idx}]`)) {
      chip.click();
      return;
    }
  }
});

/* ---------- 发送 ---------- */

const composer = $("#composer");
const questionInput = $("#question");

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (state.busy) return;

  const question = questionInput.value.trim();
  if (!question) return;
  if (!state.kbId) {
    alert("请先选择或新建一个知识库");
    return;
  }

  questionInput.value = "";
  questionInput.style.height = "auto";
  appendMessage("user", question);

  const bubble = appendMessage("assistant", "");
  bubble.classList.add("typing");

  state.busy = true;
  $("#send").disabled = true;

  let answer = "";
  let citations = [];

  try {
    await streamChat(question, (event, data) => {
      if (event === "meta" && data.conversation_id) {
        state.conversationId = data.conversation_id;
        persist();
        loadConversations(); // 新会话建好了，刷新左侧列表
      } else if (event === "token") {
        answer += data.text || "";
        // 流式过程中用纯文本，避免半截标签；等结束再渲染角标
        bubble.textContent = answer;
        $("#messages").scrollTop = $("#messages").scrollHeight;
      } else if (event === "citations") {
        citations = data.items || [];
      } else if (event === "warning") {
        const warn = document.createElement("div");
        warn.className = "warn";
        warn.textContent = data.text;
        bubble.after(warn);
      } else if (event === "error") {
        answer += `\n\n[出错了] ${data.message}`;
      }
    });
  } catch (err) {
    answer += `\n\n[出错了] ${err.message}`;
  } finally {
    bubble.classList.remove("typing");
    renderAnswer(bubble, answer, citations);
    state.busy = false;
    $("#send").disabled = false;
    $("#messages").scrollTop = $("#messages").scrollHeight;
  }
});

/**
 * 发请求并逐帧解析 SSE。
 *
 * 为什么不用 EventSource：它只支持 GET，带不了 JSON body。
 * 所以用 fetch 拿到 ReadableStream，自己按 "\n\n" 切帧。
 */
async function streamChat(question, onEvent) {
  const resp = await fetch("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      kb_id: state.kbId,
      question,
      conversation_id: state.conversationId,
    }),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      let event = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;

      try {
        onEvent(event, JSON.parse(dataLines.join("\n")));
      } catch (err) {
        console.error("解析 SSE 失败", frame, err);
      }
    }
  }
}

/* ------------------------------------------------------------------ 启动 */

restore();

loadKbs()
  .then(loadConversations)
  .then(loadHistory)
  .catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div class="warn" style="margin:12px">加载失败：${escapeHtml(err.message)}` +
      `（检查后端是否已启动，以及 /api/health 是否正常）</div>`
  );
});
