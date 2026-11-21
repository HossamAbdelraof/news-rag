// News Assistant UI. It only calls the API (/ai/v1/...); no RAG logic lives here.

const API = "/ai/v1";
const $ = (id) => document.getElementById(id);

const state = { threadId: null, threads: [], busy: false };

// [graph step, label, steps it waits for]
const STEPS = [
  ["input_guard", "Checking the question", []],
  ["rewrite", "Understanding", []],
  ["search", "Searching the news", ["rewrite"]],
  ["relevance_gate", "Checking the sources", ["input_guard", "search"]],
  ["generate", "Writing the answer", ["relevance_gate"]],
  ["output_guard", "Verifying", ["generate"]],
];

const NOTICES = {
  blocked: ["warning", "Refused by the input guard"],
  not_found: ["info", "Not found in the sources"],
  unsafe: ["warning", "The answer failed the output checks"],
  error: ["error", "Something went wrong"],
};

const SUGGESTIONS = [
  ["Oil", "Why did OPEC cut oil output?"],
  ["Interest rates", "Why did Westpac cut its New Zealand lending rate?"],
  ["Takeovers", "Which companies announced takeover bids?"],
  ["بالعربي", "ما الاتفاق الذي توصلت إليه غانا مع إيران بشأن النفط؟"],
];

// ---- small helpers
function el(tag, cls, html) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(method, path, body) {
  const res = await fetch(API + path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.signal || `HTTP ${res.status}`);
  return data;
}

function timeAgo(iso) {
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return "now";
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

function scrollToBottom() {
  const sc = $("scroller");
  sc.scrollTop = sc.scrollHeight;
}

// Tiny markdown: paragraphs, bullet/numbered lists, **bold**, *italic*, `code`, and [n] citations.
function renderMarkdown(text) {
  const inline = (line) => escapeHtml(line)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d{1,2})\]/g, '<button class="cite" data-n="$1">$1</button>');

  let html = "", list = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const bullet = line.match(/^[-*•]\s+(.*)/);
    const numbered = line.match(/^\d+[.)]\s+(.*)/);
    if (bullet || numbered) {
      const type = bullet ? "ul" : "ol";
      if (list !== type) { closeList(); html += `<${type}>`; list = type; }
      html += `<li>${inline((bullet || numbered)[1])}</li>`;
    } else if (line) {
      closeList();
      html += `<p>${inline(line)}</p>`;
    } else {
      closeList();
    }
  }
  closeList();
  return html;
}

// ---- sidebar: threads and models
async function loadThreads() {
  try {
    state.threads = (await api("GET", "/threads")).threads;
  } catch (e) {
    state.threads = [];
    $("threads").innerHTML = `<div class="empty-threads">${escapeHtml(e.message)}</div>`;
    return;
  }
  renderThreads();
}

function renderThreads() {
  const box = $("threads");
  box.innerHTML = "";
  if (!state.threads.length) {
    box.innerHTML = '<div class="empty-threads">No conversations yet.</div>';
  }
  for (const t of state.threads) {
    const item = el("div", "thread" + (t.id === state.threadId ? " active" : ""));
    item.innerHTML = `<span class="thread-title" dir="auto">${escapeHtml(t.title)}</span>
      <span class="thread-meta">${timeAgo(t.updated_at)}</span>
      <button class="thread-del" title="Delete"><svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg></button>`;
    item.onclick = () => openThread(t.id);
    item.querySelector(".thread-del").onclick = (ev) => { ev.stopPropagation(); deleteThread(t); };
    box.appendChild(item);
  }
  const current = state.threads.find((t) => t.id === state.threadId);
  $("topbar-title").textContent = current ? current.title : "";
}

async function loadModels() {
  try {
    const h = await api("GET", "/");
    $("models").innerHTML = `<span class="dot"></span>API connected<br>
      Embedding: ${escapeHtml(h.embedding_model)}<br>Reranker: ${escapeHtml(h.reranker_model || "off")}`;
  } catch {
    $("models").innerHTML = '<span class="dot off"></span>API not reachable';
  }
}

async function openThread(id) {
  if (state.busy) return;
  state.threadId = id;
  renderThreads();
  closeSidebar();
  showView("chat");
  const box = $("messages");
  box.innerHTML = "";
  try {
    const data = await api("GET", `/threads/${id}`);
    for (const m of data.messages) {
      if (m.role === "user") addUserMessage(m.content);
      else renderAnswer(addAssistantMessage(), m.content, m.details || {});
    }
  } catch (e) {
    box.appendChild(notice("error", "Could not load the conversation", e.message));
  }
  scrollToBottom();
}

async function deleteThread(t) {
  if (!confirm(`Delete "${t.title}"?`)) return;
  try { await api("DELETE", `/threads/${t.id}`); } catch (e) { alert(e.message); return; }
  if (t.id === state.threadId) newChat();
  loadThreads();
}

function newChat() {
  if (state.busy) return;
  state.threadId = null;
  renderThreads();
  showWelcome();
  closeSidebar();
  showView("chat");
  $("question").focus();
}

// ---- messages
function showWelcome() {
  const box = $("messages");
  box.innerHTML = "";
  const w = el("div", "welcome", `<div class="logo">N</div>
    <h1>What would you like to know?</h1>
    <p>Answers come only from Reuters financial news (1987), with the sources cited.
       Ask in English or Arabic; follow-up questions remember the conversation.</p>`);
  const grid = el("div", "suggestions");
  for (const [topic, q] of SUGGESTIONS) {
    const b = el("button", "suggestion", `<small>${escapeHtml(topic)}</small><span dir="auto">${escapeHtml(q)}</span>`);
    b.onclick = () => send(q);
    grid.appendChild(b);
  }
  w.appendChild(grid);
  box.appendChild(w);
}

function addUserMessage(text) {
  const msg = el("div", "msg user");
  const bubble = el("div", "bubble");
  bubble.dir = "auto";
  bubble.textContent = text;
  msg.appendChild(bubble);
  $("messages").appendChild(msg);
}

function addAssistantMessage() {
  const msg = el("div", "msg assistant", '<div class="avatar">N</div>');
  const body = el("div", "body");
  msg.appendChild(body);
  $("messages").appendChild(msg);
  return body;
}

function notice(kind, title, reason) {
  return el("div", `notice ${kind}`, `<div><strong>${escapeHtml(title)}</strong>${reason ? " · " + escapeHtml(reason) : ""}</div>`);
}

// Draw a finished answer: text, notice, "searched as", sources and a footer.
function renderAnswer(body, content, d) {
  body.innerHTML = "";
  const answer = el("div", "answer", renderMarkdown(content || ""));
  answer.dir = "auto";
  body.appendChild(answer);

  if (NOTICES[d.status]) {
    const [kind, title] = NOTICES[d.status];
    body.appendChild(notice(kind, title, d.reason));
  }
  if (d.search_question && d.question && d.search_question !== d.question) {
    body.appendChild(el("div", "searched-as", `Searched as: <b dir="auto">${escapeHtml(d.search_question)}</b>`));
  }

  const sources = d.status === "answered" ? d.sources || [] : [];
  if (sources.length) {
    const wrap = el("div", "sources", `<div class="sources-head">Sources · ${sources.length}</div>`);
    const list = el("div", "source-list");
    for (const s of sources) {
      const score = s.rerank_score != null ? `rerank ${s.rerank_score.toFixed(2)}` : "vector order";
      const card = el("button", "source", `<div class="source-top">
          <span class="source-n">${s.n}</span><span class="source-id">${escapeHtml(s.doc_id)}</span>
          <span class="tag">${escapeHtml(s.category)}</span></div>
        <div class="source-snippet">${escapeHtml(s.text)}</div>
        <div class="score">${score}</div>`);
      card.dataset.n = s.n;
      card.onclick = () => card.classList.toggle("open");
      list.appendChild(card);
    }
    wrap.appendChild(list);
    body.appendChild(wrap);
    // Clicking [n] in the answer opens that source
    answer.querySelectorAll(".cite").forEach((c) => {
      c.onclick = () => {
        const card = list.querySelector(`.source[data-n="${c.dataset.n}"]`);
        if (!card) return;
        card.classList.add("open", "flash");
        card.scrollIntoView({ behavior: "smooth", block: "nearest" });
        setTimeout(() => card.classList.remove("flash"), 1200);
      };
    });
  }

  const foot = el("div", "msg-foot");
  const copy = el("button", "", '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/></svg>Copy');
  copy.onclick = async () => {
    await navigator.clipboard.writeText(content);
    copy.lastChild.textContent = "Copied";
    setTimeout(() => (copy.lastChild.textContent = "Copy"), 1500);
  };
  foot.appendChild(copy);

  const timings = d.timings || {};
  if (Object.keys(timings).length) {
    const total = Object.values(timings).reduce((a, b) => a + b, 0);
    const btn = el("button", "", `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>${(total / 1000).toFixed(1)} s`);
    const rerank = d.reranked ? `reranker on ${d.reranker_device || "?"}` : "reranker off";
    const detail = el("span", "timings",
      Object.entries(timings).map(([k, v]) => `${k.replace(/_ms$/, "")} ${v} ms`).join(" · ") + ` · ${rerank}`);
    btn.onclick = () => detail.classList.toggle("show");
    foot.append(btn, detail);
  }
  body.appendChild(foot);
}

// ---- sending a question (streamed)
async function send(question) {
  question = question.trim();
  if (!question || state.busy) return;
  setBusy(true);
  if (!state.threadId) $("messages").innerHTML = "";  // remove the welcome screen

  addUserMessage(question);
  const body = addAssistantMessage();
  const progress = el("div", "progress");
  const answer = el("div", "answer streaming");
  answer.dir = "auto";
  body.append(progress, answer);
  const done = new Set();
  const drawProgress = () => {  // done steps get a tick; steps whose inputs are all done are busy
    const isActive = (name, waitsFor) => !done.has(name) && waitsFor.every((w) => done.has(w));
    progress.innerHTML = STEPS.filter(([name, , waitsFor]) => done.has(name) || isActive(name, waitsFor))
      .map(([name, label]) => `<span class="step ${done.has(name) ? "done" : "active"}">${done.has(name) ? "✓ " : ""}${label}</span>`)
      .join("");
  };
  drawProgress();
  scrollToBottom();

  let text = "", final = null;
  try {
    const res = await fetch(`${API}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: state.threadId, rerank: $("rerank").checked }),
    });
    if ((res.headers.get("content-type") || "").startsWith("application/json")) {
      throw new Error((await res.json()).signal || `HTTP ${res.status}`);
    }
    // The stream is newline-delimited JSON events: thread, step, token, restart, final
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === "thread") {
          state.threadId = ev.thread_id;
        } else if (ev.type === "step") {
          done.add(ev.name);
          drawProgress();
        } else if (ev.type === "token") {
          text += ev.text;
          answer.innerHTML = renderMarkdown(text);
        } else if (ev.type === "restart") {
          text = "";
          done.delete("generate");
          done.delete("output_guard");
          drawProgress();
          answer.innerHTML = '<p class="muted">The answer did not pass the checks; writing it again ...</p>';
        } else if (ev.type === "final") {
          final = ev;
        }
      }
      scrollToBottom();
    }
    if (!final) throw new Error("The stream ended without an answer");
    renderAnswer(body, final.answer, final);  // the checked answer replaces the streamed draft
  } catch (e) {
    body.innerHTML = "";
    body.appendChild(notice("error", "Could not get an answer", e.message));
  }
  setBusy(false);
  scrollToBottom();
  loadThreads();
}

function setBusy(busy) {
  state.busy = busy;
  updateSendButton();
}

function updateSendButton() {
  $("send").disabled = state.busy || !$("question").value.trim();
}

// ---- search lab
async function runSearch(ev) {
  ev.preventDefault();
  const query = $("search-q").value.trim();
  if (!query) return;
  const out = $("search-results");
  out.innerHTML = '<div class="muted">Searching ...</div>';
  $("search-meta").textContent = "";
  let r;
  try {
    r = await api("POST", "/nlp/search", {
      query, top_k: +$("search-k").value, candidates: +$("search-c").value, rerank: $("search-rerank").checked,
    });
  } catch (e) {
    out.innerHTML = "";
    out.appendChild(notice("error", "Search failed", e.message));
    return;
  }
  $("search-meta").textContent = Object.entries(r.timings || {}).map(([k, v]) => `${k.replace(/_ms$/, "")} ${v} ms`).join(" · ");

  const hit = (h, rank, extra) => `<div class="hit"><div class="hit-top">
      <span class="hit-rank">${rank}</span><strong>${escapeHtml(h.doc_id)}</strong>
      <span class="tag">${escapeHtml(h.metadata?.category || "")}</span><span class="score">${extra}</span></div>
      <div class="muted" dir="auto">${escapeHtml(h.text.slice(0, 260))}${h.text.length > 260 ? " ..." : ""}</div></div>`;

  const before = r.before_rerank.map((h, i) => hit(h, i + 1, `cosine ${h.vector_score.toFixed(3)}`)).join("");
  let after = '<div class="muted">Reranking is off.</div>';
  if (r.reranker_model) {
    after = r.after_rerank.map((h, i) => {
      const move = h.vector_rank - (i + 1);
      const arrow = move > 0 ? `<span class="move up">▲ ${move}</span>` : move < 0 ? `<span class="move down">▼ ${-move}</span>` : '<span class="move same">=</span>';
      return hit(h, i + 1, `rerank ${h.rerank_score.toFixed(3)} · was #${h.vector_rank} ${arrow}`);
    }).join("");
  }
  out.innerHTML = `<div><h3>Before reranking</h3>${before}</div><div><h3>After reranking</h3>${after}</div>`;
}

// ---- layout: views and mobile sidebar
function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  $("view-chat").classList.toggle("hidden", name !== "chat");
  $("view-search").classList.toggle("hidden", name !== "search");
}

function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("scrim").classList.remove("show");
}

// ---- wiring
const textarea = $("question");
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
  updateSendButton();
});
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = textarea.value;
  textarea.value = "";
  textarea.style.height = "auto";
  updateSendButton();
  send(q);
});
$("new-chat").onclick = newChat;
$("search-form").addEventListener("submit", runSearch);
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showView(t.dataset.view)));
$("menu-btn").onclick = () => { $("sidebar").classList.add("open"); $("scrim").classList.add("show"); };
$("scrim").onclick = closeSidebar;
$("search-rerank").checked = $("rerank").checked;
$("rerank").onchange = () => ($("search-rerank").checked = $("rerank").checked);

showWelcome();
loadThreads();
loadModels();
textarea.focus();
