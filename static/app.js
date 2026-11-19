const EXAMPLES = [
  "What are the four core functions of the NIST AI RMF?",
  "Which of these functions is cross-cutting?",
  "What is the purpose of the Govern function?",
  "What major risks does the Generative AI Profile discuss?",
  "Who won the FIFA World Cup?",
];

const thread = document.getElementById("thread");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const send = document.getElementById("send");
const examples = document.getElementById("examples");
const reset = document.getElementById("reset");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

let history = [];

function setStatus(ok, text) {
  statusDot.className = `status-dot ${ok ? "ok" : "bad"}`;
  statusText.textContent = text;
}

function showEmpty() {
  thread.innerHTML = `
    <div class="empty">
      <h2>Start with a grounded question</h2>
      <p>The assistant searches the indexed NIST PDFs. Follow-ups such as “which one is cross-cutting?” use the conversation so far.</p>
    </div>`;
}

function addBubble(role, html) {
  const empty = thread.querySelector(".empty");
  if (empty) empty.remove();
  const el = document.createElement("article");
  el.className = `bubble ${role}`;
  el.innerHTML = html;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
  return el;
}

function renderSources(sources) {
  if (!sources?.length) return "";
  const chips = sources
    .map((source) => {
      const section = source.section && source.section !== "Not detected"
        ? source.section
        : "Section not detected";
      return `<span class="chip">${escapeHtml(source.document)} · p. ${escapeHtml(String(source.page))} · ${escapeHtml(section)}</span>`;
    })
    .join("");
  return `<div class="meta">Sources</div><div class="sources">${chips}</div>`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function ask(question) {
  send.disabled = true;
  addBubble("user", `<p class="answer">${escapeHtml(question)}</p>`);
  const pending = addBubble("assistant pending", `<p class="answer">Searching the NIST corpus…</p>`);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, chat_history: history }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }

    const rewritten =
      payload.standalone_question &&
      payload.standalone_question.trim() !== question.trim();
    const retryNote = payload.retry_count
      ? ` Retrieval was retried ${payload.retry_count} time(s).`
      : "";

    pending.classList.remove("pending");
    pending.innerHTML = `
      <p class="answer">${escapeHtml(payload.answer || "No answer returned.")}</p>
      ${renderSources(payload.sources)}
      ${rewritten || retryNote ? `<p class="meta-row">${rewritten ? `Interpreted as: ${escapeHtml(payload.standalone_question)}` : ""}${retryNote}</p>` : ""}
    `;

    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: payload.answer || "" });
    if (history.length > 10) history = history.slice(-10);
    setStatus(true, "Ready");
  } catch (error) {
    pending.classList.remove("pending");
    pending.innerHTML = `<p class="answer">I could not complete that request: ${escapeHtml(error.message)}</p>`;
    setStatus(false, "Request failed");
  } finally {
    send.disabled = false;
    thread.scrollTop = thread.scrollHeight;
  }
}

EXAMPLES.forEach((text) => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = text;
  button.addEventListener("click", () => {
    input.value = text;
    form.requestSubmit();
  });
  examples.appendChild(button);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || send.disabled) return;
  input.value = "";
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

reset.addEventListener("click", () => {
  history = [];
  showEmpty();
});

showEmpty();

fetch("/health")
  .then((response) => (response.ok ? setStatus(true, "Ready") : Promise.reject()))
  .catch(() => setStatus(false, "API unavailable"));
