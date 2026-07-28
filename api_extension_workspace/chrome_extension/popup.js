const $ = (id) => document.getElementById(id);

let API_BASE = "http://localhost:8000";

const tabFile = $("tab-file");
const tabUrl = $("tab-url");
const dropZone = $("drop-zone");
const fileInput = $("file-input");
const urlInput = $("url-input");
const analyzeUrlBtn = $("analyze-url-btn");
const captureTabBtn = $("capture-tab-btn");
const tabHistory = $("tab-history");
const loading = $("loading");
const result = $("result");
const error = $("error");
const errorText = $("error-text");
const retryBtn = $("retry-btn");
const veredictoBadge = $("veredicto-badge");
const confianzaEl = $("confianza");
const inestabilidadEl = $("inestabilidad");
const downloadPdfBtn = $("download-pdf-btn");
const newAnalysisBtn = $("new-analysis-btn");
const historyList = $("history-list");
const clearHistoryBtn = $("clear-history-btn");

let lastResult = null;

(async () => {
  try {
    const resp = await chrome.runtime.sendMessage({ type: "get_api_base" });
    if (resp) API_BASE = resp;
  } catch {}
  const stored = await chrome.storage.local.get("lastResult");
  if (stored.lastResult) lastResult = stored.lastResult;
})();

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    const map = { file: tabFile, url: tabUrl, history: tabHistory };
    const target = map[btn.dataset.tab] || tabFile;
    target.classList.add("active");
    hideAll();
    if (btn.dataset.tab === "history") loadHistory();
  });
});

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    analyzeFile(files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    analyzeFile(fileInput.files[0]);
  }
});

analyzeUrlBtn.addEventListener("click", () => {
  const url = urlInput.value.trim();
  if (!url) return;
  analyzeUrl(url);
});

urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") analyzeUrlBtn.click();
});

captureTabBtn.addEventListener("click", async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url) {
      urlInput.value = tab.url;
      analyzeUrlBtn.click();
    }
  } catch (err) {
    showError("No se pudo capturar la URL: " + err.message);
  }
});

retryBtn.addEventListener("click", () => {
  hideAll();
});

newAnalysisBtn.addEventListener("click", () => {
  hideAll();
  fileInput.value = "";
  urlInput.value = "";
});

downloadPdfBtn.addEventListener("click", () => {
  if (!lastResult || !lastResult.pdf_base64) return;
  const byteChars = atob(lastResult.pdf_base64);
  const byteNums = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) {
    byteNums[i] = byteChars.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNums);
  const blob = new Blob([byteArray], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `deepforensic-report-${Date.now()}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

async function analyzeFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  await doFetch("/analyze_file", { method: "POST", body: formData });
}

async function analyzeUrl(url) {
  hideAll();
  loading.classList.remove("hidden");
  error.classList.add("hidden");
  result.classList.add("hidden");

  try {
    const data = await chrome.runtime.sendMessage({
      type: "analyze_url",
      url: url,
    });
    if (!data || data.error) {
      throw new Error(data?.error || "Error al analizar la URL");
    }
    showResult(data);
  } catch (err) {
    showError(err.message);
  }
}

async function doFetch(endpoint, options) {
  hideAll();
  loading.classList.remove("hidden");
  error.classList.add("hidden");
  result.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}${endpoint}`, options);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Error ${res.status}`);
    }
    const data = await res.json();
    showResult(data);
  } catch (err) {
    showError(err.message);
  }
}

function showResult(data) {
  loading.classList.add("hidden");
  lastResult = data;

  const isDeepfake = data.veredicto === "DEEPFAKE";
  veredictoBadge.textContent = data.veredicto;
  veredictoBadge.className = `badge ${isDeepfake ? "deepfake" : "real"}`;

  confianzaEl.textContent = `${data.confianza.toFixed(2)}%`;
  inestabilidadEl.textContent = data.inestabilidad.toFixed(4);

  result.classList.remove("hidden");
}

function showError(msg) {
  loading.classList.add("hidden");
  errorText.textContent = msg;
  error.classList.remove("hidden");
}

function hideAll() {
  loading.classList.add("hidden");
  result.classList.add("hidden");
  error.classList.add("hidden");
}

clearHistoryBtn.addEventListener("click", async () => {
  await chrome.runtime.sendMessage({ type: "clear_history" });
  loadHistory();
});

async function loadHistory() {
  const history = await chrome.runtime.sendMessage({ type: "get_history" });
  if (!history || history.length === 0) {
    historyList.innerHTML = `<div class="history-empty">Sin análisis anteriores</div>`;
    return;
  }
  historyList.innerHTML = history
    .map(
      (h) => `
    <div class="history-item" data-url="${h.url}">
      <div class="history-badge ${h.veredicto === "DEEPFAKE" ? "fake" : "real"}"></div>
      <div class="history-info">
        <div class="history-url">${h.url}</div>
        <div class="history-meta">${h.veredicto} · ${h.confianza.toFixed(2)}% · ${new Date(h.timestamp).toLocaleString()}</div>
      </div>
    </div>
  `
    )
    .join("");

  historyList.querySelectorAll(".history-item").forEach((item) => {
    item.addEventListener("click", () => {
      const urlInput = $("url-input");
      urlInput.value = item.dataset.url;
      document.querySelector('[data-tab="url"]')?.click();
    });
  });
}
