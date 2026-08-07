const $ = (id) => document.getElementById(id);

const tabFile = $("tab-file");
const tabUrl = $("tab-url");
const tabHistory = $("tab-history");
const dropZone = $("drop-zone");
const fileInput = $("file-input");
const urlInput = $("url-input");
const analyzeUrlBtn = $("analyze-url-btn");
const captureTabBtn = $("capture-tab-btn");
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
const listenResultBtn = $("listen-result-btn");
const langEs = $("lang-es");
const langEn = $("lang-en");

let API_BASE = "http://localhost:8000";
let lastResult = null;
let currentData = null;

function t(key) {
  return (window.__DF_I18N?.t) ? window.__DF_I18N.t("popup", key) : key;
}

async function applyLang(lang) {
  if (window.__DF_I18N) {
    window.__DF_I18N.setLang(lang, true);
  }

  document.documentElement.lang = lang === "en" ? "en" : "es";
  langEs.classList.toggle("active", lang === "es");
  langEn.classList.toggle("active", lang === "en");
  langEs.setAttribute("aria-checked", lang === "es");
  langEn.setAttribute("aria-checked", lang === "en");

  document.title = t("app_title");
  $("subtitle-text").textContent = t("subtitle");
  $("tab-btn-file").textContent = t("tab_file");
  $("tab-btn-url").textContent = t("tab_url");
  $("tab-btn-history").textContent = t("tab_history");

  const dropText = dropZone.querySelector(".drop-text");
  if (dropText) dropText.textContent = t("drop_text");
  const dropHint = dropZone.querySelector(".drop-hint");
  if (dropHint) dropHint.textContent = t("drop_hint");
  dropZone.setAttribute("aria-label", t("drop_text"));

  const urlLabel = $("url-label");
  if (urlLabel) urlLabel.textContent = t("url_label");
  urlInput.placeholder = t("url_placeholder");
  urlInput.setAttribute("aria-label", t("url_label"));
  captureTabBtn.title = t("capture_title");
  captureTabBtn.setAttribute("aria-label", t("capture_title"));
  analyzeUrlBtn.textContent = t("analyze_url_btn");
  analyzeUrlBtn.setAttribute("aria-label", t("analyze_url_btn"));

  const loadingText = loading.querySelector("p");
  if (loadingText) loadingText.textContent = t("loading");

  $("metric-confidence").textContent = t("confidence");
  $("metric-instability").textContent = t("phase_instability");
  downloadPdfBtn.textContent = t("download_pdf");
  downloadPdfBtn.setAttribute("aria-label", t("download_pdf"));
  newAnalysisBtn.textContent = t("new_analysis");
  listenResultBtn.innerHTML = "🔊 " + t("listen");
  listenResultBtn.setAttribute("aria-label", t("listen"));

  retryBtn.textContent = t("retry");
  clearHistoryBtn.textContent = t("clear_history");

  if (result.classList.contains("hidden") === false && currentData) {
    showResult(currentData);
  }
  if (error.classList.contains("hidden") === false) {
    const msg = errorText.textContent;
  }
}

(async () => {
  if (window.__DF_I18N) {
    await window.__DF_I18N.init();
    applyLang(window.__DF_I18N.getLang());
  }

  try {
    const resp = await chrome.runtime.sendMessage({ type: "get_api_base" });
    if (resp) API_BASE = resp;
  } catch {}
  const stored = await chrome.storage.local.get("lastResult");
  if (stored.lastResult) lastResult = stored.lastResult;
})();

langEs.addEventListener("click", () => applyLang("es"));
langEn.addEventListener("click", () => applyLang("en"));

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    const map = { file: tabFile, url: tabUrl, history: tabHistory };
    const target = map[btn.dataset.tab] || tabFile;
    target.classList.add("active");
    hideAll();
    if (btn.dataset.tab === "history") loadHistory();
  });
});

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });

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
    showError(t("error_generic") + ": " + err.message);
  }
});

retryBtn.addEventListener("click", () => {
  hideAll();
});

newAnalysisBtn.addEventListener("click", () => {
  hideAll();
  fileInput.value = "";
  urlInput.value = "";
  currentData = null;
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

listenResultBtn.addEventListener("click", () => {
  if (!currentData || !window.__DF_I18N) return;
  const lang = window.__DF_I18N.getLang();
  const isFake = currentData.veredicto === "DEEPFAKE";
  const text = `${currentData.veredicto}. ${t("confidence")}: ${currentData.confianza.toFixed(2)}%. ${t("phase_instability")}: ${currentData.inestabilidad.toFixed(4)}.`;
  window.__DF_I18N.speak(text, lang);
});

async function analyzeFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  // For file uploads we still use direct fetch since we can't go through background
  hideAll();
  loading.classList.remove("hidden");
  error.classList.add("hidden");
  result.classList.add("hidden");

  try {
    const lang = window.__DF_I18N?.getLang() || "es";
    const res = await fetch(`${API_BASE}/analyze_file?lang=${lang}`, { method: "POST", body: formData });
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
      throw new Error(data?.error || t("error_generic"));
    }
    showResult(data);
  } catch (err) {
    showError(err.message);
  }
}

function showResult(data) {
  loading.classList.add("hidden");
  lastResult = data;
  currentData = data;

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
    historyList.innerHTML = `<div class="history-empty">${t("no_history")}</div>`;
    return;
  }
  historyList.innerHTML = history
    .map(
      (h) => `
    <div class="history-item" data-url="${h.url}" role="listitem" tabindex="0">
      <div class="history-badge ${h.veredicto === "DEEPFAKE" ? "fake" : "real"}" aria-hidden="true"></div>
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
      urlInput.value = item.dataset.url;
      document.querySelector('[data-tab="url"]')?.click();
    });
    item.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        urlInput.value = item.dataset.url;
        document.querySelector('[data-tab="url"]')?.click();
      }
    });
  });
}
