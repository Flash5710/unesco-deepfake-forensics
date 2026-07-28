const API_BASE = "http://localhost:8000";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyze-link",
    title: "🔊 Analizar enlace con DeepForensic",
    contexts: ["link"],
  });
  chrome.contextMenus.create({
    id: "analyze-page",
    title: "🔊 Analizar esta página con DeepForensic",
    contexts: ["page", "video"],
  });
});

chrome.storage.onChanged.addListener((changes) => {
  if (changes.df_lang) {
    updateContextMenus(changes.df_lang.newValue);
  }
});

async function updateContextMenus(lang) {
  const L = (k) => {
    const entry = {
      menu_analyze_link: {
        es: "🔊 Analizar enlace con DeepForensic",
        en: "🔊 Analyze link with DeepForensic",
      },
      menu_analyze_page: {
        es: "🔊 Analizar esta página con DeepForensic",
        en: "🔊 Analyze this page with DeepForensic",
      },
    };
    return entry[k]?.[lang] || entry[k]?.es;
  };

  try {
    await chrome.contextMenus.removeAll();
    chrome.contextMenus.create({
      id: "analyze-link",
      title: L("menu_analyze_link"),
      contexts: ["link"],
    });
    chrome.contextMenus.create({
      id: "analyze-page",
      title: L("menu_analyze_page"),
      contexts: ["page", "video"],
    });
  } catch (e) {}
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const url = info.linkUrl || info.pageUrl;
  if (!url) return;

  const result = await analyzeUrl(url);
  if (result?.pdf_base64) {
    await saveResult(url, result);
    const stored = await chrome.storage.local.get("df_lang");
    const lang = stored.df_lang || "es";
    const label =
      lang === "en"
        ? `DeepForensic: ${result.veredicto} — Confidence: ${result.confianza.toFixed(2)}%`
        : `DeepForensic: ${result.veredicto} — Confianza: ${result.confianza.toFixed(2)}%`;
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon128.png",
      title: "DeepForensic: " + result.veredicto,
      message: label,
      priority: result.veredicto === "DEEPFAKE" ? 2 : 0,
    });
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "analyze_url") {
    analyzeUrl(message.url)
      .then(async (result) => {
        if (result) await saveResult(message.url, result);
        sendResponse(result);
      })
      .catch((err) => sendResponse({ error: err.message }));
    return true;
  }

  if (message.type === "get_last_result") {
    chrome.storage.local.get("lastResult", (data) => {
      sendResponse(data.lastResult || null);
    });
    return true;
  }

  if (message.type === "get_history") {
    chrome.storage.local.get("df_history", (data) => {
      sendResponse(data.df_history || []);
    });
    return true;
  }

  if (message.type === "clear_history") {
    chrome.storage.local.remove("df_history", () => sendResponse({ ok: true }));
    return true;
  }

  if (message.type === "get_api_base") {
    sendResponse(API_BASE);
    return true;
  }

  if (message.type === "get_lang") {
    chrome.storage.local.get("df_lang", (data) => {
      sendResponse(data.df_lang || "es");
    });
    return true;
  }
});

async function saveResult(url, result) {
  const data = await chrome.storage.local.get("df_history");
  const history = data.df_history || [];
  history.unshift({
    url,
    veredicto: result.veredicto,
    confianza: result.confianza,
    inestabilidad: result.inestabilidad,
    timestamp: Date.now(),
  });
  if (history.length > 100) history.length = 100;
  await chrome.storage.local.set({ df_history: history, lastResult: result });
}

async function analyzeUrl(url) {
  try {
    const res = await fetch(`${API_BASE}/analyze_url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Error ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.error("DeepForensic analyze error:", err);
    return null;
  }
}
