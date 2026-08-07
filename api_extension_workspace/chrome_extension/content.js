(function () {
  const host = window.location.hostname.toLowerCase();
  const supported =
    host.includes("youtube.com") || host.includes("youtu.be") ||
    host.includes("twitter.com") || host.includes("x.com") ||
    host.includes("instagram.com") || host.includes("tiktok.com") ||
    host.includes("facebook.com") || host.includes("fb.watch") ||
    host.includes("fb.com");
  if (!supported) return;
  console.log("[DeepForensic] Active");

  let lang = "es";

  (async () => {
    try {
      const resp = await chrome.runtime.sendMessage({ type: "get_lang" });
      if (resp) lang = resp;
    } catch {}
    document.documentElement?.setAttribute("lang", lang === "en" ? "en" : "es");
  })();

  function t(key) {
    const dict = {
      analyzing: { es: "Analizando...", en: "Analyzing..." },
      forensic_result: { es: "Resultado Forense", en: "Forensic Result" },
      confidence: { es: "Confianza", en: "Confidence" },
      phase_instability: { es: "Inestabilidad de fase", en: "Phase Instability" },
      error: { es: "Error", en: "Error" },
      fab_aria: { es: "Analizar esta página con DeepForensic", en: "Analyze this page with DeepForensic" },
      listen: { es: "Escuchar resultado", en: "Listen to result" },
      close: { es: "Cerrar", en: "Close" },
      server_error: { es: "Error del servidor", en: "Server error" },
    };
    const entry = dict[key];
    return entry ? (entry[lang] || entry.es) : key;
  }

  const style = document.createElement("style");
  style.textContent = `
    #df-fab, #df-modal { all: initial; }
    #df-fab {
      position: fixed; bottom: 24px; right: 24px; z-index: 9999999;
      width: 52px; height: 52px; border-radius: 50%;
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: #fff; font: 22px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      cursor: pointer; box-shadow: 0 4px 20px rgba(102,126,234,0.5);
      border: none; display: flex; align-items: center; justify-content: center;
      transition: transform .2s,box-shadow .2s; user-select: none;
    }
    #df-fab:hover { transform: scale(1.08); box-shadow: 0 6px 28px rgba(102,126,234,0.65); }
    #df-fab:focus-visible { outline: 2px solid #fff; outline-offset: 3px; }
    #df-fab.df-spin { animation: df-spin .8s linear infinite; }
    @keyframes df-spin { to { transform: rotate(360deg); } }
    #df-modal {
      position: fixed; bottom: 88px; right: 24px; z-index: 9999999;
      min-width: 260px; max-width: 320px;
      background: #0f1117; border: 1px solid #2a2d3a; border-radius: 14px;
      padding: 20px; box-shadow: 0 8px 40px rgba(0,0,0,0.6);
      font: 13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      color: #e1e4eb;
      opacity: 0; transform: translateY(12px) scale(0.96);
      transition: opacity .25s,transform .25s; pointer-events: none;
    }
    #df-modal.show { opacity: 1; transform: translateY(0) scale(1); pointer-events: auto; }
    #df-modal .df-m-title { font-size: 14px; font-weight: 600; color: #8b8fa3; margin-bottom: 10px; }
    #df-modal .df-m-badge { font-size: 24px; font-weight: 700; margin-bottom: 12px; }
    #df-modal .df-m-badge.fake { color: #ef4444; }
    #df-modal .df-m-badge.real { color: #22c55e; }
    #df-modal .df-m-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }
    #df-modal .df-m-row .l { color: #8b8fa3; }
    #df-modal .df-m-row .v { font-weight: 600; }
    #df-modal .df-m-url { font-size: 11px; color: #5a5e72; margin-top: 10px; word-break: break-all; }
    #df-modal .df-m-close {
      position: absolute; top: 8px; right: 10px;
      background: none; border: none; color: #5a5e72;
      cursor: pointer; font-size: 16px; padding: 2px 6px; border-radius: 4px; font-family: inherit;
    }
    #df-modal .df-m-close:hover { background: #1a1d28; color: #e1e4eb; }
    #df-modal .df-m-close:focus-visible { outline: 2px solid #667eea; outline-offset: 1px; }
    #df-modal .df-m-loading { display: flex; align-items: center; gap: 10px; padding: 16px 0; }
    #df-modal .df-m-spinner { width: 18px; height: 18px; border: 2px solid #2a2d3a; border-top-color: #667eea; border-radius: 50%; animation: df-spin2 .7s linear infinite; }
    @keyframes df-spin2 { to { transform: rotate(360deg); } }
    #df-modal .df-m-listen {
      display: block; width: 100%; margin-top: 12px; padding: 8px;
      background: transparent; border: 1px solid #667eea; border-radius: 8px;
      color: #667eea; font: 13px inherit; font-weight: 600; cursor: pointer;
      transition: background .2s;
    }
    #df-modal .df-m-listen:hover { background: rgba(102,126,234,0.12); }
    #df-modal .df-m-listen:focus-visible { outline: 2px solid #667eea; outline-offset: 2px; }
  `;
  document.head.appendChild(style);

  const fab = document.createElement("div");
  fab.id = "df-fab";
  fab.textContent = "🔍";
  fab.setAttribute("role", "button");
  fab.setAttribute("tabindex", "0");
  fab.setAttribute("aria-label", t("fab_aria"));
  document.body.appendChild(fab);

  const modal = document.createElement("div");
  modal.id = "df-modal";
  modal.setAttribute("role", "dialog");
  modal.setAttribute("aria-label", t("forensic_result"));
  modal.setAttribute("aria-live", "polite");
  document.body.appendChild(modal);

  function showModal(html) {
    modal.innerHTML = html;
    modal.classList.add("show");
    const close = modal.querySelector(".df-m-close");
    if (close) close.addEventListener("click", () => modal.classList.remove("show"));
    const listen = modal.querySelector(".df-m-listen");
    if (listen) {
      listen.addEventListener("click", function () {
        speak(this.dataset.text);
      });
    }
  }

  function closeModal() {
    modal.classList.remove("show");
  }

  function speak(text) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === "en" ? "en-US" : "es-ES";
    u.rate = 0.9;
    window.speechSynthesis.speak(u);
  }

  fab.addEventListener("click", async () => {
    if (fab.classList.contains("df-spin")) return;
    fab.classList.add("df-spin");
    showModal(`
      <button class="df-m-close" aria-label="${t("close")}">✕</button>
      <div class="df-m-loading" role="status">
        <div class="df-m-spinner"></div>
        <span style="color:#8b8fa3;">${t("analyzing")}</span>
      </div>
    `);

    try {
      const data = await chrome.runtime.sendMessage({
        type: "analyze_url",
        url: window.location.href,
      });
      fab.classList.remove("df-spin");
      if (!data || data.error) throw new Error(data?.error || t("server_error"));

      const isFake = data.veredicto === "DEEPFAKE";
      const speakText = `${data.veredicto}. ${t("confidence")}: ${data.confianza.toFixed(2)}%. ${t("phase_instability")}: ${data.inestabilidad.toFixed(4)}.`;

      showModal(`
        <button class="df-m-close" aria-label="${t("close")}">✕</button>
        <div class="df-m-title">${t("forensic_result")}</div>
        <div class="df-m-badge ${isFake ? "fake" : "real"}" role="alert">${data.veredicto}</div>
        <div class="df-m-row"><span class="l">${t("confidence")}</span><span class="v">${data.confianza.toFixed(2)}%</span></div>
        <div class="df-m-row"><span class="l">${t("phase_instability")}</span><span class="v">${data.inestabilidad.toFixed(4)}</span></div>
        <div class="df-m-url">${window.location.hostname}</div>
        <button class="df-m-listen" data-text="${speakText.replace(/"/g, "&quot;")}" aria-label="${t("listen")}">🔊 ${t("listen")}</button>
      `);

      const h = (await chrome.storage.local.get("df_history")).df_history || [];
      h.unshift({ url: window.location.href, veredicto: data.veredicto, confianza: data.confianza, inestabilidad: data.inestabilidad, timestamp: Date.now() });
      if (h.length > 100) h.length = 100;
      await chrome.storage.local.set({ df_history: h, lastResult: data });

    } catch (err) {
      fab.classList.remove("df-spin");
      showModal(`
        <button class="df-m-close" aria-label="${t("close")}">✕</button>
        <div class="df-m-title" style="color:#ef4444;" role="alert">${t("error")}</div>
        <div style="color:#8b8fa3;font-size:13px;">${err.message}</div>
      `);
    }
  });

  fab.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fab.click(); }
  });
})();
