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

  const style = document.createElement("style");
  style.textContent = `
    #df-fab {
      all: initial;
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 9999999;
      width: 52px;
      height: 52px;
      border-radius: 50%;
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 22px;
      cursor: pointer;
      box-shadow: 0 4px 20px rgba(102,126,234,0.5);
      border: none;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.2s, box-shadow 0.2s;
      user-select: none;
    }
    #df-fab:hover {
      transform: scale(1.08);
      box-shadow: 0 6px 28px rgba(102,126,234,0.65);
    }
    #df-fab.df-spin {
      animation: df-fab-spin 0.8s linear infinite;
    }
    @keyframes df-fab-spin {
      to { transform: rotate(360deg); }
    }

    #df-modal {
      all: initial;
      position: fixed;
      bottom: 88px;
      right: 24px;
      z-index: 9999999;
      min-width: 260px;
      max-width: 320px;
      background: #0f1117;
      border: 1px solid #2a2d3a;
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 8px 40px rgba(0,0,0,0.6);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #e1e4eb;
      opacity: 0;
      transform: translateY(12px) scale(0.96);
      transition: opacity 0.25s, transform 0.25s;
      pointer-events: none;
    }
    #df-modal.show {
      opacity: 1;
      transform: translateY(0) scale(1);
      pointer-events: auto;
    }
    #df-modal .df-m-title {
      font-size: 14px;
      font-weight: 600;
      color: #8b8fa3;
      margin-bottom: 10px;
    }
    #df-modal .df-m-badge {
      font-size: 24px;
      font-weight: 700;
      margin-bottom: 12px;
    }
    #df-modal .df-m-badge.fake { color: #ef4444; }
    #df-modal .df-m-badge.real { color: #22c55e; }
    #df-modal .df-m-row {
      display: flex;
      justify-content: space-between;
      padding: 4px 0;
      font-size: 13px;
    }
    #df-modal .df-m-row .l { color: #8b8fa3; }
    #df-modal .df-m-row .v { font-weight: 600; }
    #df-modal .df-m-url {
      font-size: 11px;
      color: #5a5e72;
      margin-top: 10px;
      word-break: break-all;
    }
    #df-modal .df-m-close {
      position: absolute;
      top: 8px;
      right: 10px;
      background: none;
      border: none;
      color: #5a5e72;
      cursor: pointer;
      font-size: 16px;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: inherit;
    }
    #df-modal .df-m-close:hover {
      background: #1a1d28;
      color: #e1e4eb;
    }
    #df-modal .df-m-loading {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 16px 0;
    }
    #df-modal .df-m-spinner {
      width: 18px;
      height: 18px;
      border: 2px solid #2a2d3a;
      border-top-color: #667eea;
      border-radius: 50%;
      animation: df-spin 0.7s linear infinite;
    }
    @keyframes df-spin { to { transform: rotate(360deg); } }
  `;
  document.head.appendChild(style);

  /* Button */
  const fab = document.createElement("div");
  fab.id = "df-fab";
  fab.textContent = "🔍";
  document.body.appendChild(fab);

  /* Modal */
  const modal = document.createElement("div");
  modal.id = "df-modal";
  document.body.appendChild(modal);

  function showModal(html) {
    modal.innerHTML = html;
    modal.classList.add("show");
    const close = modal.querySelector(".df-m-close");
    if (close) close.addEventListener("click", () => modal.classList.remove("show"));
  }

  function closeModal() {
    modal.classList.remove("show");
  }

  fab.addEventListener("click", async () => {
    if (fab.classList.contains("df-spin")) return;

    fab.classList.add("df-spin");
    showModal(`
      <button class="df-m-close">✕</button>
      <div class="df-m-loading">
        <div class="df-m-spinner"></div>
        <span style="color:#8b8fa3;font-size:13px;">Analizando...</span>
      </div>
    `);

    try {
      const data = await chrome.runtime.sendMessage({
        type: "analyze_url",
        url: window.location.href,
      });

      fab.classList.remove("df-spin");

      if (!data || data.error) throw new Error(data?.error || "Error del servidor");

      const isFake = data.veredicto === "DEEPFAKE";
      showModal(`
        <button class="df-m-close">✕</button>
        <div class="df-m-title">Resultado Forense</div>
        <div class="df-m-badge ${isFake ? "fake" : "real"}">${data.veredicto}</div>
        <div class="df-m-row"><span class="l">Confianza</span><span class="v">${data.confianza.toFixed(2)}%</span></div>
        <div class="df-m-row"><span class="l">Inestabilidad de fase</span><span class="v">${data.inestabilidad.toFixed(4)}</span></div>
        <div class="df-m-url">${window.location.hostname}</div>
      `);

      const h = (await chrome.storage.local.get("df_history")).df_history || [];
      h.unshift({ url: window.location.href, veredicto: data.veredicto, confianza: data.confianza, inestabilidad: data.inestabilidad, timestamp: Date.now() });
      if (h.length > 100) h.length = 100;
      await chrome.storage.local.set({ df_history: h, lastResult: data });

    } catch (err) {
      fab.classList.remove("df-spin");
      showModal(`
        <button class="df-m-close">✕</button>
        <div class="df-m-title" style="color:#ef4444;">Error</div>
        <div style="color:#8b8fa3;font-size:13px;">${err.message}</div>
      `);
    }
  });
})();
