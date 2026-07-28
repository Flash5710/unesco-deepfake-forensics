const STRINGS = {
  "meta": { "version": 1, "source_lang": "es" },
  "popup": {
    "app_title": { "es": "DeepForensic", "en": "DeepForensic" },
    "subtitle": { "es": "Detector de Deepfake Audio", "en": "Deepfake Audio Detector" },
    "tab_file": { "es": "Archivo", "en": "File" },
    "tab_url": { "es": "URL", "en": "URL" },
    "tab_history": { "es": "Historial", "en": "History" },
    "drop_text": { "es": "Suelta un archivo aquí o haz clic para seleccionar", "en": "Drop a file here or click to select" },
    "drop_hint": { "es": ".wav .mp4 .mp3 .m4a", "en": ".wav .mp4 .mp3 .m4a" },
    "url_label": { "es": "URL de video/audio:", "en": "Video/audio URL:" },
    "url_placeholder": { "es": "https://twitter.com/...", "en": "https://twitter.com/..." },
    "capture_title": { "es": "Capturar URL de la pestaña activa", "en": "Capture current tab URL" },
    "analyze_url_btn": { "es": "Analizar URL", "en": "Analyze URL" },
    "loading": { "es": "Analizando...", "en": "Analyzing..." },
    "confidence": { "es": "Confianza", "en": "Confidence" },
    "phase_instability": { "es": "Inestabilidad de fase", "en": "Phase Instability" },
    "download_pdf": { "es": "Descargar PDF", "en": "Download PDF" },
    "new_analysis": { "es": "Nuevo análisis", "en": "New analysis" },
    "retry": { "es": "Reintentar", "en": "Retry" },
    "clear_history": { "es": "Limpiar historial", "en": "Clear history" },
    "no_history": { "es": "Sin análisis anteriores", "en": "No previous analyses" },
    "error_generic": { "es": "Error al analizar la URL", "en": "Error analyzing URL" },
    "lang_es": { "es": "ES", "en": "ES" },
    "lang_en": { "es": "EN", "en": "EN" },
    "listen": { "es": "Escuchar resultado", "en": "Listen to result" }
  },
  "content": {
    "analyzing": { "es": "Analizando...", "en": "Analyzing..." },
    "forensic_result": { "es": "Resultado Forense", "en": "Forensic Result" },
    "confidence": { "es": "Confianza", "en": "Confidence" },
    "phase_instability": { "es": "Inestabilidad de fase", "en": "Phase Instability" },
    "error": { "es": "Error", "en": "Error" },
    "fab_aria": { "es": "Analizar esta página con DeepForensic", "en": "Analyze this page with DeepForensic" },
    "listen": { "es": "Escuchar resultado", "en": "Listen to result" },
    "close": { "es": "Cerrar", "en": "Close" },
    "server_error": { "es": "Error del servidor", "en": "Server error" }
  },
  "background": {
    "menu_analyze_link": { "es": "🔊 Analizar enlace con DeepForensic", "en": "🔊 Analyze link with DeepForensic" },
    "menu_analyze_page": { "es": "🔊 Analizar esta página con DeepForensic", "en": "🔊 Analyze this page with DeepForensic" },
    "notif_title": { "es": "DeepForensic: ", "en": "DeepForensic: " },
    "notif_confidence": { "es": "Confianza: ", "en": "Confidence: " }
  }
};

(function () {
  const I18N = window.__DF_I18N || {};
  let currentLang = "es";

  I18N.getLang = function () {
    return currentLang;
  };

  I18N.setLang = function (lang, persist) {
    if (lang !== "es" && lang !== "en") lang = "es";
    currentLang = lang;
    if (persist !== false && typeof chrome !== "undefined" && chrome.storage) {
      try {
        chrome.storage.local.set({ df_lang: lang });
      } catch (e) {}
    }
    document.documentElement?.setAttribute("lang", lang === "en" ? "en" : "es");
  };

  I18N.init = async function () {
    try {
      const data = await chrome.storage.local.get("df_lang");
      const stored = data.df_lang || navigator.language?.slice(0, 2);
      I18N.setLang(stored === "en" ? "en" : "es", false);
    } catch {
      I18N.setLang("es", false);
    }
    return currentLang;
  };

  I18N.t = function (section, key, params) {
    const entry = STRINGS[section]?.[key];
    if (!entry) return key;
    let val = entry[currentLang] || entry["es"] || key;
    if (Array.isArray(val)) val = val.join(", ");
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        val = val.replace(new RegExp(`\\{${k}\\}`, "g"), v);
      }
    }
    return val;
  };

  I18N.speak = function (text, lang) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === "en" ? "en-US" : "es-ES";
    u.rate = 0.9;
    u.pitch = 1.0;
    window.speechSynthesis.speak(u);
  };

  window.__DF_I18N = I18N;
})();
