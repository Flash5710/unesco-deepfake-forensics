import sys
import os
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch

from src.physics.audio_processing import (
    extraer_audio_de_video, calcular_stft_y_mel, generar_mapa_calor_interactivo,
    cargar_y_normalizar_audio, convertir_a_tensor_pytorch,
    predecir_audio_completo
)
from src.math_core.metrics import calcular_regularidad_fase
from src.ai.model import DeepfakeAudioCNN
from src.ai.custom_dataset import DeepfakeAudioDataset
from src.utils.social_downloader import descargar_video_de_red_social
from src.utils.pdf_generator import generar_reporte_forense
from src.ai.gradcam import generar_mapa_gradcam, _figura_superposicion
from src.physics.audio_processing import normalizar_espectrograma, HOP_LENGTH, MAX_FRAMES
from src.utils.i18n import t as _t, set_default_lang
import numpy as np
import json

TARGET_SR = 16000

# === Language selector (must be first) ===
if "df_lang" not in st.session_state:
    st.session_state.df_lang = "es"
set_default_lang(st.session_state.df_lang)

L = lambda k, **p: _t("app", k, st.session_state.df_lang, **p)
PDF_L = lambda k, **p: _t("pdf", k, st.session_state.df_lang, **p)


# === IA ===
@st.cache_resource
def inicializar_red_neuronal():
    try:
        modelo = DeepfakeAudioCNN()
        ruta_pesos = "src/ai/best_model_v2_augmented.pth"
        if os.path.exists(ruta_pesos):
            modelo.load_state_dict(torch.load(ruta_pesos, map_location=torch.device('cpu')))
        else:
            st.error(PDF_L("no_model_error", path=ruta_pesos))
        modelo.eval()
        return modelo
    except Exception as e:
        st.error(L("model_error", msg=e))
        return None


st.set_page_config(page_title=L("page_title"), page_icon="🔎", layout="wide")

# === Language selector in sidebar ===
with st.sidebar:
    st.markdown(f"### {L('lang_label')}")
    new_lang = st.radio(
        L("lang_label"),
        options=["es", "en"],
        format_func=lambda x: "Español" if x == "es" else "English",
        index=0 if st.session_state.df_lang == "es" else 1,
        label_visibility="collapsed",
        key="lang_radio",
    )
    if new_lang != st.session_state.df_lang:
        st.session_state.df_lang = new_lang
        set_default_lang(new_lang)
        st.rerun()

st.title(L("main_title"))
st.markdown(f"### {L('main_subtitle')}")
st.markdown("---")

modelo_ia = inicializar_red_neuronal()

if 'ultimo_archivo' not in st.session_state:
    st.session_state.ultimo_archivo = None

st.markdown(f"### {L('ingest_title')}")


def _procesar_y_mostrar(ruta_video_o_audio, ruta_audio, nombre_original, es_audio):
    archivos_limpiar = [ruta_audio]
    if not es_audio and ruta_video_o_audio != ruta_audio:
        archivos_limpiar.append(ruta_video_o_audio)

    st.markdown("---")
    st.markdown(f"### {L('result_title')}")

    st.session_state.ultimo_nombre = nombre_original

    with st.spinner(L("spinner_process")):
        os.makedirs("data", exist_ok=True)

        inestabilidad_fase = 0.0
        porcentaje_ia = 0.0
        fig_mapa_calor = None
        predicciones_por_ventana = []
        tiempos_ventanas = []

        try:
            if not es_audio:
                extraer_audio_de_video(ruta_video_o_audio, ruta_audio)

            if not os.path.exists(ruta_audio):
                raise Exception(L("no_audio_error"))

            y, sr = cargar_y_normalizar_audio(ruta_audio, target_sr=TARGET_SR)
            st.session_state.ultimo_audio_y = y
            st.session_state.ultimo_audio_sr = sr
            mel_db, mfccs, stft_compleja = calcular_stft_y_mel(y, sr)

            inestabilidad_fase = calcular_regularidad_fase(y)
            fig_mapa_calor, warning_msg = generar_mapa_calor_interactivo(mel_db, sr, y_audio=y, stft_compleja=stft_compleja)

            if warning_msg:
                st.warning(warning_msg)

            if modelo_ia is not None:
                porcentaje_ia, predicciones_por_ventana, tiempos_ventanas = predecir_audio_completo(
                    mel_db, modelo_ia, device=torch.device("cpu")
                )
                st.session_state.ultimo_porcentaje_ia = porcentaje_ia

        except Exception as e:
            st.error(L("process_error", msg=e))

        finally:
            for p in archivos_limpiar:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"#### {L('heatmap_title')}")
        if fig_mapa_calor:
            st.plotly_chart(fig_mapa_calor, use_container_width=True)
        else:
            st.info(L("rendering"))

        # Accessible textual summary of spectral findings
        if porcentaje_ia > 0:
            num_anomalies = sum(1 for p in predicciones_por_ventana if (p > 50) == (porcentaje_ia > 50))
            summary = PDF_L("summary_accessible",
                veredicto="DEEPFAKE" if porcentaje_ia > 50 else "REAL",
                confianza=f"{porcentaje_ia:.1f}" if porcentaje_ia > 50 else f"{100 - porcentaje_ia:.1f}",
                inestabilidad=f"{inestabilidad_fase:.4f}",
                anomalias=num_anomalies
            )
            st.caption(summary)

    with col2:
        st.markdown(f"#### {L('metrics_title')}")

        if porcentaje_ia > 85:
            estado_alerta = L("alert_high")
            color_delta = "inverse"
        elif porcentaje_ia > 60:
            estado_alerta = L("alert_warning")
            color_delta = "off"
        else:
            estado_alerta = L("alert_safe")
            color_delta = "normal"

        st.metric(
            label=L("deepfake_prob"),
            value=f"{porcentaje_ia:.1f}%",
            delta=estado_alerta,
            delta_color=color_delta
        )
        st.progress(int(porcentaje_ia) / 100)

        with st.expander(L("tech_expander")):
            st.write(f"**{L('tech_arch')}**")
            st.write(f"**{L('tech_sampling', sr=TARGET_SR)}**")
            st.write(f"**{L('tech_tensors')}**")
            if len(predicciones_por_ventana) > 1:
                st.write(L("tech_windows", count=len(predicciones_por_ventana)))
                for t_ini, prob in zip(tiempos_ventanas, predicciones_por_ventana):
                    st.write(f"  • {L('tech_second', time=t_ini, prob=prob)}")

        st.download_button(
            label=L("download_pdf_btn"),
            data=generar_reporte_forense(
                nombre_original, y, sr, porcentaje_ia,
                inestabilidad_fase=inestabilidad_fase,
                tiempos_ventanas=tiempos_ventanas,
                predicciones_por_ventana=predicciones_por_ventana,
                idioma=st.session_state.df_lang,
            ).read(),
            file_name=f"reporte_forense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        if porcentaje_ia > 0:
            lang = st.session_state.df_lang
            verdict = "DEEPFAKE" if porcentaje_ia > 50 else "REAL"
            conf = f"{porcentaje_ia:.1f}" if porcentaje_ia > 50 else f"{100 - porcentaje_ia:.1f}"
            text = json.dumps(L("speak_result", veredicto=verdict, confianza=conf))
            voice_lang = "es-ES" if lang == "es" else "en-US"
            st.components.v1.html(f"""
            <button onclick="
                var msg = new SpeechSynthesisUtterance({text});
                msg.lang = '{voice_lang}';
                msg.rate = 0.9;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(msg);
            " style="width:100%; padding:8px; border:1px solid #ccc; border-radius:8px; background:#f0f2f6; cursor:pointer; font-size:14px;">
                🔊 {L("listening")}
            </button>
            """, height=50)

        st.divider()

        st.metric(
            label=L("phase_metric"),
            value=f"{inestabilidad_fase:.4f}",
            delta=L("phase_rupture") if inestabilidad_fase > 3.5 else L("phase_stable"),
            delta_color="inverse" if inestabilidad_fase > 3.5 else "normal"
        )

        if modelo_ia is not None and len(predicciones_por_ventana) > 0:
            st.divider()
            st.markdown(f"#### {L('gradcam_title')}")

            idx_max = int(np.argmax(predicciones_por_ventana))
            t_ventana = tiempos_ventanas[idx_max]
            inicio_frame = int(t_ventana * TARGET_SR / HOP_LENGTH)
            ventana_mel = mel_db[:, inicio_frame:inicio_frame + MAX_FRAMES]

            ventana_norm = normalizar_espectrograma(ventana_mel)
            tensor_entrada = convertir_a_tensor_pytorch(ventana_norm, normalizar=False)

            target_class = 1 if porcentaje_ia > 50 else 0
            cam = generar_mapa_gradcam(modelo_ia, tensor_entrada, target_class=target_class)

            buf = _figura_superposicion(ventana_mel, cam, sr=TARGET_SR, hop_length=HOP_LENGTH)
            st.image(buf, use_container_width=True)
            st.caption(L("gradcam_caption", time=t_ventana, prob=predicciones_por_ventana[idx_max]))


tab_upload, tab_link = st.tabs([L("tab_upload"), L("tab_link")])

with tab_upload:
    archivo_video = st.file_uploader(L("upload_label"), type=["mp4", "avi", "mov", "wav", "mp3"])

    if archivo_video is not None:
        if st.session_state.ultimo_archivo != archivo_video.name:
            st.session_state.ultimo_archivo = archivo_video.name

        st.success(f"✅ {L('upload_success')} '{archivo_video.name}'")
        st.video(archivo_video)

        id_unico = str(uuid.uuid4())[:8]
        ruta_temp_video = f"data/temp_video_{id_unico}.mp4"
        ruta_audio = f"data/temp_audio_{id_unico}.wav"

        nombre_archivo = archivo_video.name.lower()
        es_audio = nombre_archivo.endswith(".wav") or nombre_archivo.endswith(".mp3")

        if es_audio:
            with open(ruta_audio, "wb") as f:
                f.write(archivo_video.getbuffer())
            _procesar_y_mostrar(ruta_audio, ruta_audio, archivo_video.name, True)
        else:
            with open(ruta_temp_video, "wb") as f:
                f.write(archivo_video.getbuffer())
            _procesar_y_mostrar(ruta_temp_video, ruta_audio, archivo_video.name, False)

with tab_link:
    url = st.text_input(
        L("link_label"),
        placeholder=L("link_placeholder")
    )

    if url and st.button(L("analyze_btn"), type="primary"):
        try:
            with st.spinner(L("spinner_download")):
                ruta_descargada = descargar_video_de_red_social(url)

            nombre_original = os.path.basename(ruta_descargada)
            st.success(f"✅ {L('download_success')}")
            st.video(ruta_descargada)

            id_unico = str(uuid.uuid4())[:8]
            ruta_audio = f"data/temp_audio_{id_unico}.wav"
            _procesar_y_mostrar(ruta_descargada, ruta_audio, nombre_original, False)

        except ValueError as e:
            st.error(str(e))
        except RuntimeError as e:
            st.error(str(e))
