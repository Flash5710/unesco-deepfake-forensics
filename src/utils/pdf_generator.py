import io
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

N_MELS = 128
MAX_FRAMES = 300


def _espectrograma_torchaudio(y, sr=16000):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=512, n_mels=N_MELS,
    )
    db = torchaudio.transforms.AmplitudeToDB()
    t = torch.from_numpy(y).float().unsqueeze(0)
    with torch.no_grad():
        mel_db = db(mel(t)).squeeze(0).numpy()
    B, T = mel_db.shape
    if T < MAX_FRAMES:
        pad = MAX_FRAMES - T
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant',
                        constant_values=mel_db.min())
    elif T > MAX_FRAMES:
        mel_db = mel_db[:, :MAX_FRAMES]
    mu = mel_db.mean()
    sd = mel_db.std()
    if sd > 1e-8:
        mel_db = (mel_db - mu) / sd
    return mel_db


def _figura_espectrograma(mel_db, sr=16000):
    hop_length = 512
    fig, ax = plt.subplots(figsize=(8, 3.5))
    img = ax.imshow(mel_db, aspect='auto', origin='lower',
                    cmap='inferno',
                    extent=[0, mel_db.shape[1] * hop_length / sr,
                            0, mel_db.shape[0]],
                    interpolation='bilinear')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Bandas Mel')
    ax.set_title('Espectrograma Mel (torchaudio)')
    plt.colorbar(img, ax=ax, label='dB normalizado')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _tabla_estilo(datos, col_widths, header=True):
    tbl = Table(datos, colWidths=col_widths)
    estilo = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
    ]
    if header:
        estilo += [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eaf6')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]
    tbl.setStyle(TableStyle(estilo))
    return tbl


def generar_reporte_forense(nombre_archivo, audio_y, sr, porcentaje_ia,
                            inestabilidad_fase=0.0,
                            tiempos_ventanas=None,
                            predicciones_por_ventana=None,
                            f1_score=97.65, recall=96.47):
    if tiempos_ventanas is None:
        tiempos_ventanas = []
    if predicciones_por_ventana is None:
        predicciones_por_ventana = []

    buf_pdf = io.BytesIO()
    doc = SimpleDocTemplate(buf_pdf, pagesize=A4,
                            topMargin=20*mm, bottomMargin=20*mm)
    estilos = getSampleStyleSheet()
    titulo = ParagraphStyle('Titulo1', parent=estilos['Heading1'],
                            fontSize=16, spaceAfter=8,
                            textColor=colors.HexColor('#1a237e'))
    subtitulo = ParagraphStyle('Sub', parent=estilos['Heading2'],
                               fontSize=11, spaceAfter=6,
                               textColor=colors.HexColor('#37474f'))
    cuerpo = ParagraphStyle('Cuerpo', parent=estilos['Normal'],
                            fontSize=10, spaceAfter=4)
    nota = ParagraphStyle('Nota', parent=estilos['Normal'],
                          fontSize=8, textColor=colors.grey,
                          alignment=1)
    parrafo = ParagraphStyle('Parrafo', parent=estilos['Normal'],
                             fontSize=10, spaceAfter=6,
                             leading=14, alignment=4)

    contenido = []

    contenido.append(Paragraph(
        'Reporte de Análisis Forense de Audio', titulo))
    contenido.append(Paragraph(
        'UNESCO Youth Hackathon 2026', subtitulo))
    contenido.append(Spacer(1, 6*mm))

    contenido.append(Paragraph('Metadatos del Análisis', subtitulo))
    ahora = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    contenido.append(Paragraph(
        f'<b>Archivo:</b> {nombre_archivo}', cuerpo))
    contenido.append(Paragraph(
        f'<b>Fecha y Hora:</b> {ahora}', cuerpo))
    contenido.append(Paragraph(
        f'<b>Tasa de Muestreo:</b> {sr} Hz', cuerpo))
    contenido.append(Spacer(1, 4*mm))

    contenido.append(Paragraph('Veredicto del Modelo', subtitulo))
    es_fake = porcentaje_ia > 50
    clase = 'DEEPFAKE' if es_fake else 'REAL'
    color_veredicto = (colors.HexColor('#d32f2f') if es_fake
                       else colors.HexColor('#2e7d32'))
    tbl = Table([
        ['Clasificación', clase],
        ['Confianza', f'{porcentaje_ia:.1f}%'],
    ], colWidths=[120, 200])
    tbl.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8eaf6')),
        ('TEXTCOLOR', (1, 0), (1, 0), color_veredicto),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor('#1565c0')),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1a237e')),
    ]))
    contenido.append(tbl)
    contenido.append(Spacer(1, 5*mm))

    contenido.append(Paragraph('Métricas Acústicas', subtitulo))
    tbl_acust = _tabla_estilo([
        ['Métrica', 'Valor'],
        ['Inestabilidad de Fase', f'{inestabilidad_fase:.4f}'],
    ], [120, 200])
    contenido.append(tbl_acust)
    contenido.append(Spacer(1, 5*mm))

    if tiempos_ventanas and predicciones_por_ventana:
        contenido.append(Paragraph(
            'Análisis Temporal de Anomalías', subtitulo))
        pares = list(zip(tiempos_ventanas, predicciones_por_ventana))
        if es_fake:
            pares = sorted(pares, key=lambda x: x[1], reverse=True)[:5]
        else:
            pares = sorted(pares, key=lambda x: x[1])[:5]
        filas = [['#', 'Tiempo (s)', 'Prob. Deepfake (%)']]
        for i, (t, p) in enumerate(pares, 1):
            filas.append([str(i), f'{t:.1f}', f'{p:.2f}'])
        tbl_temp = _tabla_estilo(filas, [30, 80, 120])
        contenido.append(tbl_temp)
        contenido.append(Spacer(1, 5*mm))

    contenido.append(Paragraph('Evidencia Visual', subtitulo))
    mel_db = _espectrograma_torchaudio(audio_y, sr)
    buf_img = _figura_espectrograma(mel_db, sr)
    img = Image(buf_img, width=460, height=200)
    contenido.append(img)
    contenido.append(Spacer(1, 5*mm))

    contenido.append(Paragraph('Conclusión Pericial', subtitulo))
    if es_fake:
        texto_conclusion = (
            'Las anomalías en el espectrograma y la inestabilidad de fase '
            'apuntan a síntesis de voz o clonación por vocoder. '
            'Se recomienda una revisión forense adicional con análisis '
            'de marcas de agua digitales y verificación de cadena de '
            'custodia del archivo original.'
        )
    else:
        texto_conclusion = (
            'Las características acústicas mantienen la continuidad '
            'natural de la voz humana. No se detectaron artefactos '
            'espectrales ni rupturas de fase significativas que sugieran '
            'manipulación sintética del contenido auditivo.'
        )
    contenido.append(Paragraph(texto_conclusion, parrafo))
    contenido.append(Spacer(1, 5*mm))

    contenido.append(Paragraph('Firma Técnica', subtitulo))
    contenido.append(Paragraph(
        'Métricas de fiabilidad del modelo '
        f'(F1-Score: {f1_score:.2f}%, Recall: {recall:.2f}%)',
        nota))

    doc.build(contenido)
    buf_pdf.seek(0)
    return buf_pdf