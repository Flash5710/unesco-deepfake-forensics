@echo off
rem DeepForensic - Arrancar solo Streamlit (uso local, como siempre)
cd /d "%~dp0"

if not exist "venv_cuda\Scripts\streamlit.exe" (
  echo [ERROR] No se encontro venv_cuda\Scripts\streamlit.exe
  pause
  exit /b 1
)

echo Iniciando Streamlit en http://localhost:8501 ...
echo (Deja esta ventana abierta. Para detener, cierrala o Ctrl+C)
venv_cuda\Scripts\streamlit.exe run app.py --server.headless true --server.port 8501
pause
