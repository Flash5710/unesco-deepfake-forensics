@echo off
rem DeepForensic - Tunel publico de Cloudflare hacia localhost:8501
cd /d "%~dp0"

set "CF=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
if exist "%CF%" goto :found
where cloudflared >nul 2>nul
if not errorlevel 1 (
  set "CF=cloudflared"
  goto :found
)
echo [ERROR] No se encontro cloudflared. Instalalo con:  winget install Cloudflare.cloudflared
pause
exit /b 1

:found
echo Tunel iniciado. Abajo aparece tu URL publica:
echo   https://XXXX.trycloudflare.com   (copia esa URL para tu celular)
echo Tambien queda guardada en tunel.log
echo.
echo Requiere que Streamlit ya este corriendo en localhost:8501
echo (Deja esta ventana abierta. Para detener el tunel, cierrala o Ctrl+C)
echo.
"%CF%" tunnel --url http://localhost:8501 --no-autoupdate 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command "$input | Tee-Object -FilePath tunel.log"
pause
