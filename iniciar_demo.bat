@echo off
rem DeepForensic - DEMO MOVIL: arranca Streamlit + tunel Cloudflare en un clic
setlocal
cd /d "%~dp0"

rem Verificaciones previas
if not exist "venv_cuda\Scripts\streamlit.exe" (
  echo [ERROR] No se encontro venv_cuda\Scripts\streamlit.exe
  pause
  exit /b 1
)
set "CF=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
if not exist "%CF%" (
  where cloudflared >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] No se encontro cloudflared. Instalalo con:  winget install Cloudflare.cloudflared
    pause
    exit /b 1
  )
)

echo [1/3] Abriendo Streamlit en su propia ventana...
start "DeepForensic - Streamlit" cmd /k "%~dp0iniciar_app.bat"

echo [2/3] Esperando a que Streamlit responda en localhost:8501 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0;$i -lt 60;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing 'http://localhost:8501/_stcore/health' -TimeoutSec 2; if($r.Content -eq 'ok'){ $ok=$true; break } } catch {}; Start-Sleep -Seconds 1 }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
  echo [ERROR] Streamlit no respondio. Revisa la ventana DeepForensic - Streamlit.
  pause
  exit /b 1
)
echo       Streamlit OK.

echo [3/3] Abriendo el tunel publico de Cloudflare en su propia ventana...
start "DeepForensic - Tunel (tu URL)" cmd /k "%~dp0iniciar_tunel.bat"

echo.
echo =====================================================================
echo   LISTO. En la ventana  DeepForensic - Tunel  aparecera tu URL:
echo     https://XXXXXXX.trycloudflare.com
echo   Abre esa URL desde tu celular (puede tardar unos segundos en
echo   cargar la primera vez).
echo   Para detener todo, cierra las dos ventanas nuevas.
echo =====================================================================
echo.
pause
