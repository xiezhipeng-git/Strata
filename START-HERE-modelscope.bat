@echo off
rem Strata for Windows: the first run installs everything and starts the model; later runs just start it.
rem Needs only an NVIDIA or AMD graphics driver. Python is installed for your user account if it is missing (no admin needed).
setlocal
title Strata
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run

call :findpy
if defined PY goto venv
echo.
echo  Python 3.10 or newer is not installed. Installing Python 3.12 for your user account ...
where winget >nul 2>nul
if errorlevel 1 goto pyorg
winget install -e --id Python.Python.3.12 --scope user --silent --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
call :findpy
if defined PY goto venv
:pyorg
echo  Downloading the Python installer from python.org ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe -OutFile \"$env:TEMP\strata-python-setup.exe\""
if exist "%TEMP%\strata-python-setup.exe" "%TEMP%\strata-python-setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
call :findpy
if defined PY goto venv
echo.
echo  Python could not be installed automatically.
echo  Install 64-bit Python 3.12 from https://www.python.org/downloads/ ("Add python.exe to PATH"),
echo  then double-click START-HERE.bat again.
pause
exit /b 1

:venv
rem a private environment inside this folder, so nothing is installed into the system Python
%PY% -m venv .venv
if exist ".venv\Scripts\python.exe" goto run
echo  Could not create the Python environment in .venv
pause
exit /b 1

:run
rem --- Model download via ModelScope (domestic mirror; HuggingFace is unreachable here) ---
rem Strata's setup.py only fetches models from huggingface.co, which fails on this PC
rem (TLS/SSL -> 502). We pull the exact Swift 1.5 IQ3_XXS GGUF shards + vision encoder
rem from ModelScope into the STANDARD data folder OUTSIDE the repo (<repo>\..\Strata-data),
rem so setup.py's own "move the data folder out of the repo" step is a no-op and --gguf-dir
rem points at the real files. Files already present are skipped (resumable).
rem The system Python 3.13 (py -3.13) has the modelscope SDK; the .venv does not.
set "STRATA_MODELS_DIR=%~dp0..\Strata-data\models\swift-IQ3_XXS"
rem The MTP draft layer (~5 GB, from the original Qwen checkpoint) is fetched by setup.py's
rem mtp_fetch.py: it is hardcoded to huggingface.co but DOES support the HF_ENDPOINT mirror
rem variable. ModelScope cannot serve it (not HF-protocol-compatible), so we point it at the
rem HF-compatible mirror hf-mirror.com. The model GGUFs themselves still come from ModelScope
rem via --gguf-dir above (no HuggingFace involved for those).
set "HF_ENDPOINT=https://hf-mirror.com"
py -3.13 "%~dp0download_swift_model.py"
if errorlevel 1 (
  echo.
  echo  ModelScope download failed. You can download the 3 GGUF files manually into
  echo  "%~dp0..\Strata-data\models\swift-IQ3_XXS" and re-run.
  pause
  exit /b 1
)
rem --- Run setup.py NON-INTERACTIVELY: pass the choices + point it at the local GGUFs ---
rem --family swift --model IQ3_XXS = Swift 1.5 IQ3_XXS; --context 131072 = 128K; --kv int8 = 8-bit KV;
rem --vision yes = images on; --yes = accept recommended answers; --gguf-dir = use local files (no HF download).
".venv\Scripts\python.exe" setup.py --family swift --model IQ3_XXS --context 131072 --kv int8 --vision yes --gguf-dir "%~dp0..\Strata-data\models\swift-IQ3_XXS" --yes %*
if errorlevel 1 pause
exit /b

:findpy
rem the py launcher first, then python on PATH (not the Microsoft Store stub), then the usual per-user folders
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3" & goto :eof
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python" & goto :eof
for %%V in (313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"" & goto :eof
goto :eof
