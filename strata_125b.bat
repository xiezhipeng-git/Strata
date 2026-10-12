@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Strata 125B - Swift 1.5 IQ3_XXS (ModelScope)

cd /d "%~dp0"

REM ==========================================================
REM  Guard: do NOT start a second Strata 125B on the same port.
REM  Two instances would share one GPU and roughly halve
REM  throughput, exactly like two vLLM instances on one RTX.
REM  ASCII-only comments: cmd.exe reads .bat in the OEM codepage,
REM  so non-ASCII text gets garbled and can break parsing.
REM  Fails open: if curl is missing we just proceed.
REM ==========================================================
set "PORT=8080"
curl -s -m 3 --noproxy "*" "http://127.0.0.1:%PORT%/v1/models" 2>nul | findstr /c:"swift-1.5-iq3_xxs" >nul 2>&1
if not errorlevel 1 (
  echo.
  echo [SKIP] Strata 125B is already serving on http://127.0.0.1:%PORT%/v1
  echo        Close that window / stop it first if you want to restart.
  echo.
  pause
  exit /b 0
)

REM ==========================================================
REM  Step 1: ensure Python 3.10+ and a private .venv.
REM  Mirrors START-HERE-modelscope.bat. The .venv is created
REM  inside this folder so nothing touches the system Python.
REM ==========================================================
if exist ".venv\Scripts\python.exe" goto venv
call :findpy
if defined PY goto mkenv
echo Python 3.10 or newer is not installed. Installing Python 3.12 ...
where winget >nul 2>nul
if errorlevel 1 goto pyorg
winget install -e --id Python.Python.3.12 --scope user --silent --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
call :findpy
if defined PY goto mkenv
:pyorg
echo Downloading the Python installer from python.org ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe -OutFile \"$env:TEMP\strata-python-setup.exe\""
if exist "%TEMP%\strata-python-setup.exe" "%TEMP%\strata-python-setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
call :findpy
if defined PY goto mkenv
echo Python could not be installed automatically. Install 64-bit Python 3.12 from https://www.python.org/downloads and re-run.
pause
exit /b 1

:mkenv
%PY% -m venv .venv
if exist ".venv\Scripts\python.exe" goto venv
echo Could not create the Python environment in .venv
pause
exit /b 1

:venv
REM ==========================================================
REM  Step 2: download the 125B (Swift 1.5 IQ3_XXS) GGUF + vision
REM  encoder from ModelScope. HuggingFace is unreachable here
REM  (TLS/SSL -> 502), so we pull from the domestic mirror into
REM  the STANDARD data folder OUTSIDE the repo (../Strata-data),
REM  then point setup.py at those local files with --gguf-dir.
REM  Skipped if the GGUF is already present (resumable).
REM  The system Python 3.13 (py -3.13) has the modelscope SDK;
REM  the .venv does not, so the download runs on py -3.13.
REM ==========================================================
set "STRATA_MODELS_DIR=%~dp0..\Strata-data\models\swift-IQ3_XXS"
set "GGUF=%STRATA_MODELS_DIR%\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
if exist "%GGUF%" (
  echo [OK] 125B GGUF already present, skipping ModelScope download
) else (
  echo Downloading 125B model from ModelScope into %STRATA_MODELS_DIR% ...
  set "HF_ENDPOINT=https://hf-mirror.com"
  py -3.13 "%~dp0download_swift_model.py"
  if errorlevel 1 (
    echo.
    echo ModelScope download failed. You can download the GGUF files manually into
    echo %STRATA_MODELS_DIR% and re-run.
    pause
    exit /b 1
  )
)

REM ==========================================================
REM  Step 3: run setup.py non-interactively. This installs the
REM  engine, points at the local GGUFs, and STARTS the model.
REM  --family swift --model IQ3_XXS = Swift 1.5 IQ3_XXS (125.7B MoE)
REM  --context 131072 = 128K  --kv int8 = 8-bit KV  --vision yes
REM  --gguf-dir = use local files (no HuggingFace download).
REM  The MTP draft layer is fetched by setup.py via the HF mirror.
REM ==========================================================
echo.
echo ==========================================================
echo   Strata 125B : Swift 1.5 IQ3_XXS  (125.7B MoE)
echo   endpoint    : http://127.0.0.1:%PORT%/v1   (OpenAI-compatible)
echo   model dir   : %STRATA_MODELS_DIR%
echo ----------------------------------------------------------
echo   First start verifies/installs the engine; later starts
echo   are faster. Close this window to stop the server.
echo ==========================================================
echo.

REM 2026-10-13: **默认启动「带 logprobs 的引擎」**（build-lp\...\strata-lp.exe）。
REM   start_server.py 做三件事：跑 setup.py --no-start（只装不启）→ 派生一份 exe 指向
REM   strata-lp.exe 的配置 → 用 local\logprobs\server_logprobs.py 启动。
REM   没编出 strata-lp.exe 时**自动退回原版引擎**，所以这个启动永远不会因此失败。
REM   传 --stock 强制原版；--setup / --check / --no-start 等则完全交给 setup.py 自己收尾。
".venv\Scripts\python.exe" local\logprobs\start_server.py --family swift --model IQ3_XXS --context 131072 --kv int8 --vision yes --gguf-dir "%STRATA_MODELS_DIR%" --yes %*
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
