@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Strata - build strata-lp (engine + per-request logprobs, for AnyJev)

REM ============================================================
REM  Builds `build-lp\local\logprobs\build\strata-lp.exe`
REM  = the Strata engine + per-request logprobs (what AnyJev needs).
REM
REM  It is a SEPARATE binary. The stock `engine\strata.exe` is never
REM  touched, and no file under `src\` is ever edited: the patched
REM  source is generated as a copy by
REM      python local\logprobs\build_engine.py --emit
REM  (that script has a sha256 tripwire and refuses to write if the
REM   original file already carries the patch).
REM
REM  Requirements (the script checks both and prints the exact fix):
REM    1. Visual Studio 2022 Build Tools with the C++ workload.
REM       This also supplies the cmake + ninja it uses -- nothing else
REM       to install for the build system.
REM    2. CUDA Toolkit for Windows, i.e. `nvcc.exe`.
REM       NOTE: on this PC the CUDA folders under
REM       "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"
REM       are empty shells (only `extras\`) -- they are NOT a toolkit.
REM
REM  Keep this file ASCII-only: cmd.exe reads .bat in the OEM codepage,
REM  so non-ASCII text gets garbled and can break REM/command parsing.
REM ============================================================

cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "BUILD=%ROOT%\build-lp"

echo ============================================================
echo   Strata : build strata-lp.exe  (logprobs patched engine)
echo   repo   : %ROOT%
echo   build  : %BUILD%
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 1. Python (the patch generator needs it). Prefer the repo venv.
REM ------------------------------------------------------------
set "PY="
if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not defined PY (
  py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
)
if not defined PY (
  python -c "import sys" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] no Python found. Run START-HERE-modelscope.bat once ^(it creates .venv^).
  goto :fail
)
echo [ok] python : %PY%

REM ------------------------------------------------------------
REM 2. Visual Studio 2022 Build Tools (vswhere ships with the VS installer)
REM ------------------------------------------------------------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSPATH="
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
)
if not defined VSPATH (
  echo.
  echo [ERROR] Visual Studio 2022 Build Tools with the C++ workload was not found.
  echo         Install it with:
  echo.
  echo   "%%ProgramFiles(x86)%%\Microsoft Visual Studio\Installer\setup.exe" ^
  echo       install --quiet --wait --norestart --nocache ^
  echo       --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
  echo.
  echo         ^(that installer already exists on this PC; it needs admin once^)
  goto :fail
)
echo [ok] vs     : %VSPATH%

set "CMAKE=%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not exist "%CMAKE%" (
  echo [ERROR] cmake.exe not found under the VS install. Add the component
  echo         "C++ CMake tools for Windows" to the Build Tools install.
  goto :fail
)
if not exist "%NINJA%" (
  echo [ERROR] ninja.exe not found under the VS install. Add the component
  echo         "C++ CMake tools for Windows" to the Build Tools install.
  goto :fail
)
echo [ok] cmake  : %CMAKE%
echo [ok] ninja  : %NINJA%

REM ------------------------------------------------------------
REM 3. CUDA toolkit (nvcc). Look for the newest usable one.
REM ------------------------------------------------------------
set "NVCC="
for %%V in (13.3 13.2 13.1 13.0 12.9 12.8 12.6 12.5 12.4 12.1 11.8) do (
  if not defined NVCC if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%%V\bin\nvcc.exe" set "NVCC=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%%V\bin\nvcc.exe"
)
if not defined NVCC (
  echo.
  echo [ERROR] nvcc.exe not found -- the CUDA Toolkit for Windows is not installed.
  echo         The CUDA folders under
  echo           "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\"
  echo         exist but are empty shells, so they cannot be used.
  echo.
  echo         Install the toolkit ^(12.6 or newer; the release engine was built
  echo         with 13.0^) from:
  echo           https://developer.nvidia.com/cuda-downloads
  echo         Pick "Windows / x86_64 / exe (local)" and install. ~3 GB.
  echo         Then re-run this script.
  goto :fail
)
echo [ok] nvcc   : %NVCC%

REM ------------------------------------------------------------
REM 4. Generate the patched source copy (never edits src\)
REM ------------------------------------------------------------
echo.
echo ---- [1/4] generating the patched source ----
%PY% "%ROOT%\local\logprobs\build_engine.py" --emit
if errorlevel 1 goto :fail

REM ------------------------------------------------------------
REM 5. Make sure the root CMakeLists enables the target (idempotent)
REM ------------------------------------------------------------
echo.
echo ---- [2/4] enabling the strata-lp target in CMakeLists.txt ----
findstr /c:"add_subdirectory(local/logprobs/build EXCLUDE_FROM_ALL)" "%ROOT%\CMakeLists.txt" >nul 2>&1
if errorlevel 1 (
  echo.>> "%ROOT%\CMakeLists.txt"
  echo # [local/logprobs] strata-lp: engine + per-request logprobs. One line, delete to revert.>> "%ROOT%\CMakeLists.txt"
  echo # Must stay at the END: the target inherits _strata_gpu_include_directories />> "%ROOT%\CMakeLists.txt"
  echo # strata_engine / strata_prefill / strata_spec / _strata_gpu_runtime_target.>> "%ROOT%\CMakeLists.txt"
  echo add_subdirectory(local/logprobs/build EXCLUDE_FROM_ALL)>> "%ROOT%\CMakeLists.txt"
  echo   appended the add_subdirectory line
) else (
  echo   already present, nothing to do
)

REM ------------------------------------------------------------
REM 6. Configure
REM ------------------------------------------------------------
echo.
echo ---- [3/4] cmake configure ----
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [ERROR] vcvars64.bat failed
  goto :fail
)
"%CMAKE%" -S "%ROOT%" -B "%BUILD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_CUDA_COMPILER="%NVCC%" ^
  -DSTRATA_GGML_DIR="%ROOT%\third_party\llama.cpp"
if errorlevel 1 goto :fail

REM ------------------------------------------------------------
REM 7. Build
REM ------------------------------------------------------------
echo.
echo ---- [4/4] cmake build (this takes a while) ----
"%CMAKE%" --build "%BUILD%" --target strata-lp
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo   DONE
echo   binary : %BUILD%\local\logprobs\build\strata-lp.exe
echo.
echo   Start it with the logprobs-aware server:
echo     .venv\Scripts\python.exe local\logprobs\server_logprobs.py ^
echo         --config strata-swift-iq3_xxs.json ^
echo         --engine "%BUILD%\local\logprobs\build\strata-lp.exe" --port 8080
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [FAILED] see the message above.
pause
exit /b 1
