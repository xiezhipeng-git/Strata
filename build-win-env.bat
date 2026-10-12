@echo off
REM ============================================================
REM  build-win-env.bat -- set up the MSVC environment WITHOUT vcvars64.bat
REM
REM  Why not vcvars64.bat: it shells out to reg.exe to find the install
REM  paths, which sandboxes and locked-down machines block. This script
REM  reads the same two directories off the filesystem instead (MSVC's
REM  versioned tools dir and the Windows SDK), then exports INCLUDE / LIB.
REM
REM  Usage:  call build-win-env.bat     [environment only, then return]
REM          or:    build-win-env.bat    [environment, then stay in cmd]
REM
REM  Keep this file ASCII-only: cmd.exe reads .bat in the OEM codepage.
REM ============================================================
setlocal EnableExtensions EnableDelayedExpansion

set "VSPATH="
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VSPATH=%%i"
if not defined VSPATH (
  if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools" set "VSPATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
)
if not defined VSPATH (
  echo [ERROR] Visual Studio with the C++ workload was not found.
  exit /b 1
)

REM ---- MSVC tools dir = the newest version under VC\Tools\MSVC ----
set "MSVCROOT=%VSPATH%\VC\Tools\MSVC"
set "MSVC="
if exist "%MSVCROOT%" for /d %%d in ("%MSVCROOT%\*.*") do set "MSVC=%%~fd"
if not defined MSVC (
  echo [ERROR] no versioned MSVC tools dir under "%MSVCROOT%"
  exit /b 1
)

REM ---- Windows SDK = the newest 10.0.xxxxx that has all three subdirs ----
set "SDKROOT=%ProgramFiles(x86)%\Windows Kits\10"
set "SDKVER="
if exist "%SDKROOT%\Include" for /d %%d in ("%SDKROOT%\Include\10.0.*") do (
  if exist "%%~fd\ucrt" if exist "%%~fd\um" if exist "%SDKROOT%\Lib\%%~nxd\ucrt" set "SDKVER=%%~nxd"
)
if not defined SDKVER (
  echo [ERROR] no usable Windows SDK under "%SDKROOT%"
  exit /b 1
)

set "INCLUDE=%MSVC%\include;%SDKROOT%\Include\%SDKVER%\ucrt;%SDKROOT%\Include\%SDKVER%\um;%SDKROOT%\Include\%SDKVER%\shared;%SDKROOT%\Include\%SDKVER%\cppwinrt;%INCLUDE%"
set "LIB=%MSVC%\lib\x64;%SDKROOT%\Lib\%SDKVER%\ucrt\x64;%SDKROOT%\Lib\%SDKVER%\um\x64;%LIB%"

if exist "%MSVC%\bin\Hostx64\x64\cl.exe" set "PATH=%MSVC%\bin\Hostx64\x64;%PATH%"
if exist "%SDKROOT%\bin\%SDKVER%\x64" set "PATH=%SDKROOT%\bin\%SDKVER%\x64;%PATH%"

echo [ok] MSVC  : !MSVC!
echo [ok] SDK   : !SDKVER!
echo [ok] cl.exe on PATH

REM endlocal but keep the env we just built
endlocal & set "INCLUDE=%INCLUDE%" & set "LIB=%LIB%" & set "PATH=%PATH%" & set "MSVC_TOOLS=%MSVC%"
exit /b 0
