@echo off
rem Strata training on this machine: WSL2 only.
rem
rem Why not Windows: the training stack is Triton (NF4 fused kernels) + fla + the
rem kernels package, none of which have a Windows build -- the WSL2 numbers in
rem BENCH-SFT-VS-MEZO.md are the only ones that exist. Inference is the opposite:
rem the engine binary (build-lp\strata.exe) is Windows-native, so serving stays on
rem Windows and this window is only about training.
rem
rem Both use the same weights: train.py reads the model pinned in
rem launcher\launch-choice.json -- the one Strata CN.lnk starts. Override with --gguf.
rem
rem Do not run this and the server at the same time: one 48 GiB card, and a
rem T=2048 training step wants ~38 GiB of it.
rem
rem Usage: double-click (train.py defaults), or from a shell:
rem   START-TRAIN.bat --algo mezo --steps 20
rem   START-TRAIN.bat --data jsonl:/root/my.jsonl --out /root/lora_me.jsonl.pt
rem   START-TRAIN.bat --list          (algorithms / objectives / datasets)
setlocal
title Strata training (WSL2)
chcp 65001 >nul

rem --cd takes an absolute Windows path and converts it itself, so this works
rem from any checkout without guessing the drive letter or case.
echo Training in WSL2 -- %~dp0examples\training
echo (models, pack and datasets default to what train.py picks; see --list / --dry-run)
echo.

wsl.exe --cd "%~dp0examples\training" -- python3 train.py %*
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" echo [exit %RC%]
rem keep the window open on a double-click; harmless from a shell too
pause
exit /b %RC%
