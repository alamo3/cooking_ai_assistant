@echo off
rem Double-click to start the kitchen assistant. Extra arguments go to start.ps1 (e.g. -NoVoice -Open).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 pause
