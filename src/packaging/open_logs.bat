@echo off
set "LOG_DIR=%APPDATA%\MedEye3D\logs"
if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%" 2>nul
)
explorer.exe "%LOG_DIR%"
