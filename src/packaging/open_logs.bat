@echo off
:: Determine log directory from log_config.json or fall back to default
set "LOG_DIR=%APPDATA%\MedEye3D\logs"
set "CONFIG_FILE=%~dp0..\data\log_config.json"
if exist "%CONFIG_FILE%" (
    for /f "tokens=2 delims=:}" %%a in ('type "%CONFIG_FILE%" ^| findstr /i "log_dir"') do (
        set "CUSTOM_DIR=%%~a"
    )
    if defined CUSTOM_DIR (
        :: Trim leading/trailing spaces and quotes
        for /f "tokens=* delims= " %%b in ("%CUSTOM_DIR%") do set "CUSTOM_DIR=%%~b"
        if not "%CUSTOM_DIR%"=="" set "LOG_DIR=%CUSTOM_DIR%"
    )
)
if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%" 2>nul
)
explorer.exe "%LOG_DIR%"
