@echo off
title MedEye3D Live Log Monitor
:: Determine log directory from log_config.json or fall back to default
set "LOG_DIR=%APPDATA%\MedEye3D\logs"
set "CONFIG_FILE=%~dp0..\data\log_config.json"
if exist "%CONFIG_FILE%" (
    for /f "tokens=2 delims=:}" %%a in ('type "%CONFIG_FILE%" ^| findstr /i "log_dir"') do (
        set "CUSTOM_DIR=%%~a"
    )
    if defined CUSTOM_DIR (
        for /f "tokens=* delims= " %%b in ("%CUSTOM_DIR%") do set "CUSTOM_DIR=%%~b"
        if not "%CUSTOM_DIR%"=="" set "LOG_DIR=%CUSTOM_DIR%"
    )
)
set "LOG_FILE=%LOG_DIR%\medeye3d_output.log"
powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Write-Host '===================================================' -ForegroundColor Cyan; Write-Host '           MedEye3D Live Log Monitor               ' -ForegroundColor Cyan; Write-Host '===================================================' -ForegroundColor Cyan; Write-Host 'Watching log file: ' -NoNewline; Write-Host '%LOG_FILE%' -ForegroundColor Yellow; Write-Host 'Press Ctrl+C to stop monitoring.' -ForegroundColor Gray; Write-Host ''; if (Test-Path '%LOG_FILE%') { Get-Content -Path '%LOG_FILE%' -Wait -Tail 40 } else { Write-Host 'Waiting for MedEye3D to launch and create logs...' -ForegroundColor Yellow; while (-not (Test-Path '%LOG_FILE%')) { Start-Sleep -Milliseconds 500 }; Get-Content -Path '%LOG_FILE%' -Wait -Tail 40 }"
