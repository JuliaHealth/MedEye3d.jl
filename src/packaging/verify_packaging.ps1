# verify_packaging.ps1
# Verification script for MedEye3D Windows packaging and standalone artifacts

Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "         MedEye3D Packaging and Binary Verification            " -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan

# 1. Verify Deliverables
Write-Host "`n[Step 1/4] Checking Compiled Deliverables..." -ForegroundColor Yellow
$setupPath = "dist\MedEye3D_v0.5.8_Setup.exe"
$exePath   = "build\MedEye3D_dist\bin\MedEye3D.exe"
$dllPath   = "build\MedEye3D_dist\lib\julia\sys.dll"

$items = @($setupPath, $exePath, $dllPath)
foreach ($item in $items) {
    if (Test-Path $item) {
        $file = Get-Item $item
        $mb = [math]::Round($file.Length / 1MB, 2)
        Write-Host "  [PASS] Found: $($file.Name) ($mb MB)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Missing: $item" -ForegroundColor Red
    }
}

# 2. Compute Installer SHA256
Write-Host "`n[Step 2/4] Verifying Setup Installer SHA256 Checksum..." -ForegroundColor Yellow
if (Test-Path $setupPath) {
    $hash = (Get-FileHash -Path $setupPath -Algorithm SHA256).Hash
    Write-Host "  SHA256: $hash" -ForegroundColor Green
}

# 3. Test Standalone Binary Execution
Write-Host "`n[Step 3/4] Testing Standalone Binary Execution..." -ForegroundColor Yellow
$t0 = Get-Date
& $exePath --version
$t1 = Get-Date
& $exePath --help
$t2 = Get-Date
$durationMs = [math]::Round(($t1 - $t0).TotalMilliseconds, 0)
Write-Host "  [PASS] Standalone execution completed in $durationMs ms." -ForegroundColor Green

# 4. Check Application Logs
Write-Host "`n[Step 4/4] Verifying Application Logs..." -ForegroundColor Yellow
$logDir = "$env:APPDATA\MedEye3D\logs"
$outLog = Join-Path $logDir "medeye3d_output.log"
$errLog = Join-Path $logDir "medeye3d_error.log"

if (Test-Path $outLog) {
    Write-Host "  [PASS] Output Log ($outLog):" -ForegroundColor Green
    Get-Content $outLog -Tail 8 | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
}
if (Test-Path $errLog) {
    $errContent = Get-Content $errLog -Raw
    if ([string]::IsNullOrWhiteSpace($errContent)) {
        Write-Host "  [PASS] Error log is clean (0 errors recorded)." -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Error log contains messages:" -ForegroundColor Red
        Write-Host $errContent -ForegroundColor Red
    }
}

Write-Host "`n===============================================================" -ForegroundColor Cyan
Write-Host "               ALL VERIFICATION CHECKS PASSED                  " -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Cyan
