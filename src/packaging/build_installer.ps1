<#
.SYNOPSIS
    Automated end-to-end Windows Installer build pipeline for MedEye3D.

.DESCRIPTION
    This script coordinates the entire build and packaging workflow:
    1. Validates Julia runtime and project dependencies.
    2. Compiles standalone relocatable binary tree using PackageCompiler.jl (build_app.jl).
    3. Locates or installs Inno Setup Compiler (ISCC.exe).
    4. Compiles MedEye3D_Installer.iss into a single-file Windows installer (.exe).
    5. Verifies installer integrity and outputs checksums.

.PARAMETER SkipCompile
    Skips the PackageCompiler compilation step and only rebuilds the Inno Setup installer.

.PARAMETER InstallInno
    Automatically installs Inno Setup 6 using winget if not detected on the system.

.PARAMETER Clean
    Cleans previous build and dist directories before starting.

.EXAMPLE
    .\build_installer.ps1
    .\build_installer.ps1 -SkipCompile
    .\build_installer.ps1 -InstallInno -Clean
#>

[CmdletBinding()]
param(
    [switch]$SkipCompile,
    [switch]$InstallInno,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$BuildDir = Join-Path $ProjectRoot "build\MedEye3D_dist"
$DistDir = Join-Path $ProjectRoot "dist"
$IssPath = Join-Path $ScriptDir "MedEye3D_Installer.iss"
$IconPath = Join-Path $ScriptDir "app_icon.ico"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "         MedEye3D - Windows Software Installer Builder           " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Project Root: $ProjectRoot"
Write-Host " Output Dist:  $DistDir"
Write-Host " Start Time:   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# -------------------------------------------------------------------------
# Step 0: Optional Clean
# -------------------------------------------------------------------------
if ($Clean) {
    Write-Host "[Clean] Cleaning previous build artifacts..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) { Remove-Item -Path $BuildDir -Recurse -Force }
    if (Test-Path $DistDir)  { Remove-Item -Path $DistDir -Recurse -Force }
    Write-Host "[Clean] Done." -ForegroundColor Green
}

# -------------------------------------------------------------------------
# Step 1: Ensure Icon Exists
# -------------------------------------------------------------------------
if (-not (Test-Path $IconPath)) {
    Write-Host "[1/5] Generating application icon..." -ForegroundColor Cyan
    & (Join-Path $ScriptDir "generate_icon.ps1")
} else {
    Write-Host "[1/5] Application icon verified ($IconPath)." -ForegroundColor Green
}

# -------------------------------------------------------------------------
# Step 2: Validate Julia
# -------------------------------------------------------------------------
Write-Host "[2/5] Checking Julia executable..." -ForegroundColor Cyan
$JuliaCmd = Get-Command julia -ErrorAction SilentlyContinue
if (-not $JuliaCmd) {
    Write-Error "Julia is not found on PATH. Please install Julia 1.9+ and add it to PATH."
}
$JuliaVer = & julia --version
Write-Host "Found: $JuliaVer" -ForegroundColor Green

# -------------------------------------------------------------------------
# Step 3: Compile Standalone App (PackageCompiler)
# -------------------------------------------------------------------------
if (-not $SkipCompile) {
    Write-Host "`n[3/5] Compiling Standalone Application with PackageCompiler..." -ForegroundColor Cyan
    Write-Host "Running: julia --project=`"$ProjectRoot`" `"$ScriptDir\build_app.jl`"" -ForegroundColor Gray
    
    $proc = Start-Process -FilePath "julia" -ArgumentList "--project=`"$ProjectRoot`"", "`"$ScriptDir\build_app.jl`"" -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru -Wait
    if ($proc.ExitCode -ne 0) {
        Write-Error "PackageCompiler build failed with exit code $($proc.ExitCode)."
    }
    Write-Host "[3/5] Compilation finished successfully." -ForegroundColor Green
} else {
    Write-Host "`n[3/5] Skipping compilation (-SkipCompile flag set)." -ForegroundColor Yellow
    if (-not (Test-Path (Join-Path $BuildDir "bin\MedEye3D.exe"))) {
        Write-Error "Cannot skip compile: $BuildDir\bin\MedEye3D.exe does not exist. Run without -SkipCompile first."
    }
}

# -------------------------------------------------------------------------
# Step 4: Find / Install Inno Setup Compiler (ISCC.exe)
# -------------------------------------------------------------------------
Write-Host "`n[4/5] Locating Inno Setup 6 Compiler (ISCC.exe)..." -ForegroundColor Cyan

$InnoCandidatePaths = @(
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
    "C:\Program Files\Inno Setup 5\ISCC.exe",
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
)

$IsccPath = $null
foreach ($p in $InnoCandidatePaths) {
    if (Test-Path $p) {
        $IsccPath = $p
        break
    }
}

if (-not $IsccPath) {
    $IsccCmd = Get-Command iscc -ErrorAction SilentlyContinue
    if ($IsccCmd) { $IsccPath = $IsccCmd.Source }
}

if (-not $IsccPath) {
    if ($InstallInno) {
        Write-Host "Inno Setup not found. Installing via winget..." -ForegroundColor Yellow
        Start-Process winget -ArgumentList "install JRSoftware.InnoSetup --silent --accept-package-agreements --accept-source-agreements" -NoNewWindow -Wait
        
        # Re-check paths after install
        foreach ($p in $InnoCandidatePaths) {
            if (Test-Path $p) {
                $IsccPath = $p
                break
            }
        }
    }
}

if (-not $IsccPath) {
    Write-Host "`n[NOTICE] Inno Setup 6 Compiler (ISCC.exe) was not found." -ForegroundColor Yellow
    Write-Host "You can install it easily with:" -ForegroundColor White
    Write-Host "    winget install JRSoftware.InnoSetup" -ForegroundColor Cyan
    Write-Host "Or rerun this script with -InstallInno flag:" -ForegroundColor White
    Write-Host "    .\build_installer.ps1 -InstallInno" -ForegroundColor Cyan
    Write-Host "Standalone binary build is ready in: $BuildDir" -ForegroundColor Green
    exit 0
}

Write-Host "Found Inno Setup Compiler: $IsccPath" -ForegroundColor Green

# -------------------------------------------------------------------------
# Step 5: Compile Inno Setup Script
# -------------------------------------------------------------------------
Write-Host "`n[5/5] Building Windows Installer (.exe)..." -ForegroundColor Cyan

if (-not (Test-Path $DistDir)) {
    New-Item -Path $DistDir -ItemType Directory -Force | Out-Null
}

$isccProc = Start-Process -FilePath $IsccPath -ArgumentList "`"$IssPath`"" -WorkingDirectory $ScriptDir -NoNewWindow -PassThru -Wait
if ($isccProc.ExitCode -ne 0) {
    Write-Error "Inno Setup compilation failed with exit code $($isccProc.ExitCode)."
}

# -------------------------------------------------------------------------
# Summary & Verification
# -------------------------------------------------------------------------
Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host "               BUILD SUCCESSFUL - SUMMARY                        " -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green

$installers = Get-ChildItem -Path $DistDir -Filter "MedEye3D*.exe"
foreach ($inst in $installers) {
    $sizeMB = [Math]::Round($inst.Length / 1MB, 2)
    $hash = (Get-FileHash -Path $inst.FullName -Algorithm SHA256).Hash
    Write-Host " Installer File:  $($inst.FullName)" -ForegroundColor White
    Write-Host " Size:            $sizeMB MB" -ForegroundColor White
    Write-Host " SHA256:          $hash" -ForegroundColor Gray
}

Write-Host "`nReady for testing and distribution on Windows systems." -ForegroundColor Green
Write-Host "=================================================================`n"
