# start_docker_worker.ps1 - Launch MedEye3D AI Docker container on Windows
[CmdletBinding()]
param(
    [string]$Port = "5005",
    [switch]$BuildIfMissing = $true,
    [string]$InferenceDir = ""
)

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..") -ErrorAction SilentlyContinue

Write-Host "[MedEye3D AI] Checking Docker environment on Windows..." -ForegroundColor Cyan

# 1. Verify Docker CLI is accessible
if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Warning "[MedEye3D AI] Docker CLI not found in PATH. Please install Docker Desktop (https://www.docker.com) and add 'docker' to your PATH."
    exit 1
}

# 2. Check if Docker daemon is running; if not, try to start Docker Desktop
$daemonRunning = $false
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        $daemonRunning = $true
    }
} catch {
    $daemonRunning = $false
}

if (-not $daemonRunning) {
    $desktopCandidates = @(
        "C:\Program Files\Docker\Docker\Docker Desktop.exe",
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Docker\Docker\Docker Desktop.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Docker\Docker Desktop.exe")
    )
    $foundDesktop = $null
    foreach ($cand in $desktopCandidates) {
        if ($cand -and (Test-Path $cand)) {
            $foundDesktop = $cand
            break
        }
    }
    if ($foundDesktop) {
        Write-Host "[MedEye3D AI] Docker daemon not running. Launching Docker Desktop ($foundDesktop)..." -ForegroundColor Yellow
        Start-Process $foundDesktop
        # Wait up to 45 seconds for Docker daemon to become responsive
        for ($i = 1; $i -le 15; $i++) {
            Start-Sleep -Seconds 3
            try {
                $null = docker info 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $daemonRunning = $true
                    Write-Host "[MedEye3D AI] Docker Desktop is now running." -ForegroundColor Green
                    break
                }
            } catch {}
            Write-Host "[MedEye3D AI] Waiting for Docker Desktop to initialize ($($i*3)/45s)..." -ForegroundColor Gray
        }
    }
}

if (-not $daemonRunning) {
    Write-Warning "[MedEye3D AI] Docker daemon is not running. Please start Docker Desktop manually."
    exit 2
}

# 3. Check if medeye3d-ai container is already running
$running = docker ps -q -f "name=medeye3d-ai" 2>$null
if ($running) {
    Write-Host "[MedEye3D AI] Container 'medeye3d-ai' is already running." -ForegroundColor Green
    exit 0
}

# 4. Remove any stopped container with the same name
docker rm -f medeye3d-ai 2>$null | Out-Null

# 5. Check if image exists; build if missing
$image = docker images -q medeye3d-ai:latest 2>$null
if (-not $image) {
    if ($BuildIfMissing) {
        Write-Host "[MedEye3D AI] Building Docker image 'medeye3d-ai:latest' from $ScriptDir..." -ForegroundColor Cyan
        docker build -t medeye3d-ai:latest "$ScriptDir"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "[MedEye3D AI] Failed to build Docker image."
            exit 3
        }
    } else {
        Write-Warning "[MedEye3D AI] Docker image 'medeye3d-ai:latest' not found."
        exit 4
    }
}

# 6. Prepare volume directories
if (-not $InferenceDir) {
    if ($ProjectRoot -and (Test-Path $ProjectRoot)) {
        $candidate = Join-Path $ProjectRoot "tmp_inference"
        try {
            if (-not (Test-Path $candidate)) { New-Item -ItemType Directory -Path $candidate -Force | Out-Null }
            $testFile = Join-Path $candidate ".perm_test"
            [IO.File]::WriteAllText($testFile, "test")
            Remove-Item $testFile -Force
            $InferenceDir = $candidate
        } catch {
            $InferenceDir = $null
        }
    }
    if (-not $InferenceDir) {
        $userDir = Join-Path $env:USERPROFILE ".medeye3d\tmp_inference"
        try {
            if (-not (Test-Path $userDir)) { New-Item -ItemType Directory -Path $userDir -Force | Out-Null }
            $InferenceDir = $userDir
        } catch {
            $tempDir = Join-Path $env:TEMP "medeye3d_inference"
            if (-not (Test-Path $tempDir)) { New-Item -ItemType Directory -Path $tempDir -Force | Out-Null }
            $InferenceDir = $tempDir
        }
    }
} else {
    if (-not (Test-Path $InferenceDir)) {
        New-Item -ItemType Directory -Path $InferenceDir -Force | Out-Null
    }
}

# Convert Windows paths to POSIX slashes for Docker volume mounts
$HostInference = ($InferenceDir -replace '\\', '/')
$HostApp = ($ScriptDir -replace '\\', '/')

# Check GPU support
$gpuArgs = @()
try {
    $null = docker run --rm --gpus all medeye3d-ai:latest nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $gpuArgs = @("--gpus", "all")
        Write-Host "[MedEye3D AI] NVIDIA GPU acceleration detected and enabled." -ForegroundColor Green
    }
} catch {
    Write-Host "[MedEye3D AI] Note: running container with standard CPU configuration."
}

Write-Host "[MedEye3D AI] Starting container 'medeye3d-ai' on port $Port..." -ForegroundColor Cyan

$dockerArgs = @(
    "run", "-d", "--restart=unless-stopped",
    "--name", "medeye3d-ai",
    "-p", "${Port}:5005",
    "--shm-size=16g"
)
if ($gpuArgs.Count -gt 0) {
    $dockerArgs += $gpuArgs
}
$dockerArgs += @(
    "-v", "${HostInference}:/tmp/medeye3d_inference",
    "-v", "${HostApp}:/app",
    "medeye3d-ai:latest"
)

& docker @dockerArgs
if ($LASTEXITCODE -eq 0) {
    Write-Host "[MedEye3D AI] Container 'medeye3d-ai' started successfully." -ForegroundColor Green
    exit 0
} else {
    Write-Error "[MedEye3D AI] Failed to start Docker container."
    exit 5
}
