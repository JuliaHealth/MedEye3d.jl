# MedEye3D - Windows Standalone Packaging & Operational Guide

This directory contains the production-ready build, packaging, testing, and operational infrastructure to transform **MedEye3D.jl** into an installable, standalone Microsoft Windows application (`MedEye3D_v<version>_Setup.exe`).

---

## 📁 Packaging Subsystem Structure

| File | Purpose |
| :--- | :--- |
| [`AppMain.jl`](file:///D:/MedEye3d.jl/src/packaging/AppMain.jl) | Standalone application entrypoint (`julia_main()`), GUI initialization, stdio redirection to `%APPDATA%\MedEye3D\logs` to prevent Windows GUI subsystem descriptor `-2` crashes, AI startup prompt dialog, and CLI argument dispatch (`--demo`, `--help`, `--start-ai`, `--no-ai`, image files). |
| [`precompile_app.jl`](file:///D:/MedEye3d.jl/src/packaging/precompile_app.jl) | Workload execution tracer passed to `PackageCompiler.jl`. Exercises all types, `TextureSpec`, array transformations, resamplings, OpenGL/Vulkan structures, and SQLite FTS5 anatomy indices during build to eliminate JIT latency. |
| [`build_app.jl`](file:///D:/MedEye3d.jl/src/packaging/build_app.jl) | Julia build script executing `PackageCompiler.create_app()`, bundling runtime sysimage (`sys.dll`), C entrypoint (`MedEye3D.exe`), ontologies (`FoundationalAnatomy.csv`, `max_anatomy_to_ontology.json`), and containerized AI worker scripts (`scripts/ai/*`). |
| [`MedEye3D_Installer.iss`](file:///D:/MedEye3d.jl/src/packaging/MedEye3D_Installer.iss) | Inno Setup 6 script configured for LZMA2 ultra-compression, Start Menu/Desktop shortcuts, Windows Add/Remove programs registry integration, and medical file associations (`.nii`, `.nii.gz`, `.mha`, `.h5`). |
| [`build_installer.ps1`](file:///D:/MedEye3d.jl/src/packaging/build_installer.ps1) | One-command end-to-end PowerShell build pipeline with Inno Setup auto-detection, Julia version channel management, and SHA256 integrity verification. |
| [`generate_icon.ps1`](file:///D:/MedEye3d.jl/src/packaging/generate_icon.ps1) | Generates the multi-resolution `.ico` icon file (16x16 to 256x256) embedded in the installer and executable. |
| [`app_icon.ico`](file:///D:/MedEye3d.jl/src/packaging/app_icon.ico) | Multi-resolution icon for MedEye3D. |

---

## 🚀 Building the Windows Installer

### Prerequisites
1. **Julia 1.11+ (x86_64)**:
   - For CxxWrap/JLL binary compatibility, Julia 1.11 channel is strongly recommended:
     ```powershell
     juliaup add 1.11
     juliaup default 1.11
     ```
2. **Inno Setup 6**:
   - Install via `winget` (or pass `-InstallInno` to the build script):
     ```powershell
     winget install JRSoftware.InnoSetup
     ```
3. **PowerShell 5.1+ or PowerShell 7+** (Run as Administrator or standard user).

### One-Command Build Pipeline
From the repository root or packaging directory, run:

```powershell
powershell -ExecutionPolicy Bypass -File D:\MedEye3d.jl\src\packaging\build_installer.ps1
```

The script automatically executes:
1. **Step 1/5**: Verifies / generates the application icon (`app_icon.ico`).
2. **Step 2/5**: Validates the Julia executable (using `+1.11` channel if installed).
3. **Step 3/5**: Runs `build_app.jl` with `PackageCompiler.create_app()`:
   - Compiles Ahead-of-Time sysimage (`build\MedEye3D_dist\lib\julia\sys.dll`).
   - Copies UBERON ontologies, metadata schemas, and AI worker scripts into `build\MedEye3D_dist`.
4. **Step 4/5**: Auto-locates Inno Setup compiler (`ISCC.exe`).
5. **Step 5/5**: Compiles `MedEye3D_Installer.iss` into:
   ```
   D:\MedEye3d.jl\dist\MedEye3D_v0.5.11_Setup.exe
   ```

### 🛠️ Build Options & Parameters

| Parameter | Description |
| :--- | :--- |
| *(default)* | Full end-to-end clean compilation (PackageCompiler + Inno Setup). Takes ~15-20 min. |
| `-SkipCompile` | Skips PackageCompiler step; packages existing `build\MedEye3D_dist` into installer. Takes ~10-15 min. |
| `-Clean` | Deletes previous `build\MedEye3D_dist` and `dist\` directories before starting. |
| `-InstallInno` | Automatically downloads and installs Inno Setup 6 via `winget` if not detected. |

Examples:
```powershell
# Rebuild only installer after updating scripts or assets (fast):
powershell -ExecutionPolicy Bypass -File .\src\packaging\build_installer.ps1 -SkipCompile

# Full clean rebuild from scratch:
powershell -ExecutionPolicy Bypass -File .\src\packaging\build_installer.ps1 -Clean -InstallInno
```

---

## 🧪 Testing the Windows Installer & Application

### 1. Running the Setup Wizard
Double-click `dist\MedEye3D_v0.5.11_Setup.exe` to test the installation wizard:
- Verify custom install location (defaults to `%LOCALAPPDATA%\Programs\MedEye3D` without requiring administrator elevation).
- Verify optional desktop and Start Menu shortcuts.
- Verify medical file associations (`.nii`, `.nii.gz`, `.mha`, `.h5`).

### 2. Testing CLI Commands & Flags
Open Command Prompt or PowerShell and test the installed executable:

```powershell
# Display help and supported CLI flags
MedEye3D.exe --help

# Display version
MedEye3D.exe --version

# Launch interactive demo with synthetic CT/PET phantom (no patient files needed)
MedEye3D.exe --demo

# Launch directly with patient scan
MedEye3D.exe "C:\scans\patient_ct.nii.gz"

# Force enable local AI inference without prompting:
MedEye3D.exe --start-ai

# Force disable local AI inference (viewer-only mode):
MedEye3D.exe --no-ai
```

### 3. Testing AI Features in GUI
1. On application startup, when prompted with **"Would you like to run the AI inference models (nnInteractive & HELPNet) locally?"**, click **Yes**.
2. MedEye3D connects to `127.0.0.1:5005` (or automatically launches the Docker container via `start_docker_worker.ps1`).
3. In the Lesion Metadata Window (`AI:` row):
   - **`NNInteractive`**: Paint 1-2 scribble strokes across the lesion, click **Run AI**. nnInteractive segments the volume and streams back the prediction.
   - **`HELPNet (AI)`**: Click or paint on a PET/CT lesion, click **Run AI**. HELPNet generates the 3D patch prediction, filters the largest connected component (with automatic CPU fallback if host GPU drivers are unsupported), and paints the lesion.

---

## 🔍 Locating Logs & Debugging Issues

When diagnosing issues (e.g., failed AI inference, missing files, or crashes), consult the following log sources:

### 1. Application Runtime Logs (Windows Client)
Because `MedEye3D.exe` runs as a native Windows GUI application, `stdout` and `stderr` are redirected to disk to prevent OS descriptor crashes:

- **Log Directory**: `%APPDATA%\MedEye3D\logs\` (usually `C:\Users\<Username>\AppData\Roaming\MedEye3D\logs\`)
- **Key Files**:
  - `medeye3d_output.log`: Standard output (startup messages, model results, voxel counts).
  - `medeye3d_error.log`: Standard error and warning traces.
  - `medeye3d_session.log`: Chronological history of all sessions, launch arguments, and exit codes.
  - `medeye3d_output_prev.log` / `medeye3d_error_prev.log`: Retained logs from the preceding session.

> [!TIP]
> **Viewing Active Logs While MedEye3D is Running**:
> Because Windows locks files opened for writing, use `Get-Content -Wait` with shared access in PowerShell:
> ```powershell
> Get-Content -Path "$env:APPDATA\MedEye3D\logs\medeye3d_output.log" -Tail 50 -Wait
> ```
> Or in Python:
> ```python
> with open(r"C:\Users\<user>\AppData\Roaming\MedEye3D\logs\medeye3d_output.log", "r", encoding="utf-8", errors="replace") as f:
>     print(f.read())
> ```

### 2. Docker AI Worker Logs
To inspect deep learning model execution inside the container:

```powershell
# Show last 100 log lines from the AI container
docker logs medeye3d-ai --tail 100

# Live stream container logs
docker logs -f medeye3d-ai
```

### 3. Temporary Inference Exchange Directory
During inference, Julia and Docker exchange volume patches via a shared directory:
- **Location**: `D:\MedEye3d.jl\tmp_inference` (or `%APPDATA%\MedEye3D\tmp_inference`)
- **Key Files**:
  - `ct_in.nii.gz`: Extracted 64³ CT patch (or full volume for nnInteractive).
  - `pet_in.nii.gz`: Extracted 64³ PET patch.
  - `point_in.nii.gz`: Center point click prompt (`[33, 33, 33]`) for HELPNet.
  - `helpnet_prediction.nii.gz`: Binary segmentation output produced by HELPNet.
  - `nn_ct_<hash>.nii.gz`: Cached full CT volume for nnInteractive.

---

## 🐳 Handling Docker on Windows

MedEye3D uses containerized AI workers so that complex Python, PyTorch, and CUDA dependencies do not pollute the Windows host system.

### 1. Requirements for Docker Desktop
1. **Docker Desktop for Windows** installed with **WSL 2 backend** enabled:
   - Docker Desktop Settings $\to$ **General** $\to$ check **"Use the WSL 2 based engine"**.
2. **NVIDIA GPU Passthrough**:
   - Install the latest NVIDIA Windows Display Driver (WSL 2 GPU acceleration is natively supported).
   - In Docker Desktop Settings $\to$ **Resources** $\to$ **WSL Integration** $\to$ enable integration with your default WSL distro.
   - Verify GPU access inside Docker:
     ```powershell
     docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
     ```

### 2. WSL 2 Memory & Resource Configuration
Deep learning models (especially nnInteractive and TotalSegmentator) require adequate RAM. Configure `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=16GB
processors=8
swap=8GB
```
Restart WSL after editing: `wsl --shutdown`.

### 3. Managing the Container

#### Automated Launcher (Recommended)
Use the included PowerShell launcher script:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ai\start_docker_worker.ps1
```
The script will:
1. Detect Docker Desktop install paths and launch the daemon if not running.
2. Build the Docker image `medeye3d-ai:latest` if not present.
3. Start the container with GPU passthrough (`--gpus all`) and port mapping (`5005:5005`).

#### Manual Docker Commands
```powershell
# Check if container is running
docker ps -a --filter "name=medeye3d-ai"

# Start existing stopped container
docker start medeye3d-ai

# Stop container
docker stop medeye3d-ai

# Restart container
docker restart medeye3d-ai

# Test TCP port connectivity from Windows host
Test-NetConnection -ComputerName 127.0.0.1 -Port 5005
```

### 4. Common Docker on Windows Issues & Fixes

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| **`Connection refused at 127.0.0.1:5005`** | Docker daemon is not running, or container `medeye3d-ai` is stopped. | Start Docker Desktop, or run `powershell -File .\scripts\ai\start_docker_worker.ps1`. |
| **`CUDA_ERROR_NOT_SUPPORTED` on host** | Host Windows NVIDIA driver / CUDA mismatch during Julia `KernelAbstractions` post-processing. | MedEye3D automatically catches this and falls back to CPU KA kernels. Ensure you use MedEye3D v0.5.11+. |
| **`FileNotFoundError: /tmp/medeye3d_inference/D:\...`** | Cross-platform backslash paths sent to Linux container. | Handled automatically via `pathlib.PureWindowsPath` in `python_worker.py` and container-relative paths in `InferenceClient.jl`. |
| **Docker daemon startup timeout** | WSL2 taking too long to initialize. | Open PowerShell as Administrator, run `wsl --update`, then `wsl --shutdown`. Rerun `start_docker_worker.ps1`. |
| **Port 5005 already in use** | A dangling process or previous container instance is bound to port 5005. | Run `docker rm -f medeye3d-ai`, check `netstat -ano \| findstr 5005`, and restart the worker. |
| **GPU out of memory (OOM)** | Large volumes or multiple models loaded simultaneously. | Ensure at least 4 GB VRAM is free before triggering `Run AI`. For 4-6 GB cards (e.g. Quadro P2000), `NNInteractive` uses autozoom and HELPNet uses 64³ focal patches. |

