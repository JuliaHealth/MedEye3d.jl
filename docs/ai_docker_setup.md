# MedEye3D: AI Worker Setup, Windows Docker & Troubleshooting Guide

## Architecture Overview

MedEye3D follows a decoupled client-server architecture:

1. **Lightweight GUI Client (Windows Standalone Installer / Local App)**:
   - Contains **ONLY** the visualization pipeline (Vulkan/ModernGL shaders, GLFW windowing, HDF5/NIfTI data streaming, multi-planar QuadView, interactive annotation controls).
   - Zero heavyweight machine learning frameworks (no PyTorch, CUDA binaries, or large model weight files bundled into the client installer).
   - Runs smoothly on standard laptops and workstations.

2. **AI Inference Server (Local Docker on Windows or Remote GPU Server)**:
   - Containerized environment (`medeye3d-ai`) with direct GPU access.
   - Hosts all deep learning models:
     - **HELPNet**: Interactive PET/CT lesion segmentation.
     - **nnInteractive**: Real-time scribble-guided 3D volume segmentation with inline Base64 tensor streaming.
     - **TotalSegmentator**: Full-body anatomical structure and organ segmentation.
     - **Bone Subsegmentation**: PyTorch-based cortical surface vs. trabecular bone marrow subsegmentation.
   - Communicates with clients via a **TCP JSON RPC Protocol** (default port `5005`).

```
┌─────────────────────────────────────────────────────────────┐
│              Local Workstation / Windows GUI                │
│  - MedEye3D.exe (Visualization, QuadView & Annotation)      │
│  - InferenceClient.jl                                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ TCP (127.0.0.1:5005)
         ┌─────────────────────┴─────────────────────┐
         │                                           │
         ▼ (Scenario A: Local Docker)                ▼ (Scenario B: Remote Server)
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│     Docker Desktop on Windows   │   │   SSH Tunnel (ssh -L 5005:...)  │
│   - WSL 2 Engine + NVIDIA GPU   │   └────────────────┬────────────────┘
│   - Container: medeye3d-ai      │                    │ Encrypted SSH
│   - TCP Port 5005               │                    ▼
└─────────────────────────────────┘   ┌─────────────────────────────────┐
                                      │   Remote Linux GPU Server       │
                                      │   - Container: medeye3d-ai      │
                                      │   - PyTorch / nnUNet / HELPNet  │
                                      └─────────────────────────────────┘
```

---

## 🐳 Scenario A: Local Docker on Windows (WSL 2)

If your Windows workstation has an NVIDIA GPU (e.g. GeForce RTX, Quadro, or Tesla), you can run the AI models locally inside Docker.

### 1. Requirements for Docker Desktop on Windows
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

### 3. Starting the Container

#### Automated Launcher (Recommended)
Use the included PowerShell launcher script from the repository or installer:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ai\start_docker_worker.ps1
```
The script will:
1. Detect Docker Desktop install paths and launch the daemon if not running.
2. Build the Docker image `medeye3d-ai:latest` if not present.
3. Start the container with GPU passthrough (`--gpus all`) and port mapping (`5005:5005`).

#### Manual Docker Command
```powershell
# Create inference exchange directory
New-Item -ItemType Directory -Force -Path .\tmp_inference

# Run container
docker run -d --restart=unless-stopped `
  --name medeye3d-ai `
  --gpus all `
  --shm-size=16g `
  -p 5005:5005 `
  -v "${PWD}\tmp_inference:/tmp/medeye3d_inference" `
  -v "${PWD}\scripts\ai:/app" `
  medeye3d-ai:latest
```

---

## 🌐 Scenario B: Remote AI Worker via SSH Tunneling

If your local Windows machine lacks an NVIDIA GPU, you can run the Docker container on a remote Linux server and tunnel port `5005`.

### Step 1: Launch the AI Docker Worker on Remote Server

```bash
# 1. SSH into your remote GPU server
ssh user@remote-gpu-server-ip

# 2. Start the Docker container
docker run -d --restart=unless-stopped \
  --name medeye3d-ai \
  --gpus all \
  --shm-size=64g \
  -p 5005:5005 \
  -v /path/to/MedEye3d.jl/tmp_inference:/tmp/medeye3d_inference \
  -v /path/to/MedEye3d.jl/scripts/ai:/app \
  medeye3d-ai:latest

# 3. Verify server is listening
docker logs -f medeye3d-ai
# Output: [Worker] TCP JSON Server listening on port 5005...
```

### Step 2: Establish the SSH Tunnel on Local Windows Machine

```powershell
# Forward local port 5005 to remote server port 5005
ssh -N -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3" -L 5005:localhost:5005 user@remote-gpu-server-ip
```

---

## 🔍 Locating Logs & Debugging Issues

When diagnosing issues (such as failed AI inference, missing files, or crashes), consult the following three log sources:

### 1. Application Runtime Logs (Windows Client)
Because `MedEye3D.exe` runs as a native Windows GUI application, `stdout` and `stderr` are redirected to disk to prevent OS descriptor `-2` crashes:

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

## 🛠️ Diagnostics: Why nnInteractive Works but HELPNet Failed (Root Cause Analysis)

### The Architectural Difference
1. **`nnInteractive` (Inline Base64 Transfer)**:
   - When nnInteractive completes inference on the GPU inside Docker, it returns a 3D binary mask directly encoded as a Base64 string in the TCP JSON payload.
   - Julia receives the string, decodes it into a CPU bit array, and paints it immediately into the GUI texture. No host GPU post-processing is executed.

2. **`HELPNet` (Focal Patch + Host Post-processing)**:
   - HELPNet generates a $64 \times 64 \times 64$ patch file `helpnet_prediction.nii.gz`.
   - The Julia host reads this patch and performs **Largest Connected Component (LCC)** filtering via `ConnectedComponents.extract_largest_connected_component` to isolate the true lesion from surrounding background noise.
   - **The Failure**: In older builds, `extract_largest_connected_component` attempted to compile and execute a CUDA kernel on the Windows host using `KernelAbstractions.jl`. If the host Windows NVIDIA driver did not support JIT compilation (e.g. `CUDA error: operation not supported (code 801, ERROR_NOT_SUPPORTED)`), the function threw an unhandled exception.
   - **The Fix**: In MedEye3D v0.5.11+, `ConnectedComponents.jl` wraps GPU execution in a `try ... catch` block and automatically falls back to multithreaded CPU KA kernels. The prediction (e.g. 56 voxels $\to$ 44 cleaned voxels) is seamlessly extracted and placed into the patient volume.

---

## 📦 Building & Testing the Windows Installer

For instructions on building and packaging the standalone Windows installer:
- See the dedicated packaging guide: [`src/packaging/README.md`](file:///D:/MedEye3d.jl/src/packaging/README.md).
- Quick clean rebuild command:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\src\packaging\build_installer.ps1 -Clean
  ```
- Output deliverable: `dist\MedEye3D_v0.5.11_Setup.exe`.
