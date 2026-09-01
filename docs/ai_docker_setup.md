# MedEye3D: Remote AI Worker & SSH Tunneling Guide

## Architecture Overview

MedEye3D follows a decoupled client-server architecture:

1. **Lightweight GUI Client (Windows Standalone Installer / Local App)**:
   - Contains **ONLY** the visualization pipeline (Vulkan/ModernGL shaders, GLFW windowing, HDF5/NIfTI data streaming, multi-planar QuadView, interactive annotation controls).
   - Zero heavyweight machine learning frameworks (no PyTorch, CUDA binaries, or large model weight files bundled into the client installer).
   - Runs smoothly on standard laptops and workstations.

2. **Remote AI Inference Server (Docker on GPU Server)**:
   - Containerized environment (`medeye3d-ai`) running on a dedicated Linux machine equipped with NVIDIA GPUs.
   - Hosts all deep learning models:
     - **HELPNet**: Interactive PET/CT lesion segmentation.
     - **nnInteractive**: Real-time scribble-guided 3D volume segmentation with inline Base64 tensor streaming.
     - **TotalSegmentator**: Full-body anatomical structure and organ segmentation.
     - **Bone Subsegmentation**: PyTorch-based cortical surface vs. trabecular bone marrow subsegmentation.
   - Communicates with clients via a **TCP JSON RPC Protocol** (default port `5005`).

```
┌───────────────────────────────────────────────────┐
│        Local Workstation / Windows GUI            │
│  - MedEye3D.exe (Visualization & Annotation)      │
│  - InferenceClient.jl                             │
└─────────────────────────┬─────────────────────────┘
                          │ Local TCP (127.0.0.1:5005)
                          ▼
┌───────────────────────────────────────────────────┐
│          Encrypted SSH Tunnel (`ssh -L`)          │
└─────────────────────────┬─────────────────────────┘
                          │ Encrypted Remote Traffic
                          ▼
┌───────────────────────────────────────────────────┐
│        Remote Linux Server with NVIDIA GPU        │
│  - Docker Container: `medeye3d-ai`                │
│  - Python Worker TCP Server (0.0.0.0:5005)        │
│  - PyTorch / nnUNet / TotalSegmentator / HELPNet  │
└───────────────────────────────────────────────────┘
```

---

## Step-by-Step Setup Guide

### Step 1: Launch the AI Docker Worker on the Remote GPU Server

On your remote GPU server, run the `medeye3d-ai` container and publish port `5005`:

```bash
# 1. SSH into your remote GPU server
ssh user@remote-gpu-server-ip

# 2. Navigate to MedEye3d repository on server
cd /path/to/MedEye3d.jl

# 3. Start the Docker AI Worker container with GPU acceleration
docker rm -f medeye3d-ai 2>/dev/null

docker run -d --restart=unless-stopped \
  --name medeye3d-ai \
  --gpus all \
  --shm-size=64g \
  -p 5005:5005 \
  -v /path/to/MedEye3d.jl/tmp_inference:/tmp/medeye3d_inference \
  -v /path/to/MedEye3d.jl/scripts/ai:/app \
  -e TOTALSEG_LICENSE_NUMBER="your_license_here" \
  medeye3d-ai:latest

# 4. Verify that the Python TCP server is listening on port 5005
docker logs -f medeye3d-ai
# Output: [Worker] TCP JSON Server listening on port 5005...
```

---

### Step 2: Establish the SSH Tunnel on the Local Windows Machine

On your local Windows machine, forward local port `5005` to the remote server's port `5005` using OpenSSH.

#### Option A: Quick Command Line (PowerShell or Windows Terminal)

```powershell
# Open PowerShell and run:
ssh -N -L 5005:localhost:5005 user@remote-gpu-server-ip
```

*Flags explained:*
- `-N`: Do not execute remote commands (forward ports only).
- `-L 5005:localhost:5005`: Binds `127.0.0.1:5005` on your local machine to `localhost:5005` on the remote server.

#### Option B: Background Persistent SSH Tunnel via Windows OpenSSH

To keep the tunnel alive in the background:

```powershell
# Run in PowerShell (background job)
Start-Job -ScriptBlock {
    ssh -N -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3" -L 5005:localhost:5005 user@remote-gpu-server-ip
}
```

#### Option C: PuTTY / Pageant Configuration

1. Open **PuTTY**.
2. Under **Host Name**, enter `remote-gpu-server-ip` and port `22`.
3. In the left panel, navigate to **Connection** > **SSH** > **Tunnels**.
4. In **Source port**, enter `5005`.
5. In **Destination**, enter `localhost:5005`.
6. Click **Add**, then click **Open** and authenticate.

---

### Step 3: Configure Local GUI Client Registration

The MedEye3D standalone executable automatically looks for the AI worker on `127.0.0.1:5005` by default (which routes through your SSH tunnel).

If you use custom ports or a direct private network / VPN, configure the environment variables:

| Environment Variable | Default Value | Description |
|---|---|---|
| `MEDEYE3D_AI_HOST` | `127.0.0.1` | Host IP or domain of the AI worker (local forward or remote IP) |
| `MEDEYE3D_AI_PORT` | `5005` | TCP port of the AI worker |

#### Setting Environment Variables on Windows:

```powershell
# Temporary (for current PowerShell session)
$env:MEDEYE3D_AI_HOST = "127.0.0.1"
$env:MEDEYE3D_AI_PORT = "5005"
& "C:\Users\jakub\AppData\Local\Programs\MedEye3D\bin\MedEye3D.exe"

# Permanent (System / User Level)
[Environment]::SetEnvironmentVariable("MEDEYE3D_AI_HOST", "127.0.0.1", "User")
[Environment]::SetEnvironmentVariable("MEDEYE3D_AI_PORT", "5005", "User")
```

---

### Step 4: Verify Connection & Interactive Inference

1. Launch MedEye3D:
   ```powershell
   & "C:\Users\jakub\AppData\Local\Programs\MedEye3D\bin\MedEye3D.exe"
   ```
2. Check the startup logs in `%APPDATA%\MedEye3D\logs\medeye3d_output.log`:
   ```
   [InferenceClient] Ensuring MedEye3d AI Worker is reachable at 127.0.0.1:5005...
   [InferenceClient] AI Worker is ready at 127.0.0.1:5005.
   ```
3. Test Interactive Annotation in QuadView:
   - **Click & Scribble Annotation**: Paint a stroke over a lesion or organ and trigger `nnInteractive` or `HELPNet`.
   - The GUI transmits 3D scribble coordinates or patch arrays over the SSH tunnel.
   - The remote GPU computes the segmentation in < 2 seconds and returns an inline Base64-compressed mask directly into the viewport.

---

## Troubleshooting & Diagnostics

### 1. Connection Refused (`ECONNREFUSED` / `connect: connection refused`)
- **Cause**: The SSH tunnel is not running or the Docker container is not active on the server.
- **Fix**:
  1. On server: `docker ps` to verify `medeye3d-ai` is up and `-p 5005:5005` is listed.
  2. On client: Verify tunnel with `Test-NetConnection -ComputerName 127.0.0.1 -Port 5005` in PowerShell.

### 2. SSH Tunnel Dropped / Disconnected
- **Fix**: Add keep-alive flags to your SSH command:
  ```powershell
  ssh -N -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3" -L 5005:localhost:5005 user@remote-gpu-server-ip
  ```

### 3. Out of GPU Memory on Remote Server
- **Fix**: Check `nvidia-smi` on the remote server. TotalSegmentator and HELPNet require ~4-6 GB of available VRAM. If multiple GPUs are present, specify the target GPU: `--gpus '"device=1"'`.
