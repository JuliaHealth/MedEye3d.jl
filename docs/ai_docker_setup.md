# MedEye3d: AI Inference Docker Setup

## Architecture

MedEye3d uses a two-container architecture:

1. **GUI Container** (`sharp_ramanujan`) — Runs the Julia/Vulkan viewer + Makie control panel
2. **AI Container** (`medeye3d-ai`) — Runs the Python inference worker (HELPNet, nnInteractive, TotalSegmentator)

The AI container communicates with the GUI container via a **TCP JSON protocol on port 5005**.
Since both run as Docker containers, they share a network namespace so that `127.0.0.1:5005`
is reachable from both.

## Quick Start

### Step 1: Start the AI Container (from host)

```bash
# From the host machine (NOT from inside the GUI container):
cd /mnt/big/project_ssd/project_ssd/MedEye3d.jl
bash scripts/ai/start_docker_worker.sh
```

This script:
- Detects if the GUI container (`sharp_ramanujan`) is running
- Shares its network namespace (`--network=container:sharp_ramanujan`)
- Uses GPU 1 for inference (`--gpus "device=1"`)
- Mounts the shared inference directory

### Step 2: Verify the Worker is Ready

```bash
docker logs medeye3d-ai
# Should see: [Worker] TCP JSON Server listening on port 5005...
```

### Step 3: Test Connectivity from GUI Container

```bash
docker exec sharp_ramanujan julia -e '
using Sockets; connect("127.0.0.1", 5005) |> close; println("OK")
'
```

### Step 4: Start MedEye3d

```bash
docker exec -it \
  -e DISPLAY=$DISPLAY \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  sharp_ramanujan \
  bash -c "cd /workspaces/MedEye3d.jl && JULIA_NUM_THREADS=3,1 julia --project=. scripts/app/run_interactive_mrb.jl"
```

## Manual Docker Commands

If `start_docker_worker.sh` doesn't work, run manually:

```bash
docker rm -f medeye3d-ai 2>/dev/null

docker run -d --rm --name medeye3d-ai \
  --gpus '"device=1"' \
  --shm-size=64g \
  --network=container:sharp_ramanujan \
  -v /mnt/big/project_ssd/project_ssd/MedEye3d.jl/tmp_inference:/tmp/medeye3d_inference \
  -v /mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/ai:/app \
  -v /mnt/big/project_ssd/project_ssd:/mnt/big/project_ssd/project_ssd \
  -v /home/jm:/home/jm \
  medeye3d-ai:latest
```

**Key flag**: `--network=container:sharp_ramanujan` — shares the GUI container's network
namespace so `127.0.0.1:5005` is reachable from both containers.

## Troubleshooting

### "Connection refused" error

```
IOError: connect: connection refused (ECONNREFUSED)
```

1. Check if AI container is running: `docker ps | grep medeye3d-ai`
2. If not running: `bash scripts/ai/start_docker_worker.sh` (from host)
3. Check AI container logs: `docker logs medeye3d-ai`

### AI container exits immediately

Check logs: `docker logs medeye3d-ai`

Common causes:
- GPU not available (change `--gpus "device=1"` to `"device=0"`)
- Out of GPU memory (~4GB needed)
- Missing model weights (auto-downloaded on first run, ~2GB)

### Rebuilding the AI Docker image

```bash
docker rm -f medeye3d-ai 2>/dev/null
docker rmi medeye3d-ai:latest
bash scripts/ai/start_docker_worker.sh
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MEDEYE3D_GUI_CONTAINER` | `sharp_ramanujan` | Name of GUI container for network sharing |
| `TOTALSEG_LICENSE_NUMBER` | `aca_XHEO7L1IH2U7G7` | TotalSegmentator license key |
