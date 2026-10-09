#!/bin/bash
# Interactive 4-pane medical image visualization from MRB
# Run this inside the Docker container to start the viewer
set -e
export DEBIAN_FRONTEND=noninteractive

# Ensure clipboard support for Makie textboxes (Ctrl+C/V)
if ! command -v xclip &>/dev/null && ! command -v xsel &>/dev/null; then
    echo "Installing xclip for clipboard support..."
    sudo apt-get update -qq && sudo apt-get install -y -qq xclip 2>/dev/null || true
fi

# Use Mesa software renderer if no GPU driver available
export MESA_GL_VERSION_OVERRIDE=4.3
export MESA_GLSL_VERSION_OVERRIDE=430

# Ensure MPICH is installed (needed by HDF5_jll mpi+mpich artifact variant)
if ! dpkg -l libmpich-dev &>/dev/null 2>&1; then
    echo "Installing MPICH for HDF5_jll..."
    sudo apt-get update -qq && sudo apt-get install -y -qq libmpich-dev 2>/dev/null || true
fi

# Fix HDF5_jll RTLD_DEEPBIND issue: libhdf5_fortran.so needs symbols from libhdf5.so
# to be globally visible. The JLL wrapper uses RTLD_DEEPBIND which isolates symbol scopes.
# Patch the wrapper to add RTLD_GLOBAL alongside RTLD_DEEPBIND.
HDF5_WRAPPER=$(find /data/packages/HDF5_jll -name "x86_64-linux-gnu-*mpi+mpich.jl" 2>/dev/null | head -1)
if [ -n "$HDF5_WRAPPER" ] && ! grep -q "RTLD_GLOBAL" "$HDF5_WRAPPER" 2>/dev/null; then
    echo "  [HDF5] Patching RTLD_DEEPBIND → RTLD_DEEPBIND | RTLD_GLOBAL..."
    python3 -c "
with open('$HDF5_WRAPPER', 'r') as f:
    c = f.read()
c = c.replace('RTLD_LAZY | RTLD_DEEPBIND,', 'RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL,')
with open('$HDF5_WRAPPER', 'w') as f:
    f.write(c)
" 2>/dev/null || true
    # Clear stale precompiled caches
    find /data/compiled -name "*HDF5*" -exec rm -rf {} + 2>/dev/null || true
    find /data/compiled -name "*MedImages*" -exec rm -rf {} + 2>/dev/null || true
    find /data/compiled -name "*MedEye3d*" -exec rm -rf {} + 2>/dev/null || true
fi



echo ""
echo "============================================"
echo "  MedEye3d Interactive 4-Pane Viewer (MRB)"
echo "============================================"
echo ""
echo "  Display: $DISPLAY"
echo "  Controls:"
echo "    Mouse wheel  - scroll through slices"
echo "    Right-click  - jump all planes to clicked point"
echo "    Double-click - zoom panel / restore 4-pane"
echo "    Left-drag    - paint on mask"
echo "    Close window - exit"
echo ""
echo "============================================"
echo ""

cd "$(dirname "$0")/../.."
LOG_FILE="data/app_interactive.log"
echo "  Logging to: $(pwd)/$LOG_FILE"
echo ""

# Auto-resolve Manifest.toml if generated for a different Julia version.
# This handles shared bind-mount between host and Docker with different Julia patch versions.
CURRENT_JULIA_VER=$(julia -e 'print(VERSION)')
MANIFEST_VER=$(grep -m1 'julia_version' Manifest.toml 2>/dev/null | sed 's/.*"\(.*\)"/\1/')
if [ -n "$MANIFEST_VER" ] && [ "$CURRENT_JULIA_VER" != "$MANIFEST_VER" ]; then
    echo "  [MANIFEST] Version mismatch: Manifest=$MANIFEST_VER, Julia=$CURRENT_JULIA_VER"
    echo "  [MANIFEST] Auto-resolving (rm + instantiate)..."
    rm -f Manifest.toml
    julia --project=. -e '
        using Pkg
        devpkgs = PackageSpec[]
        # Re-add local dev dependency if MedImages.jl is available
        for p in ["/workspaces/MedImages.jl", "/mnt/big/project_ssd/project_ssd/MedImages.jl"]
            if isdir(p)
                push!(devpkgs, PackageSpec(path=p))
                break
            end
        end
        # Re-add local ITKIOWrapper.jl (PythonCall fork) if available
        for p in ["/workspaces/ITKIOWrapper.jl", "/mnt/big/project_ssd/project_ssd/ITKIOWrapper.jl"]
            if isdir(p)
                push!(devpkgs, PackageSpec(path=p))
                break
            end
        end
        if !isempty(devpkgs)
            Pkg.develop(devpkgs)
        end
        Pkg.instantiate()
    '
    echo "  [MANIFEST] Resolved for Julia $CURRENT_JULIA_VER"
fi

# Julia 1.11 has a known multi-threading SIGSEGV bug during JIT compilation of large
# codebases (Makie/GLMakie). Using 1 thread avoids the race in jl_mutex_wait.
# Threads.@spawn for IO and AI inference still works (runs on the single-thread pool).
julia --project=. --threads=1 scripts/app/run_interactive_mrb.jl "$@" 2>&1 | tee "$LOG_FILE"
