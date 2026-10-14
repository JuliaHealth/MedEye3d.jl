#!/bin/bash
# Helper to verify and configure X11 authorization inside Docker
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -n "$DISPLAY" ]; then
    # Auto-merge host X11 auth cookie if present
    if [ -f "$REPO_ROOT/.docker.xauth" ]; then
        xauth nmerge "$REPO_ROOT/.docker.xauth" 2>/dev/null || true
    fi

    # Verify X11 connection before attempting GUI launch
    if ! xdpyinfo >/dev/null 2>&1; then
        echo ""
        echo "============================================================"
        echo "  [X11] Cannot connect to display '$DISPLAY' (Authorization failed)"
        echo ""
        echo "  Please run this on your HOST terminal to grant access:"
        echo ""
        echo "      xhost +local:"
        echo ""
        echo "============================================================"
        echo ""
        for _wait in {1..5}; do
            if [ -f "$REPO_ROOT/.docker.xauth" ]; then
                xauth nmerge "$REPO_ROOT/.docker.xauth" 2>/dev/null || true
            fi
            if xdpyinfo >/dev/null 2>&1; then
                echo "  [OK] X11 display access granted!"
                break
            fi
            sleep 1
        done

        if ! xdpyinfo >/dev/null 2>&1; then
            echo "ERROR: X11 display '$DISPLAY' is not accessible. Run 'xhost +local:' on host."
            exit 1
        fi
    fi
fi
