#!/usr/bin/env python3
"""
sync_cads_to_xnat.py - Uploads CADS-patched preprocessed_volumes.h5 to XNAT MEDEYE3D
resource for any cases where the local file is newer/larger than the remote version.
"""
import os
import sys
import json
import time
import re

sys.path.insert(0, "/workspaces/MedEye3d.jl/scripts/preprocessing")
from batch_process_cohort_dual_gpu import XnatManager, PROJ_ID, CENSUS_PATH, OUT_BASE_DIR


def main():
    xnat = XnatManager()
    with open(CENSUS_PATH, "r") as f:
        census = json.load(f)

    all_pids = sorted(census.keys(), key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 9999)

    queue = []
    for pid in all_pids:
        h5_local = os.path.join(OUT_BASE_DIR, pid, "preprocessed_volumes.h5")
        if not os.path.isfile(h5_local):
            continue
        local_sz = os.path.getsize(h5_local)
        info = census[pid]
        sid = info["subj_id"]
        slabel = info["subj_label"]
        remote_h5 = f"{slabel}_preprocessed.h5"

        try:
            res = xnat.get_json(f"/data/projects/{PROJ_ID}/subjects/{sid}/resources/MEDEYE3D/files?format=json")
            items = res.get("ResultSet", {}).get("Result", [])
            remote_sz = 0
            for item in items:
                if item.get("Name") == remote_h5:
                    remote_sz = int(item.get("Size") or 0)
                    break
            if abs(local_sz - remote_sz) > 1000:
                queue.append((pid, sid, slabel, remote_h5, h5_local, local_sz, remote_sz))
        except Exception as e:
            print(f"[{pid}] Error checking remote status: {e}")

    print(f"Found {len(queue)} cases to sync to XNAT.")
    for idx, (pid, sid, slabel, remote_h5, h5_local, local_sz, remote_sz) in enumerate(queue, 1):
        print(f"[{idx}/{len(queue)}] Uploading {pid} ({local_sz/1e6:.1f} MB, diff={local_sz-remote_sz})...")
        t0 = time.time()
        try:
            xnat.request("PUT", f"/data/projects/{PROJ_ID}/subjects/{sid}/resources/MEDEYE3D")
            uri = f"/data/projects/{PROJ_ID}/subjects/{sid}/resources/MEDEYE3D/files/{remote_h5}?overwrite=true"
            ok = xnat.upload_file_streaming(uri, h5_local)
            print(f"  [{pid}] Upload completed in {time.time() - t0:.1f}s (ok={ok})")
        except Exception as e:
            print(f"  [{pid}] Upload failed: {e}")

    print("All sync operations finished.")


if __name__ == "__main__":
    main()
