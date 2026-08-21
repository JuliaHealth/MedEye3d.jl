
import traceback
import sys
import os
import gc
from pathlib import Path

# Strict imports — no fallbacks. If nnInteractive is not installed, crash immediately.
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.model_management import ensure_model_available, get_default_model_id

import json
import numpy as np
import nibabel as nib
import torch
import socket
import threading


# Setup bundle paths for models
CURRENT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = CURRENT_DIR / "helpnet_bundle"
sys.path.insert(0, str(BUNDLE_DIR))

from model import HELPNet_Lesion

# Global Model states
HELPNET_MODEL = None
NN_INTERACTIVE_SESSION = None
DEVICE = None
INFERENCE_LOCK = threading.Lock()  # Serialize all inference calls (defense-in-depth)

CURRENT_LOADED_CT_PATH = None
CURRENT_LOADED_CT_SHAPE = None
CURRENT_LOADED_CT_AFFINE = None  # Store CT's NIfTI affine so predictions match

def init_models():
    global HELPNET_MODEL, DEVICE, NN_INTERACTIVE_SESSION
    
    import torch
    if torch.cuda.is_available():
        gpu_idx = int(os.environ.get("MEDEYE3D_GPU_INDEX", "0"))
        if gpu_idx >= torch.cuda.device_count():
            gpu_idx = 0
        DEVICE = torch.device(f"cuda:{gpu_idx}")
        print(f"[Worker] Initializing models on GPU: {torch.cuda.get_device_name(DEVICE)} (cuda:{gpu_idx})...")
    else:
        DEVICE = torch.device("cpu")
        print("[Worker] WARNING: No CUDA GPU detected, falling back to CPU...")
    
    # Load Helpnet — strict, no fallbacks
    HELPNET_MODEL = HELPNet_Lesion()
    ckpt = os.path.join(BUNDLE_DIR, "checkpoints", "helpnet_model_final.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"HELPNet checkpoint not found at {ckpt}. No fallbacks allowed.")
    HELPNET_MODEL.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    HELPNET_MODEL.to(DEVICE)
    HELPNET_MODEL.eval()
    print("[Worker] HELPNet Checkpoint loaded.")
    
    # Load nnInteractive Session — strict, no fallbacks
    print("[Worker] Initializing nnInteractive deep learning model...")
    model_dir = ensure_model_available(get_default_model_id())
    NN_INTERACTIVE_SESSION = nnInteractiveInferenceSession(device=DEVICE)
    NN_INTERACTIVE_SESSION.initialize_from_trained_model_folder(str(model_dir))
    print("[Worker] nnInteractive initialized.")
        
    print("[Worker] Models ready.")

def resolve_path(p, out_dir="/tmp/medeye3d_inference"):
    if not p:
        return None
    if os.path.exists(p):
        return str(p)
    if not p.endswith(".gz") and os.path.exists(p + ".gz"):
        return str(p + ".gz")
    
    fname = os.path.basename(p)
    fname_gz = fname if fname.endswith(".gz") else fname + ".gz"
    
    for cand_dir in [out_dir, "/tmp/medeye3d_inference"]:
        if not cand_dir:
            continue
        c1 = Path(cand_dir) / fname
        if c1.exists():
            return str(c1)
        c2 = Path(cand_dir) / fname_gz
        if c2.exists():
            return str(c2)
        if fname.endswith(".gz"):
            c3 = Path(cand_dir) / fname[:-3]
            if c3.exists():
                return str(c3)
                
    return str(Path(out_dir) / fname)

def run_helpnet(ct_path, pet_path, point_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ct_path = resolve_path(ct_path, out_dir)
    pet_path = resolve_path(pet_path, out_dir)
    point_path = resolve_path(point_path, out_dir)
    
    ct_nib = nib.load(ct_path)
    ct_vol = ct_nib.get_fdata().astype(np.float32)
    pet_nib = nib.load(pet_path)
    pet_vol = pet_nib.get_fdata().astype(np.float32)
    point_nib = nib.load(point_path)
    point_vol = point_nib.get_fdata().astype(np.float32)
    
    ct_mean = ct_vol.mean()
    ct_std = ct_vol.std() + 1e-6
    ct_norm = (ct_vol - ct_mean) / ct_std
    
    x = np.stack([ct_norm, pet_vol, point_vol], axis=0)
    x_tensor = torch.from_numpy(x).unsqueeze(0).float()
    
    if HELPNET_MODEL is None:
        raise RuntimeError("HELPNet model is not loaded. No fallbacks allowed.")
    
    with torch.no_grad():
        logits = HELPNET_MODEL(x_tensor.to(DEVICE))
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        print(f"[Worker] HELPNet deep learning model produced {np.sum(pred)} voxels.")
            
    pred_path = out_dir / "helpnet_prediction.nii.gz"
    nib.save(nib.Nifti1Image(pred, ct_nib.affine), str(pred_path))
    return str(pred_path)

def preload_ct(ct_path, out_dir="/tmp/medeye3d_inference"):
    """Preload CT into nnInteractive GPU memory for faster subsequent inference.
    Starts background preprocessing via set_image() without blocking for completion.
    """
    global NN_INTERACTIVE_SESSION, CURRENT_LOADED_CT_PATH, CURRENT_LOADED_CT_SHAPE, CURRENT_LOADED_CT_AFFINE
    
    if NN_INTERACTIVE_SESSION is None:
        raise RuntimeError("nnInteractive session is not initialized. No fallbacks allowed.")
    
    out_dir = Path(out_dir)
    ct_path = resolve_path(ct_path, out_dir)
    
    if CURRENT_LOADED_CT_PATH == ct_path:
        print(f"[Worker] CT already preloaded: {ct_path}")
        return
    
    print(f"[Worker] Preloading CT into GPU: {ct_path}")
    ct_nib = nib.load(ct_path)
    ct_vol = ct_nib.get_fdata()
    CURRENT_LOADED_CT_AFFINE = ct_nib.affine
    ct_vol_zyx = np.transpose(ct_vol, (2, 1, 0))
    image = ct_vol_zyx[np.newaxis, ...].astype(np.float32)
    # set_image offloads preprocessing to a background thread
    NN_INTERACTIVE_SESSION.set_image(image)
    CURRENT_LOADED_CT_PATH = ct_path
    CURRENT_LOADED_CT_SHAPE = ct_vol_zyx.shape
    # Don't call _finish_preprocessing here — let it run in background.
    # run_nninteractive will wait for it when needed.
    print(f"[Worker] CT preload initiated (preprocessing runs in background)")

def run_nninteractive(ct_path, point_path, out_dir, scribble_coords=None, autozoom=True, inline_result=False):
    global NN_INTERACTIVE_SESSION, CURRENT_LOADED_CT_PATH, CURRENT_LOADED_CT_SHAPE, CURRENT_LOADED_CT_AFFINE
    print("[Worker] Running NNInteractive (CT-only mode)...")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "nninteractive_prediction.nii.gz"
    
    if NN_INTERACTIVE_SESSION is None:
        raise RuntimeError("nnInteractive session is not initialized. No fallbacks allowed.")
    
    ct_path = resolve_path(ct_path, out_dir)
    
    # Check if CT is already cached (possibly by preload_ct)
    if CURRENT_LOADED_CT_PATH != ct_path:
        print(f"[Worker] Loading new CT scan into nnInteractive session: {ct_path}")
        ct_nib = nib.load(ct_path)
        ct_vol = ct_nib.get_fdata()
        CURRENT_LOADED_CT_AFFINE = ct_nib.affine
        ct_vol_zyx = np.transpose(ct_vol, (2, 1, 0))
        image = ct_vol_zyx[np.newaxis, ...].astype(np.float32)
        NN_INTERACTIVE_SESSION.set_image(image)
        CURRENT_LOADED_CT_PATH = ct_path
        CURRENT_LOADED_CT_SHAPE = ct_vol_zyx.shape
    else:
        print(f"[Worker] Reusing cached CT scan in nnInteractive session: {ct_path}")
        NN_INTERACTIVE_SESSION.reset_interactions()
    
    # Always wait for preprocessing to finish (handles both preload and inline cases)
    NN_INTERACTIVE_SESSION._finish_preprocessing_and_initialize_interactions()
    
    # Configurable autozoom — disable for faster interactive speed
    NN_INTERACTIVE_SESSION.set_do_autozoom(autozoom)
    
    target_zyx = np.zeros(CURRENT_LOADED_CT_SHAPE, dtype=np.uint8)
    NN_INTERACTIVE_SESSION.set_target_buffer(target_zyx)
    
    # Determine scribbles
    if scribble_coords is not None and len(scribble_coords) > 0:
        scribble_mask_zyx = np.zeros(CURRENT_LOADED_CT_SHAPE, dtype=bool)
        for pt in scribble_coords:
            x, y, z = pt[0], pt[1], pt[2]
            if 0 <= z < CURRENT_LOADED_CT_SHAPE[0] and 0 <= y < CURRENT_LOADED_CT_SHAPE[1] and 0 <= x < CURRENT_LOADED_CT_SHAPE[2]:
                scribble_mask_zyx[z, y, x] = True
    elif point_path:
        point_path = resolve_path(point_path, out_dir)
        if os.path.exists(point_path):
            point_vol = nib.load(point_path).get_fdata()
            point_vol_zyx = np.transpose(point_vol, (2, 1, 0))
            scribble_mask_zyx = point_vol_zyx > 0
        else:
            raise FileNotFoundError(f"Point/scribble file not found at {point_path}. No fallbacks allowed.")
    else:
        raise ValueError("Neither scribble_coords nor point_path provided for NNInteractive.")
        
    print(f"[Worker] Using MIC-DKFZ nnInteractive Model with {np.count_nonzero(scribble_mask_zyx)} scribble voxels (autozoom={autozoom})")
    NN_INTERACTIVE_SESSION.add_scribble_interaction(scribble_mask_zyx, include_interaction=True, run_prediction=True)
    
    total_voxels = int(np.sum(target_zyx > 0))
    
    if inline_result and total_voxels > 0:
        # Inline transfer: find tight bounding box of nonzero voxels, base64-encode sub-mask
        import base64
        nz = np.nonzero(target_zyx)
        if len(nz[0]) > 0:
            z1, z2 = int(nz[0].min()), int(nz[0].max()) + 1
            y1, y2 = int(nz[1].min()), int(nz[1].max()) + 1
            x1, x2 = int(nz[2].min()), int(nz[2].max()) + 1
            sub_mask_zyx = target_zyx[z1:z2, y1:y2, x1:x2]
            # Transpose to Julia XYZ order
            sub_mask_xyz = np.transpose(sub_mask_zyx, (2, 1, 0)).copy()
            mask_b64 = base64.b64encode(sub_mask_xyz.astype(np.uint8).tobytes()).decode('ascii')
            # Full shape in XYZ (Julia order)
            full_shape_xyz = [CURRENT_LOADED_CT_SHAPE[2], CURRENT_LOADED_CT_SHAPE[1], CURRENT_LOADED_CT_SHAPE[0]]
            sub_shape_xyz = list(sub_mask_xyz.shape)
            # Clipped bbox in ZYX for Julia to place the sub-mask
            clipped_bbox = [[z1, z2], [y1, y2], [x1, x2]]
            print(f"[Worker] nninteractive produced {total_voxels} voxels, inline transfer ({len(mask_b64)} b64 bytes, sub_shape={sub_shape_xyz}, bbox_zyx={clipped_bbox})")
            return {"inline": True, "mask_b64": mask_b64, "mask_shape": full_shape_xyz, "sub_shape": sub_shape_xyz, "bbox": clipped_bbox, "voxel_count": total_voxels}
    
    # Fallback: save as NIfTI file
    result_mask = np.transpose(target_zyx, (2, 1, 0))
    nib.save(nib.Nifti1Image(result_mask.astype(np.uint8), np.eye(4)), pred_path)
    print(f"[Worker] nninteractive produced {total_voxels} voxels, saved to {pred_path}")
    return {"inline": False, "pred_path": str(pred_path), "voxel_count": total_voxels}

def handle_client(conn):
    try:
        # Receive full message buffer
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if len(chunk) < 65536:
                break
        
        if not chunks:
            return
            
        data = json.loads(b"".join(chunks).decode('utf-8'))
        command = data.get("command", "")
        
        response = {}
        if command == "ping":
            response = {"status": "success", "message": "pong"}
        else:
            with INFERENCE_LOCK:
                if command == "helpnet":
                    pred_path = run_helpnet(data["ct_path"], data["pet_path"], data["point_path"], data["out_dir"])
                    response = {"status": "success", "prediction_path": pred_path}
                elif command == "nninteractive":
                    ct_p = data.get("ct_path", data.get("image_path"))
                    scribble_coords = data.get("scribble_coords")
                    autozoom = data.get("autozoom", True)
                    inline_result = data.get("inline_result", False)
                    result = run_nninteractive(ct_p, data.get("point_path"), data["out_dir"],
                                              scribble_coords=scribble_coords, autozoom=autozoom,
                                              inline_result=inline_result)
                    if result.get("inline"):
                        response = {"status": "success", "mask_b64": result["mask_b64"],
                                    "mask_shape": result["mask_shape"], "sub_shape": result["sub_shape"],
                                    "bbox": result["bbox"], "voxel_count": result["voxel_count"]}
                    else:
                        response = {"status": "success", "prediction_path": result["pred_path"]}
                elif command == "preload_ct":
                    ct_p = data.get("ct_path")
                    preload_ct(ct_p, data.get("out_dir", "/tmp/medeye3d_inference"))
                    response = {"status": "success", "message": "CT preload initiated"}
                else:
                    response = {"status": "error", "message": "Unknown command"}
            
        conn.sendall(json.dumps(response).encode('utf-8'))
    except Exception as e:
        traceback.print_exc()
        err = {"status": "error", "message": str(e)}
        conn.sendall(json.dumps(err).encode('utf-8'))
    finally:
        conn.close()

def run_server(port=5005):
    init_models()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(5)
    print(f"[Worker] TCP JSON Server listening on port {port}...")
    
    while True:
        conn, addr = server.accept()
        t = threading.Thread(target=handle_client, args=(conn,))
        t.start()

if __name__ == "__main__":
    run_server()
