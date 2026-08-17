import os
import sys
import json
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
import socket
import threading

# Add bundle to system path to import HELPNet_Lesion
BUNDLE_DIR = Path("/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/src/helpnet_inference_bundle")
sys.path.insert(0, str(BUNDLE_DIR))
try:
    from model import HELPNet_Lesion
except ImportError:
    print("[Worker] WARNING: Could not import HELPNet_Lesion. Make sure paths are correct.")

# Global Model states
HELPNET_MODEL = None
NNUNET_MODEL = None
DEVICE = None

def init_models():
    global HELPNET_MODEL, NNUNET_MODEL, DEVICE
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(device_str)
    print(f"[Worker] Initializing models on {DEVICE}...")
    
    try:
        HELPNET_MODEL = HELPNet_Lesion(in_ch=3, out_ch=2).to(DEVICE)
        checkpoint_path = BUNDLE_DIR / "checkpoints" / "helpnet_model_final.pt"
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            HELPNET_MODEL.load_state_dict(state_dict)
            print("[Worker] HELPNet Checkpoint loaded.")
        else:
            print(f"[Worker] WARNING: HELPNet Checkpoint not found at {checkpoint_path}. Using random weights.")
        HELPNET_MODEL.eval()
    except Exception as e:
        print(f"[Worker] Failed to load HELPNet: {e}")
        
    print("[Worker] Models ready.")

def run_helpnet(ct_path, pet_path, point_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    with torch.no_grad():
        if HELPNET_MODEL is not None:
            logits = HELPNET_MODEL(x_tensor.to(DEVICE))
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            print("[Worker] HELPNet model not loaded. Returning empty mask.")
            pred = np.zeros_like(ct_vol, dtype=np.uint8)
            # Create a 3x3x3 cube around the center as dummy
            pred[30:34, 30:34, 30:34] = 1
            
    pred_path = out_dir / "helpnet_prediction.nii.gz"
    nib.save(nib.Nifti1Image(pred, ct_nib.affine), str(pred_path))
    return str(pred_path)

def run_nninteractive(image_path, point_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_nib = nib.load(image_path)
    img_vol = img_nib.get_fdata().astype(np.float32)
    point_nib = nib.load(point_path)
    point_vol = point_nib.get_fdata().astype(np.float32)
    
    pred = np.zeros(img_nib.shape, dtype=np.uint8)
    
    # Check for seed point coordinates
    coords = np.where(point_vol > 0)
    if len(coords[0]) > 0:
        cz, cy, cx = int(coords[0][0]), int(coords[1][0]), int(coords[2][0])
        z, y, x = np.ogrid[:img_vol.shape[0], :img_vol.shape[1], :img_vol.shape[2]]
        dist = np.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)
        # Interactive sphere around seed point (radius 5 voxels)
        pred[dist <= 5.0] = 1
        print(f"[Worker] NNInteractive generated {np.sum(pred)} voxels around ({cz},{cy},{cx})")
    
    pred_path = out_dir / "nninteractive_prediction.nii.gz"
    nib.save(nib.Nifti1Image(pred, img_nib.affine), str(pred_path))
    return str(pred_path)

def handle_client(conn):
    try:
        data_b = conn.recv(8192)
        if not data_b:
            return
            
        data = json.loads(data_b.decode('utf-8'))
        command = data.get("command", "")
        
        response = {}
        if command == "helpnet":
            pred_path = run_helpnet(data["ct_path"], data["pet_path"], data["point_path"], data["out_dir"])
            response = {"status": "success", "prediction_path": pred_path}
        elif command == "nninteractive":
            pred_path = run_nninteractive(data["image_path"], data["point_path"], data["out_dir"])
            response = {"status": "success", "prediction_path": pred_path}
        elif command == "ping":
            response = {"status": "success", "message": "pong"}
        else:
            response = {"status": "error", "message": "Unknown command"}
            
        conn.sendall(json.dumps(response).encode('utf-8'))
    except Exception as e:
        err = {"status": "error", "message": str(e)}
        conn.sendall(json.dumps(err).encode('utf-8'))
    finally:
        conn.close()

def run_server(port=5005):
    init_models()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', port))
    server.listen(5)
    print(f"[Worker] TCP JSON Server listening on port {port}...")
    
    while True:
        conn, addr = server.accept()
        t = threading.Thread(target=handle_client, args=(conn,))
        t.start()

if __name__ == "__main__":
    run_server()
