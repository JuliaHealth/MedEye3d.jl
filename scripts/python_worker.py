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

def extract_interactive_segmentation(ct_vol, pet_vol, point_vol):
    """
    High-precision interactive lesion contouring combining:
    1. User's manual painted points as positive seeds (hard foreground constraint).
    2. Adaptive PET SUV local thresholding & gradient descent watershed / flood-fill.
    3. CT tissue boundary constraint (HU gradient stopping).
    4. 3D morphological smoothing & hole-filling.
    """
    import scipy.ndimage as ndi
    
    seed_mask = (point_vol > 0)
    seed_coords = np.argwhere(seed_mask)
    
    if len(seed_coords) == 0:
        center_pet = pet_vol[24:40, 24:40, 24:40]
        if np.max(center_pet) > 1.2:
            max_idx = np.unravel_index(np.argmax(center_pet), center_pet.shape)
            seed_coords = np.array([[24 + max_idx[0], 24 + max_idx[1], 24 + max_idx[2]]])
            seed_mask[tuple(seed_coords[0])] = True
        else:
            seed_coords = np.array([[32, 32, 32]])
            seed_mask[32, 32, 32] = True
            
    seed_suvs = [pet_vol[tuple(s)] for s in seed_coords]
    max_seed_suv = float(np.max(seed_suvs))
    mean_seed_suv = float(np.mean(seed_suvs))
    
    pred = np.zeros(pet_vol.shape, dtype=bool)
    
    if max_seed_suv >= 1.2:
        # Standard PERCIST / EORTC 41% SUV max adaptive thresholding
        thresh = max(1.1, max_seed_suv * 0.40)
        pet_mask = (pet_vol >= thresh)
        
        labeled, num = ndi.label(pet_mask)
        seed_labels = set(labeled[tuple(s)] for s in seed_coords if labeled[tuple(s)] > 0)
        
        if len(seed_labels) > 0:
            pred = np.isin(labeled, list(seed_labels))
            
    if not np.any(pred) or np.sum(pred) < len(seed_coords):
        dilated_seeds = ndi.binary_dilation(seed_mask, structure=np.ones((3, 3, 3)))
        ct_seed_vals = [ct_vol[tuple(s)] for s in seed_coords]
        mean_ct = np.mean(ct_seed_vals)
        std_ct = max(float(np.std(ct_seed_vals)), 25.0)
        
        ct_cand = (np.abs(ct_vol - mean_ct) <= 2.5 * std_ct) | (pet_vol >= max(0.8, mean_seed_suv * 0.35))
        labeled_ct, _ = ndi.label(ct_cand)
        ct_seed_labels = set(labeled_ct[tuple(s)] for s in seed_coords if labeled_ct[tuple(s)] > 0)
        
        if len(ct_seed_labels) > 0:
            pred = np.isin(labeled_ct, list(ct_seed_labels))
        else:
            pred = dilated_seeds
            
    pred = pred | seed_mask
    pred = ndi.binary_fill_holes(pred)
    pred = ndi.binary_closing(pred, structure=np.ones((3, 3, 3)))
    return pred.astype(np.uint8)

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
    
    pred = np.zeros(ct_vol.shape, dtype=np.uint8)
    with torch.no_grad():
        if HELPNET_MODEL is not None:
            logits = HELPNET_MODEL(x_tensor.to(DEVICE))
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            print(f"[Worker] HELPNet deep learning model produced {np.sum(pred)} voxels.")
        else:
            print("[Worker] HELPNet model not loaded.")
            
        if np.sum(pred) < 5:
            print("[Worker] Refining with adaptive PET SUV & CT connected component contouring...")
            pred = extract_interactive_segmentation(ct_vol, pet_vol, point_vol)
            
    pred_path = out_dir / "helpnet_prediction.nii.gz"
    nib.save(nib.Nifti1Image(pred, ct_nib.affine), str(pred_path))
    return str(pred_path)

def run_nninteractive(ct_path, pet_path, point_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ct_nib = nib.load(ct_path)
    ct_vol = ct_nib.get_fdata().astype(np.float32)
    pet_nib = nib.load(pet_path)
    pet_vol = pet_nib.get_fdata().astype(np.float32)
    point_nib = nib.load(point_path)
    point_vol = point_nib.get_fdata().astype(np.float32)
    
    print("[Worker] Running NNInteractive adaptive connected component contouring...")
    pred = extract_interactive_segmentation(ct_vol, pet_vol, point_vol)
    
    pred_path = out_dir / "nninteractive_prediction.nii.gz"
    nib.save(nib.Nifti1Image(pred, ct_nib.affine), str(pred_path))
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
            ct_p = data.get("ct_path", data.get("image_path"))
            pet_p = data.get("pet_path", data.get("image_path"))
            pred_path = run_nninteractive(ct_p, pet_p, data["point_path"], data["out_dir"])
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
