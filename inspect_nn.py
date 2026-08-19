import sys
import torch
sys.path.insert(0, "./pyenv/lib/python3.11/site-packages")
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
DEVICE = "cpu"
session = nnInteractiveInferenceSession(device=DEVICE)
try:
    session.initialize_from_trained_model_folder("/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/src/nnInteractive_model")
    print("Num channels expected by model:", session.num_channels)
except Exception as e:
    print("Error:", e)
