import re
with open("scripts/ai/python_worker.py", "r") as f:
    text = f.read()

init_str = """def init_models():
    global HELPNET_MODEL, DEVICE, NN_INTERACTIVE_SESSION
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    try:
        import torch
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Worker] Initializing models on cuda...")
        
        # Load Helpnet
        try:
            HELPNET_MODEL = HELPNet_Lesion()
            ckpt = os.path.join(BUNDLE_DIR, "checkpoints", "helpnet_model_final.pt")
            if os.path.exists(ckpt):
                HELPNET_MODEL.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                HELPNET_MODEL.to(DEVICE)
                HELPNET_MODEL.eval()
                print("[Worker] HELPNet Checkpoint loaded.")
            else:
                print(f"[Worker] WARNING: HELPNet checkpoint not found at {ckpt}")
        except Exception as e:
            print(f"[Worker] HELPNet init failed: {e}")
        
        # Load nnInteractive Session
        if 'nnInteractiveInferenceSession' in globals():
            print("[Worker] Initializing nnInteractive deep learning model...")
            model_dir = ensure_model_available(get_default_model_id())
            NN_INTERACTIVE_SESSION = nnInteractiveInferenceSession(device=DEVICE)
            NN_INTERACTIVE_SESSION.initialize_from_trained_model_folder(str(model_dir))
            print("[Worker] nnInteractive initialized.")
        else:
            print("[Worker] WARNING: nnInteractive module not found! Using algorithmic fallback.")
            
        print("[Worker] Models ready.")
    except Exception as e:
        print(f"[Worker] Model initialization failed: {e}")"""

text = re.sub(r'def init_models\(\):.*?print\("\[Worker\] Models ready\."\)\n    except Exception as e:\n        print\(f"\[Worker\] Model initialization failed: \{e\}"\)', init_str, text, flags=re.DOTALL)

with open("scripts/ai/python_worker.py", "w") as f:
    f.write(text)
print("init_models fixed.")
