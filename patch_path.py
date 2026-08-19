import re
with open("scripts/ai/python_worker.py", "r") as f:
    text = f.read()

# Fix run_nninteractive
text = text.replace('pred_path = out_dir / "nninteractive_prediction.nii.gz"', 'out_dir = Path(out_dir)\n    out_dir.mkdir(parents=True, exist_ok=True)\n    pred_path = out_dir / "nninteractive_prediction.nii.gz"')

with open("scripts/ai/python_worker.py", "w") as f:
    f.write(text)
print("python_worker.py path fixed.")
