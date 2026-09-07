#!/usr/bin/env python3
"""
run_pytorch_surface_registration.py

DEPRECATED: This script previously ran PyTorch-based surface registration.
It now delegates to landmark_registration.py which uses SVD-based landmark
matching (Arun et al. 1987) — much simpler, faster, and reproducible.

For all follow-up timepoints:
1. CT 1 -> CT 0: using hip_left, hip_right, prostate extremal points
2. CT 2 -> CT 0: using hip_left, hip_right, prostate extremal points
3. MRI 3 -> CT 0: using hip_left, hip_right, prostate (+ sacrum when available)
4-6. Copies of Transform 3 (MRI modalities are co-registered from same session)

Writes the resulting ITK transforms to:
- Transform_FollowUp_to_Baseline_1.tfm
- Transform_FollowUp_to_Baseline_2.tfm
- Transform_FollowUp_to_Baseline_3.tfm
- Transform_FollowUp_to_Baseline_4.tfm (copy of 3)
- Transform_FollowUp_to_Baseline_5.tfm (copy of 3)
- Transform_FollowUp_to_Baseline_6.tfm (copy of 3)
"""

import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    case_dir = os.path.join(base_dir, "data", "cases", "psma_patient_all_tp")
    landmark_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landmark_registration.py")

    print("=" * 60)
    print("  Delegating to landmark_registration.py (SVD-based)")
    print("=" * 60)

    subprocess.run(
        [sys.executable, landmark_script, "--case-dir", case_dir],
        check=True
    )

if __name__ == "__main__":
    main()
