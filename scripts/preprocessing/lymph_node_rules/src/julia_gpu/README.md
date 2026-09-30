# Julia GPU Rules Engine

This directory contains the highly-optimized Julia GPU implementation of the Lymph Node Lymphatic Rule Engine. The goal of this module is to identically reproduce the Python gold standard output while operating orders of magnitude faster using fused CUDA mega-kernels.

---

## 🚀 Mega-Kernel Architecture & Pipeline Plan

To achieve maximal VRAM efficiency and execution speed, the Julia pipeline processes rules strictly on a **per-voxel basis** across the entire volume, orchestrating rule evaluation through a multi-pass mega-kernel driven by the Directed Acyclic Graph (DAG) dependencies of the rules.

### The 6-Step Execution Flow:
1. **Input Collection**: Extract all primary organ masks and dependencies required by any JSON rule.
2. **Optimal 4D Tensor Packing**:
   - Instead of maintaining N boolean masks (which consumes excessive memory), non-overlapping binary masks are mapped to integer IDs within a shared 3D channel.
   - For example: if 3 masks do not spatially overlap, they are packed into channel 1 (mask A=1, mask B=2, mask C=3).
   - If masks overlap, they are pushed to an additional channel. This produces a dense **4D Tensor** with the smallest possible channel dimension.
3. **CPU-to-GPU Instruction Matrix (Step N Execution)**:
   - For a given DAG level (all rules whose dependencies are fully resolved), the CPU prepares a lightweight instruction matrix.
   - Each row represents an operation. Columns represent arguments: `[OpCode, InputChannel_1, InputValue_1, DistanceArg, DirectionArg, OutputChannel, OutputValue, ...]`.
   - The GPU Mega-Kernel loops over this matrix inside a single per-voxel thread. Using `if/elseif` op-code branching, the kernel performs all independent area generations in a single parallel sweep over the 3D volume, writing results to a pre-allocated binary output tensor buffer.
4. **Iterative Re-packing**:
   - The binary output tensor from Step 3 is merged back into the packed 4D tensor (using the optimal non-overlapping integer assignment method from Step 2).
5. **Next DAG Level Execution**:
   - The pipeline advances to the next DAG level (rules that depend on the masks just generated in Step 3).
   - The CPU prepares the next instruction matrix, and the mega-kernel fires again, leveraging the newly populated channels in the 4D tensor.
6. **Completion**:
   - This iterative process continues until all DAG levels are processed. Because many areas are independent, the entire 67-area pipeline is executed in a highly compressed number of GPU kernel launches.

---

## ✅ Current Validation State

- **Synthetic Rule Exhaustive Testing (100% Parity):**
  A comprehensive suite of **46 test cases** verifying all rule varieties (`AnisotropicMargin`, `GeometricConstraint`, `PrimaryVector`, `Morphology`, Boolean operations, slice-wise flags, Connected Components) was run comparing the Python engine vs the Julia GPU engine.
  **Result:** 46/46 tests achieved `Dice ≥ 0.95`, with 44 tests achieving exact **1.000000** voxel-for-voxel identity.

- **Per-Area Pipeline Execution:**
  The `DagVm.jl` currently loads `primary_masks.h5`, resolves JSON DAG dependencies, packs masks, and runs Julia implementations per-area. Ongoing optimization continues to fuse these into the final mega-kernel format outlined above.

---

## 📂 Repository Structure

The repository has been cleaned and reorganized to separate the legacy Python reference code from the active Julia engine:

```
src/
├── julia_gpu/                   # Current Julia Mega-Kernel Engine
│   ├── main/                    # DagVm, RuleExecutors, MegaKernel logic
│   ├── README.md                # This document
│   └── ...
├── legacy_python/               # Gold Standard Python Implementation (Read-Only)
│   ├── rules/
│   ├── engine.py
│   ├── run_final_validation.py  # Script to generate MRB references
│   └── ...
└── anatomy_segmentation/        # Deep learning models & segmentation logic

data/                            # HDF5 caches, patient MRBs, and NRRD segmentations
old/                             # Deprecated src_v2 code, temporary logs, and clutter
tests/                           # Python and Julia test scripts, including the 46 synthetic suites
```

---

## 🛠 Isolated Testing Template (< 60s per Area)

Always use this isolated pattern when testing single areas to avoid full pipeline overhead:

```julia
using JSON, HDF5, NPZ, KernelAbstractions, CUDA
include("src/julia_gpu/main/DagVm.jl")
using .DagVm

h5 = h5open("data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44/primary_masks.h5", "r")
# Load mask -> Pack -> Execute specific Mega-Kernel Instruction
# ...
close(h5)
```

---

## 🛑 Strict Engineering Constraints

1. **DO NOT MODIFY ANY JSON FILES IN `jsons/`** — They are the single source of truth.
2. **DO NOT MODIFY ANY PYTHON FILES IN `src/legacy_python/`** — Read-only gold standard.
3. **ALWAYS USE `CUDA_VISIBLE_DEVICES=1`** — GPU 1 has dedicated 24GB VRAM.
4. **ALWAYS PERMUTE DIMS BEFORE SAVING TO NUMPY**: `npzwrite("...", permutedims(mask, (3, 2, 1)))`.
5. **NEVER RUN FULL PIPELINE FOR DEBUGGING** — Always use isolated scripts (< 60 seconds).

---

## ⚡ Quick Reference Commands

```bash
# Verify Gold Standard MRB Reference (runs pipeline + comparisons)
bash src/legacy_python/reproduce_pat44.sh

# Check JSON Integrity (Must return 0)
find jsons/ -name "*.json" -newermt "2026-08-23T17:00" | wc -l
```
