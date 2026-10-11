# Heat-GDT: Differentiable Heat-Based Geodesic Distance Transform for Interactive 3D Lesion Segmentation

## Mathematical Formulation

The core of Heat-GDT relies on the following mathematical formulations:
- **PDE**: $\frac{\partial u}{\partial t} = \nabla \cdot (D(x) \nabla u)$
- **Diffusivity**: $D(x) = \epsilon_D + (1-\epsilon_D)\exp(-\gamma \cdot E(x))$
- **Mask extraction**: $\text{sigmoid}((u-\theta)/\tau)$
- Features an exact adjoint state backward pass for efficient and accurate gradient computation.

## Architecture

- **Champion**: HED3D + Pure Diffusion + Dice-BCE Loss
- **HED3D**: 5-stage VGG encoder, deep side supervision, weighted fusion
- **Parameters**: $K=40$, $dt=0.16$, $\theta=0.001$, $\tau=0.0001$

## Training Results

- **Dataset**: AutoPET-III (829 train, 208 val cases)
- **Best Val Dice**: 0.4287 (vs HELPNet 0.2925)
- **Early stopping** at epoch 11 (best at epoch 6)
- **Training**: cosine LR schedule, Adam optimizer, lr=1e-4

## File Structure

- `heat_diffuse.jl`: CUDA heat diffusion with ChainRulesCore adjoint
- `heat_diffuse_v2.jl`: Extended reaction-diffusion-advection solver
- `model.jl`: Edge network definitions
- `architectures/`: Contains models like HED3D, DexiNed3D, DFrameNet, Hybrid25D
- `losses.jl`: Loss functions including Dice, BCE, DTM boundary, Focal Tversky
- `train_champion.jl`: Training pipeline script
- `benchmark_champion.jl`: Evaluation script

## Quick Start

```julia
using Pkg; Pkg.activate("/path/to/JuliaHELPNet")
include("train_champion.jl")
train_champion()
```

## Citation

*Placeholder for the paper.*
