# Developer Playbook & Architecture Guide

## 1. Physical Aspect Ratio & Viewport Layout Architecture

MedEye3d visualizes 2D and 3D medical volumes (CT, PET, SPECT, MR) across multiple viewports:
- **`SingleImage`**: Full-window single panel (or maximized panel after double-click).
- **`MultiImage`**: 2 side-by-side timepoint comparison panels (Left: Pos 1, Right: Pos 2 or 5).
- **`QuadImage`**: 4-panel quadrant display (1: Top-Left Axial, 2: Top-Right Axial PET, 3: Bottom-Left Sagittal, 4: Bottom-Right Coronal).

### 1.1 Physical Anatomical Aspect Ratio
Unlike standard 2D image viewers where pixels are square ($1:1$), medical image voxels possess physical spacings $(\Delta_x, \Delta_y, \Delta_z)$ in millimeters (e.g. $0.976\,\text{mm} \times 0.976\,\text{mm} \times 3.0\,\text{mm}$).

When extracting a 2D slice along a slicing dimension:
- $\text{imageTextureWidth} = \text{pixels along horizontal axis } (W_{\text{px}})$
- $\text{imageTextureHeight} = \text{pixels along vertical axis } (H_{\text{px}})$
- $\text{heightToWithRatio} = \frac{\Delta_{\text{vertical}}}{\Delta_{\text{horizontal}}}$

The true physical aspect ratio in millimeters is:
$$\text{ratio\_desired} = \frac{\text{Physical Height (mm)}}{\text{Physical Width (mm)}} = \text{heightToWithRatio} \times \frac{\text{imageTextureHeight}}{\text{imageTextureWidth}}$$

### 1.2 Viewport Panel Partitioning & Aspect Ratio Calculation
Given total window dimensions $W_{\text{win}}, H_{\text{win}}$ and fraction of main image $\text{frac} = \text{fractionOfMainIm}$:
The available NDC width spans $X \in [-1.0, -1.0 + 2 \cdot \text{frac}]$, with NDC midpoint $X_{\text{mid}} = -1.0 + \text{frac}$.

| Display Mode | Panel Dimensions ($W_{\text{p}}, H_{\text{p}}$) | Panel Aspect Ratio ($\text{ratio\_actual}$) | Panel NDC Bounds ($[X_{\min}, X_{\max}] \times [Y_{\min}, Y_{\max}]$) |
| :--- | :---: | :---: | :--- |
| **`SingleImage`** | $W_{\text{win}} \cdot \text{frac} \times H_{\text{win}}$ | $\frac{H_{\text{win}}}{W_{\text{win}} \cdot \text{frac}}$ | $[-1.0, -1.0 + 2\text{frac}] \times [-1.0, 1.0]$ |
| **`MultiImage` (Left, Pos 1)** | $\frac{W_{\text{win}} \cdot \text{frac}}{2} \times H_{\text{win}}$ | $\frac{2 \cdot H_{\text{win}}}{W_{\text{win}} \cdot \text{frac}}$ | $[-1.0, X_{\text{mid}}] \times [-1.0, 1.0]$ |
| **`MultiImage` (Right, Pos 2/5)** | $\frac{W_{\text{win}} \cdot \text{frac}}{2} \times H_{\text{win}}$ | $\frac{2 \cdot H_{\text{win}}}{W_{\text{win}} \cdot \text{frac}}$ | $[X_{\text{mid}}, -1.0 + 2\text{frac}] \times [-1.0, 1.0]$ |
| **`QuadImage` (Top-Left, 1)** | $\frac{W_{\text{win}} \cdot \text{frac}}{2} \times \frac{H_{\text{win}}}{2}$ | $\frac{H_{\text{win}}}{W_{\text{win}} \cdot \text{frac}}$ | $[-1.0, X_{\text{mid}}] \times [0.0, 1.0]$ |
| **`QuadImage` (Top-Right, 2)** | $\frac{W_{\text{win}} \cdot \text{frac}}{2} \times \frac{H_{\text{win}}}{2}$ | $\frac{H_{\text{win}}}{W_{\text{win}} \cdot \text{frac}}$ | $[X_{\text{mid}}, -1.0 + 2\text{frac}] \times [0.0, 1.0]$ |
| **`QuadImage` (Bottom-Left, 3)** | $\frac{W_{\text{win}} \cdot \text{frac}}{2} \times \frac{H_{\text{win}}}{2}$ | $\frac{H_{\text{win}}}{W_{\text{win}} \cdot \text{frac}}$ | $[-1.0, X_{\text{mid}}] \times [-1.0, 0.0]$ |
| **`QuadImage` (Bottom-Right, 4)** | $\frac{W_{\text{win}} \cdot \text{frac}}{2} \times \frac{H_{\text{win}}}{2}$ | $\frac{H_{\text{win}}}{W_{\text{win}} \cdot \text{frac}}$ | $[X_{\text{mid}}, -1.0 + 2\text{frac}] \times [-1.0, 0.0]$ |

### 1.3 Universal Scale & Centering Engine
Inside [`StructsManag.getMainVerticies`](file:///mnt/big/project_ssd/project_ssd/MedEye3d.jl/src/display/GLFW/DispUtils/StructsManag.jl), scaling is computed by comparing the physical desired ratio to the panel's actual pixel ratio:

- If $\text{ratio\_actual} > \text{ratio\_desired}$ (panel is taller than image):
  $$\text{scale}_y = \frac{\text{ratio\_desired}}{\text{ratio\_actual}}, \quad \text{scale}_x = 1.0$$
- If $\text{ratio\_actual} \le \text{ratio\_desired}$ (panel is wider than image):
  $$\text{scale}_x = \frac{\text{ratio\_actual}}{\text{ratio\_desired}}, \quad \text{scale}_y = 1.0$$

The centered quad vertices in OpenGL Normalized Device Coordinates (NDC) are:
$$X_{\text{center}} = \frac{X_{\min} + X_{\max}}{2}, \quad \text{half}_x = \frac{X_{\max} - X_{\min}}{2} \cdot \text{scale}_x \implies [X_{\text{center}} - \text{half}_x, X_{\text{center}} + \text{half}_x]$$
$$Y_{\text{center}} = \frac{Y_{\min} + Y_{\max}}{2}, \quad \text{half}_y = \frac{Y_{\max} - Y_{\min}}{2} \cdot \text{scale}_y \implies [Y_{\text{center}} - \text{half}_y, Y_{\text{center}} + \text{half}_y]$$

This guarantees:
1. **Zero Wasted Vertical Space**: When viewports are wider than tall (the standard condition), $\text{scale}_y = 1.0$, which means top panels extend directly to $Y = 1.0$ at the top of the OpenGL viewport without unnecessary black gaps.
2. **True Proportions**: Anatomical structures (e.g. bones, pelvis, tumors) remain strictly undistorted across all layouts and on all window resize events.
