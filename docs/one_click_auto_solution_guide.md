# One-Click Auto Parameter & Size Solution in MedEye3d

## 1. Overview & Motivation

In volumetric interactive lesion segmentation, manual parameter tuning ($\theta$, $K$, edge sensitivity, contrast boost) imposes cognitive load on the clinician and introduces inter-observer variability. Furthermore, lesions vary radically in volume (from $30\text{ mm}^3$ lymph nodes to $>50,000\text{ mm}^3$ hepatic metastases), anatomical compartment (soft tissue, bone, subdiaphragmatic, mesenteric), and morphology (spherical, lobulated, or curved tubular channels such as ureters and vessels).

To eliminate manual parameter hunting, **MedEye3d** provides an end-to-end **One-Click Solution** (`1-Click` mode) that automatically determines optimal diffusion parameters, edge normalizations, and physical boundary confinement upon a single click on a lesion seed point.

```
                  Clinician Single Left Click at Seed (cx, cy, cz)
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │ 1. Radial Wall & Geometry Sensing:     │
                     │    • Ray tracing along 6 cardinal axes │
                     │    • Distinguish Spherical vs Tubular  │
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │ 2. Automated Parameter Selection:      │
                     │    • Dynamic radius: R_phys(K, sp)     │
                     │    • Adaptive Euler steps: K_eff       │
                     │    • Dynamic barrier: E_barrier        │
                     │    • Frontier contrast boost: s_boost  │
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │ 3. 6-Sector Cardinal Normalization:    │
                     │    • Spherical partition of unity      │
                     │    • Zero strong-edge shadowing        │
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │ 4. GPU Vulkan Compute Dispatch:        │
                     │    • Continuous Dirichlet seed source  │
                     │    • Harmonic mean face conductance    │
                     │    • Instant mask commit (<15 ms)      │
                     └────────────────────┬───────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────┐
                     │ 5. Live Post-Hoc Sizing (Size ±):      │
                     │    • Non-destructive live thresholding │
                     │    • Instantaneous GPU volume growth   │
                     └────────────────────────────────────────┘
```

---

## 2. Core Components of the Auto-Choosing Engine

### 2.1 Automated Bounding Radius & Step Scaling
The effective physical diffusion front radius $R_{\text{phys}}$ is automatically calculated from the physical voxel spacing $\mathbf{s} = (\Delta x, \Delta y, \Delta z)$ and the base step count $K$:

$$\bar{s} = \frac{\Delta x + \Delta y + \Delta z}{3}$$

$$R_{\text{phys}} = \max\left(10.0\text{ mm}, \; 3.0 + 2.0 \cdot \sqrt{K \cdot \Delta t} \cdot \bar{s}\right)$$

For 1D elongated tubular structures (e.g., ureters or vessels), standard spherical step counts stall before traversing the full channel. The engine detects channel resistance and automatically scales iterations:

$$K_{\text{eff}} = \begin{cases} \max(K, 140) & \text{if tubular channel confirmed} \\ K & \text{otherwise} \end{cases}$$

---

### 2.2 Frontier Shell Sampling ($s_{\text{boost}}$)
To sharpen subtle margins (such as soft-tissue metastases or lymph node capsules) without choking internal core flow:
1. The host extracts the outer shell band $[R_{\text{phys}} - 6.0\text{ mm}, \; R_{\text{phys}}]$ in the diffusivity tensor $D(\mathbf{x})$.
2. The mean frontier edge strength $\bar{E}_{\text{frontier}}$ is evaluated:
   $$\bar{E}_{\text{frontier}} = \frac{1}{|\mathcal{S}|} \sum_{\mathbf{x} \in \mathcal{S}} \left(-\frac{\ln(\max(D(\mathbf{x}), 10^{-7}))}{\gamma}\right)$$
3. The adaptive contrast boost exponent $s_{\text{boost}}$ is evaluated:
   $$s_{\text{boost}} = \text{clamp}\left( \left(\frac{E_{\text{ref}}}{\max(\bar{E}_{\text{frontier}}, 0.01)}\right)^{\text{sensitivity}}, \; 0.5, \; 10.0 \right)$$
   where $E_{\text{ref}} = 0.20$ is the nominal reference boundary edge strength.

---

### 2.3 6-Sector Cardinal Directional Normalization
Tumors frequently abut distinct anatomical compartments (e.g., a liver dome tumor bordering lung air cranially with $E \approx 0.85$ and liver parenchyma caudally with $E \approx 0.16$). Under global scalar normalization, the strong edge shadows and disables boundary detection in all other directions.

The one-click engine evaluates independent local peak edge strengths across the 6 cardinal directions:
$$\mathcal{D} = \{+X, -X, +Y, -Y, +Z, -Z\}$$

In the Vulkan compute shader, displacement $\mathbf{d} = (\mathbf{x} - \mathbf{x}_{\text{seed}}) \odot \mathbf{s}$ is projected onto the unit sphere using a smooth quadratic partition of unity:
$$w_k(\mathbf{x}) = \max(\pm \hat{d}_i, 0)^2, \quad \sum_{k \in \mathcal{D}} w_k(\mathbf{x}) \equiv 1.0$$

$$E_{\text{peak}}(\mathbf{x}) = \sum_{k \in \mathcal{D}} w_k(\mathbf{x}) \cdot E_{\text{peak}}^k$$

$$E_{\text{rel}}(\mathbf{x}) = \text{clamp}\left( \frac{E(\mathbf{x}) - E_{\text{bg}}}{\max(E_{\text{peak}}(\mathbf{x}) - E_{\text{bg}}, 0.02)}, \; 0.0, \; 1.0 \right)$$

- **Interior Core** ($E \le E_{\text{bg}}$): $E_{\text{rel}} = 0.0 \implies D_{\text{eff}} = 1.0$ (100% full propagation speed).
- **Advancing Boundary** ($E \to E_{\text{peak}}$): $E_{\text{rel}} \to 1.0 \implies D_{\text{eff}} = 10^{-5}$ (impenetrable barrier).

---

### 2.4 Dirichlet Continuous Heat Source at Seed
In standard initial-impulse diffusion ($u(0) = 1.0$ at a single voxel), heat is conserved ($\sum u = 1.0$). For large lesions ($V > 5,000$ voxels), the equilibrium temperature drops below $\theta = 0.001$, prematurely stopping propagation at the seed diamond.

The MedEye3d Vulkan pipeline enforces a **Dirichlet continuous heat source** at every step $k \in [1, K]$:
$$\left. u(\mathbf{x}, t + \Delta t) \right|_{\|\mathbf{x} - \mathbf{x}_{\text{seed}}\|_{\text{phys}} \le r_{\text{seed}}} = 1.0$$

This guarantees continuous outward flux, enabling the heat wavefront to reach equilibrium throughout large tumors while stopping strictly at outer boundaries.

---

### 2.5 Live Post-Hoc Sizing (`Size ±` Slider)
After a 1-click segmentation commits, the clinician can immediately adjust the lesion boundary volume using the **`Size ±`** slider without re-running the diffusion:
- The GPU retains the 3D scalar heat field $u(\mathbf{x}, t_K)$ in GPU VRAM (`img_u_pong`).
- Dragging the slider dynamically re-evaluates the threshold:
  $$\theta_{\text{eff}} = \theta \cdot 10^{-\text{slider\_value}}$$
- The discrete mask is updated in $<1\text{ ms}$ on the GPU, giving the user tactile real-time control over boundary tightness.

---

## 3. How to Enable & Use the One-Click Solution in MedEye3d

### 3.1 GUI Activation
1. In the 3D viewer sidebar, expand the **Segmentation & AI** panel.
2. In the **Mode** toggle row, select **`[ 1-Click ]`** (active by default).
3. Ensure **`Adapt`** is toggled **`[ON]`** (enables automated contrast boost and directional normalization).
4. Left-click anywhere inside the lesion volume. The segmentation mask is computed and committed instantly.

```
┌────────────────────────────────────────────────────────┐
│ [▼ Segmentation & AI]                                  │
│ AI: [ Heat-GDT              ▼ ] [ Run AI ]             │
│ Mode: [ 1-Click ★ ] [ Wand (Hold) ]                    │
│ θ: [───●──────────────────────] K: [──────●──────────] │
│ Size ±: [──────────●───────────] [ Reset ]             │
│ Edge: [ Both (Combined) ] [ CT Only ] [ PET Only ]     │
│ Adapt: [ON ★]   Sens: [────●───]                       │
└────────────────────────────────────────────────────────┘
```

### 3.2 Keyboard Shortcut
- Press **`W`** anywhere in the viewer.
- This immediately selects **Heat-GDT**, activates **1-Click Mode**, primes the GPU compute pipeline, and prepares a new lesion label.

### 3.3 Guidance Modality Selection
The one-click engine operates across three distinct edge representations:
- **`[ Both ]`**: Fuses CT anatomical structures with PET metabolic boundaries (recommended for FDG and PSMA PET/CT).
- **`[ PET Only ]`**: Pure metabolic edge guidance (eliminates CT bone/contrast artifacts for soft-tissue or necrotic tumors).
- **`[ CT Only ]`**: Pure anatomical edge guidance (ideal for non-contrast or diagnostic pure CT scans).

---

## 4. API Reference (Julia / MedEye3d)

To execute one-click Heat-GDT programmatically from Julia:

```julia
using MedEye3d

# Execute 1-Click Heat-GDT at seed (cx, cy, cz)
success = MedEye3d.execute_one_click_heatgdt!(
    stateObjects, cx, cy, cz;
    K = nothing,          # Uses adaptive K (defaults to 40, auto-scales to 140 for tubular)
    theta = 0.001f0,      # Base threshold
    active_id = 1         # Target lesion label ID
)
```
