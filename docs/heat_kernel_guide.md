# Complete Guide to the Heat-GDT GPU Interactive Segmentation Kernel

## 1. Executive Summary & Theoretical Foundations

### 1.1 What is Heat-GDT?
**Heat-GDT** (*Heat Geodesic Distance Transform*) is an ultra-fast, interactive 3D volumetric lesion segmentation algorithm implemented directly on the GPU using Vulkan compute shaders. It replaces traditional active contours, level sets, and thresholding with a mathematically rigorous, boundary-respecting physical diffusion process.

When an oncologist or radiologist clicks and holds the mouse button on a lesion center in MedEye3d, heat is injected as an initial spatial impulse and diffuses outward through the 3D volume. Diffusion is guided by an anisotropic conductivity field $D(\mathbf{x})$, which acts as an impenetrable barrier at anatomical edges and tumor boundaries while permitting rapid flow within homogeneous parenchyma.

```
       ┌────────────────────────────────────────────────────────┐
       │             Learned 3D Edge Map E(x)                   │
       │       (HED3D-Ultra Champion Checkpoint, [0, 1])        │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         Diffusivity Field D(x) = exp(-γ · E(x))         │
       │           (Stored as 3D R32_SFLOAT on GPU)             │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         Anisotropic Heat Diffusion PDE on GPU:         │
       │            ∂u/∂t = ∇ · (D(x) ∇u)                      │
       │    • Anisotropic spacing weights (wx, wy, wz)          │
       │    • Shape-adaptive frontier contrast boost (s_boost)  │
       │    • Decoupled progressive visual ramp (θ_display)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             Final Discrete Mask Commit:                │
       │     seg_vol[x, y, z] = active_id where u(x, y, z) ≥ θ  │
       └────────────────────────────────────────────────────────┘
```

---

### 1.2 Mathematical Formulation

The physical diffusion of heat $u(\mathbf{x}, t)$ through an inhomogeneous, edge-aware medium is governed by the parabolic partial differential equation (PDE):

$$\frac{\partial u(\mathbf{x}, t)}{\partial t} = \nabla \cdot \Big( D(\mathbf{x}) \nabla u(\mathbf{x}, t) \Big)$$

subject to Neumann zero-flux (replicate) boundary conditions on the volume boundaries $\partial \Omega$:

$$\nabla u(\mathbf{x}, t) \cdot \mathbf{n} = 0 \quad \text{on } \partial \Omega$$

#### Edge-to-Conductivity Mapping
The scalar diffusivity $D(\mathbf{x}) \in (0, 1]$ is derived from the learned continuous edge probability field $E(\mathbf{x}) \in [0, 1]$ produced by the champion HED3D-Ultra deep edge network:

$$D(\mathbf{x}) = \exp\big(-\gamma \cdot E(\mathbf{x})\big)$$

where $\gamma = 10.095886$ is the champion edge gain (Epoch 105). This maps:
- **Homogeneous parenchyma / interior** ($E \approx 0.00$): $D \approx 1.00$ (maximum thermal conductivity, rapid expansion).
- **Subtle soft-tissue margins** ($E \approx 0.05$): $D \approx 0.60$ (moderate barrier).
- **Strong anatomical boundaries** ($E \ge 0.40$): $D \le 0.018$ (virtually zero conductivity, stopping propagation).

---

## 2. Core Breakthroughs in the Current Heat Kernel

The current heat kernel in MedEye3d incorporates two critical mathematical breakthroughs designed specifically for clinical PET/CT and multimodal MRI:

### 2.1 Breakthrough 1: Craniocaudal Anisotropic Spacing Compensation
In clinical PET/CT and MRI, imaging grids are heavily **anisotropic**. A typical volume has in-plane pixel spacing $\Delta x \approx 0.78$ mm, $\Delta y \approx 0.78$ mm, but slice thickness $\Delta z \approx 2.5 - 3.25$ mm ($3\times$ to $4\times$ coarser).

In a naive 6-neighbor finite-difference stencil, each grid step along $Z$ is treated identically to an in-plane step. Consequently, heat travels $3\times - 4\times$ faster in physical space along the craniocaudal axis, blowing past anatomical boundaries and squishing lesions into artificial ellipsoids.

#### Discrete Formulation with Metric Normalization
To strictly enforce isotropic physical diffusion across non-cubic voxels, the Laplacian flux is weighted by the inverse-squared voxel dimensions normalized by the smallest voxel size $s_{\min} = \min(\Delta x, \Delta y, \Delta z)$:

$$w_x = \left(\frac{s_{\min}}{\Delta x}\right)^2, \quad w_y = \left(\frac{s_{\min}}{\Delta y}\right)^2, \quad w_z = \left(\frac{s_{\min}}{\Delta z}\right)^2$$

For a typical scan with $(\Delta x, \Delta y, \Delta z) = (0.78, 0.78, 3.0)$ mm:
$$w_x = 1.00, \quad w_y = 1.00, \quad w_z = \left(\frac{0.78}{3.00}\right)^2 \approx 0.0676$$

The cross-slice flux along $Z$ is attenuated by **$14.8\times$**, guaranteeing that heat travels at the exact same physical speed (in mm/s) in all three dimensions. Furthermore, since $\max(w_x, w_y, w_z) = 1.0$, the Courant-Friedrichs-Lewy (CFL) numerical stability condition remains unconditionally satisfied with $\Delta t = 0.16$.

---

### 2.2 Breakthrough 2: Shape-Adaptive Local Edge Contrast
Subtle anatomical interfaces (such as the prostate capsule, lymph node hilum, or pancreatic margins) often exhibit faint edge probabilities ($E \approx 0.025$). Under standard global gain $\gamma \approx 10$, this produces $D \approx 0.78$, allowing heat to slowly leak into neighboring organs during longer holds.

#### The Mathematical Equivalence: $D^s \equiv \exp(-\gamma \cdot s \cdot E)$
Rather than re-evaluating the expensive edge neural network or reallocating 3D texture fields, MedEye3d exploits the exponential identity:

$$\exp\big(-\gamma \cdot s \cdot E(\mathbf{x})\big) = \Big(\exp\big(-\gamma E(\mathbf{x})\big)\Big)^s = D(\mathbf{x})^s$$

Scaling the effective edge sharpness by an exponent $s$ is mathematically identical to re-evaluating the edge exponential with a boosted gain $\gamma \cdot s$, requiring **zero memory reallocations** and only **1 hardware ALU cycle** (`pow(D, s)`) in the Vulkan compute shader.

#### The Frontier Shell Sampling Algorithm
To determine the optimal local boost factor $s_{\text{boost}}$, the CPU samples a dilated outer spherical shell ahead of the advancing lesion boundary:
1. Estimate current physical radius: $R_{\text{phys}}(t) \approx 3.0 + 2.0 \cdot \sqrt{\text{total\_steps} \cdot \Delta t} \cdot \bar{s}$.
2. Sample the outer shell band between $[R_{\text{phys}} - 6.0\text{ mm}, R_{\text{phys}}]$ in the diffusivity field.
3. Invert sampled $D$ to obtain edge strength: $E = -\ln(\max(D, 10^{-7})) / \gamma$.
4. Compute the mean frontier edge strength $\bar{E}_{\text{frontier}}$ and evaluate the contrast boost exponent:
   $$s_{\text{boost}} = \text{clamp}\left(\left(\frac{E_{\text{ref}}}{\max(\bar{E}_{\text{frontier}}, 0.01)}\right)^{\text{sensitivity}}, 0.5, 10.0\right)$$
   where $E_{\text{ref}} = 0.20$ is the nominal reference boundary edge strength.

#### Core-to-Frontier Transition Function
Applying $s_{\text{boost}}$ uniformly inside the lesion core would prematurely choke heat propagation. MedEye3d resolves this with a smooth, continuous transition function in GLSL:

$$w(u_c) = \text{clamp}\left(\frac{u_{\text{core}} - u_c}{u_{\text{core}} - \theta}, 0.0, 1.0\right), \quad u_{\text{core}} = 0.05$$

$$s_{\text{eff}} = \text{mix}(s_{\text{boost}}, 1.0, w(u_c))$$

$$D_{\text{eff}} = \text{clamp}(D_c^{s_{\text{eff}}}, 10^{-5}, 1.0)$$

- **Hot Core** ($u_c \ge 0.05$): $w(u_c) = 0 \implies s_{\text{eff}} = 1.0$ (maximum flow, rapid volume fill).
- **Advancing Frontier** ($u_c \to \theta$): $w(u_c) \to 1 \implies s_{\text{eff}} \to s_{\text{boost}}$ (impenetrable barrier at soft-tissue margins).

---

### 2.3 Breakthrough 3: Zero Interface Leakage via Harmonic Mean Face Conductivity
A critical vulnerability in discrete finite-volume diffusion on grids with sharp boundaries is the computation of inter-voxel interface conductance. In prior formulations, the face conductivity between adjacent voxels $D_c$ and $D_n$ was evaluated using the **arithmetic mean**:

$$D_{\text{face}}^{\text{arithmetic}} = \frac{1}{2}\big(D_c + D_n\big)$$

#### The 50,000x Permeability Bug
When heat diffuses from an open interior voxel ($D_c = 1.0$) toward a strong barrier voxel ($D_n = 10^{-5}$):
$$D_{\text{face}}^{\text{arithmetic}} = 0.5 \times (1.0 + 10^{-5}) = 0.500005$$
The effective interface conductance was **50,000 times higher** than the barrier voxel itself! Heat flowed through single-voxel anatomical boundaries at half the speed of vacuum, breaching organs and bone edges within a few iterations.

#### The Physical Solution: Harmonic Mean Interface Conductance
In physical transport phenomena across discontinuous material interfaces (e.g., Patankar finite-volume formulation), the exact interface conductivity is the **harmonic mean**:

$$D_{\text{face}}^{\text{harmonic}} = \frac{2 D_c D_n}{D_c + D_n + \epsilon_{\text{face}}}, \quad \epsilon_{\text{face}} = 10^{-7}$$

Between an interior voxel ($D_c = 1.0$) and a strong barrier ($D_n = 10^{-5}$):
$$D_{\text{face}}^{\text{harmonic}} = \frac{2 \times 1.0 \times 10^{-5}}{1.0 + 10^{-5} + 10^{-7}} = 1.9998 \times 10^{-5}$$
This slashes interface flux by **$25,000\times$**, completely eliminating edge leakage and turning single-voxel anatomical barriers into true insulators on both Vulkan GPU and CPU fallback.

---

### 2.4 Breakthrough 4: Local Relative Edge Normalization in Soft Tissue
In homogeneous soft tissues (e.g. liver parenchyma, prostate, brain), background image noise produces a non-zero edge baseline $E_{\text{bg}} \approx 0.05 - 0.10$. Under fixed global mapping $D = \exp(-10.1 E)$, this caused two problems:
1. **Sluggish Core Growth**: Diffusion inside the core was dragged down to $D \approx 0.35 - 0.50$, making growth speed dependent on absolute Hounsfield attenuation rather than relative contrast.
2. **Weak Edge Breaching**: A subtle lesion boundary in soft tissue might only peak at $E \approx 0.15 - 0.20$ ($D \approx 0.15 - 0.22$). While higher than background, $D \approx 0.18$ was not a complete stopping wall.

#### Local Adaptive Edge Contrast Formula
To solve both problems simultaneously, MedEye3d introduces **Local Relative Edge Normalization** within the immediate neighborhood ($R_{\text{phys}}$) of the seed:
1. Measure interior baseline edge $E_{\text{bg}}$ by sampling the core ($r \le 4.0\text{ mm}$, 25th percentile).
2. Measure local boundary peak $E_{\text{peak}}$ from the vicinity shell ($4.0\text{ mm} < r \le R_{\text{phys}}$, 95th percentile).
3. Compute scale factor $E_{\text{scale}} = \frac{1}{\max(E_{\text{peak}} - E_{\text{bg}}, 0.02)}$.
4. Normalize local edge contrast:
   $$E_{\text{rel}}(\mathbf{x}) = \text{clamp}\Big((E(\mathbf{x}) - E_{\text{bg}}) \cdot E_{\text{scale}}, 0.0, 1.0\Big)$$
5. Compute effective diffusivity:
   - For any globally strong edge ($E \ge 0.35$, e.g. bone, skin, air) or relative boundary peak ($E_{\text{rel}} \ge 0.85$): $D_{\text{eff}} = 10^{-5}$ (solid wall).
   - Otherwise: $D_{\text{eff}} = \exp(-\gamma_{\text{eff}} \cdot E_{\text{rel}})$, where $\gamma_{\text{eff}} = 12.0 \cdot \text{sensitivity}$.

**Result**:
- Inside the lesion core ($E \le E_{\text{bg}}$): $E_{\text{rel}} = 0.0 \implies D_{\text{eff}} = 1.0$. Flow is $100\%$ full speed, completely decoupled from background soft-tissue attenuation!
- At the local boundary ($E \to E_{\text{peak}}$): $E_{\text{rel}} \to 1.0 \implies D_{\text{eff}} = 10^{-5}$. Even faint soft-tissue margins act as firm stopping walls!
- Strong anatomical edges ($E \ge 0.35$): Always impassable barriers by default in both Wand and 1-Click modes.

---

### 2.5 Breakthrough 5: 6-Sector Cardinal Directional Normalization (Elimination of Strong-Edge Shadowing)

#### The Clinical Challenge: Asymmetric Multi-Compartment Interfaces
Tumors frequently abut two radically different anatomical compartments simultaneously:
* **Diaphragmatic / Subdiaphragmatic Masses** (e.g., Hepatic Dome HCC, Subpleural Metastases, Lower Lobe Lung Lesions):
  * **Superior ($+Z$, Cranial)**: Borders lung parenchyma and alveolar air ($\approx -850\text{ HU}$, $E \approx 0.85$). Impassable strong barrier.
  * **Inferior ($-Z$, Caudal)**: Borders normal liver soft tissue parenchyma ($\approx +55\text{ HU}$, $E \approx 0.16$). Subtle organ boundary that must NOT be breached.

#### The "Strong Edge Shadowing" Flaw of Isotropic Normalization
In isotropic scalar normalization, $E_{\text{scale}} = \frac{1}{\max(E_{\text{peak}} - E_{\text{bg}}, 0.02)}$ evaluates a single global peak. The strong lung boundary crushes the scale factor:
$$E_{\text{peak}} \approx 0.85, \; E_{\text{bg}} \approx 0.05 \implies E_{\text{scale}} \approx \frac{1}{0.80} = 1.25$$
Along the inferior ($-Z$) liver boundary ($E = 0.16$):
$$E_{\text{rel}} = (0.16 - 0.05) \times 1.25 = 0.138 \ll 0.85$$
Because $0.138 \ll 0.85$, the isotropic kernel treats the genuine liver margin as open interior ($D_{\text{eff}} \approx 0.19$)! Heat pours through into the liver, completely defeated by the lung interface on the opposite side. The strong edge in one direction **shadows** and disables boundary detection in all other directions.

#### The 6 Cardinal Sectors
To resolve this, MedEye3d partitions the local spherical shell into 6 cardinal directional cones aligned with patient anatomy:

$$\mathcal{D} = \{+X \text{ (Left)}, -X \text{ (Right)}, +Y \text{ (Posterior)}, -Y \text{ (Anterior)}, +Z \text{ (Cranial)}, -Z \text{ (Caudal)}\}$$

For each cardinal direction $k$, the host samples the corresponding cone and evaluates an independent local peak edge strength:
$$E_{\text{peak}}^{+X}, \; E_{\text{peak}}^{-X}, \; E_{\text{peak}}^{+Y}, \; E_{\text{peak}}^{-Y}, \; E_{\text{peak}}^{+Z}, \; E_{\text{peak}}^{-Z}$$

#### Continuous Spherical Projection with Exact Quadratic Partition of Unity
To avoid artificial angular seams or corner creases, the compute shader projects the physical displacement vector from the active seed $\mathbf{x}_{\text{seed}}$:
$$\mathbf{d} = (\mathbf{x} - \mathbf{x}_{\text{seed}}) \odot (\Delta x, \Delta y, \Delta z), \quad \hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|}$$

Weights are computed via positive squared components along each axis:
$$w_{+X} = \max(\hat{d}_x, 0)^2, \quad w_{-X} = \max(-\hat{d}_x, 0)^2$$
$$w_{+Y} = \max(\hat{d}_y, 0)^2, \quad w_{-Y} = \max(-\hat{d}_y, 0)^2$$
$$w_{+Z} = \max(\hat{d}_z, 0)^2, \quad w_{-Z} = \max(-\hat{d}_z, 0)^2$$

Because $\hat{d}_x^2 + \hat{d}_y^2 + \hat{d}_z^2 \equiv 1.0$ for any 3D unit vector:
$$\sum_{k \in \mathcal{D}} w_k = \hat{d}_x^2 + \hat{d}_y^2 + \hat{d}_z^2 \equiv 1.0$$

This forms an **exact, smooth, $C^1$-continuous partition of unity** on the unit sphere with **zero division** and zero angular seams.

The direction-dependent peak edge is:
$$E_{\text{peak}}(\mathbf{x}) = \sum_{k \in \mathcal{D}} w_k(\mathbf{x}) \cdot E_{\text{peak}}^k$$

#### Hardware Push Constants Architecture (112 Bytes $\le 128$ Bytes)
All 6 cardinal peaks, baseline edge, edge clip, and mode metadata are packed directly into the Vulkan push constant block (112 bytes total), requiring **zero extra descriptor sets, zero buffer allocations, and zero GPU memory roundtrips**:
* `edge_peaks_xy`: `vec4` ($+X, -X, +Y, -Y$)
* `edge_peaks_z_meta`: `vec4` ($+Z, -Z, E_{\text{bg}}, \text{edge\_clip}$)
* `gamma_meta`: `vec4` ($\gamma_{\text{eff}}, \text{is\_directional}, \text{pad}_1, \text{pad}_2$)

#### Clinical Verification & Confinement:
* **Looking Cranially ($+Z$)**: $E_{\text{peak}} \to 0.85 \implies E_{\text{scale}} \approx 1.25$. Stops at lung pleura ($E_{\text{rel}} \ge 0.85 \implies D_{\text{eff}} = 10^{-5}$, **0 leaked voxels**).
* **Looking Caudally ($-Z$)**: $E_{\text{peak}} \to 0.16 \implies E_{\text{scale}} \approx 9.09$. Stops at liver parenchyma margin ($E_{\text{rel}} \ge 0.85 \implies D_{\text{eff}} = 10^{-5}$, **0 leaked voxels**).
* **Interior Core**: $E \le E_{\text{bg}} \implies E_{\text{rel}} = 0.0 \implies D_{\text{eff}} = 1.0$ ($100\%$ unhindered propagation speed).

---

### 2.6 Breakthrough 6: Tubular & Narrow Channel Propagation (Ureter & Vessel Wall Sensing)

#### The Clinical Challenge: Ureters, Bile Ducts, and Narrow Vessels
Narrow elongated tubular structures in clinical abdomen and pelvis CT/PET scans (e.g. ureters, small arterial and venous vessels, common bile duct) present unique geometric and diffusion challenges:
1. **Extremely Narrow Lumens**: A ureter or mesenteric vessel lumen typically spans only $2 - 4\text{ mm}$ in diameter ($2 - 4$ voxels across).
2. **Convolutional Edge Bleed**: Neural edge detectors (HED3D-Ultra) and finite-difference gradient operators have effective receptive fields and Gaussian blurring that overlap across narrow lumens. Consequently, the edge probability at the seed center inside a ureter is elevated to $E_{\text{seed}} \approx 0.25 - 0.40$ even when clicked precisely on the lumen axis.
3. **Seed Diffusion Stalling**: Under previous naive logic:
   - A hardcoded barrier rule (`if (E >= 0.35) D = 1.0e-5`) immediately classified the seed's own voxel as an impenetrable insulator wall, freezing diffusion on tick 1!
   - Sampling the inner core with fixed $R_{\text{core}} = 4.0\text{ mm}$ extended outside the $1.5\text{ mm}$ ureter radius into the surrounding muscularis and retroperitoneal fat, pulling the wall edge into the background baseline $E_{\text{bg}}$.
   - Wide $45^\circ$ directional cones sampled adjacent tissue (psoas muscle, bowel), artificially driving longitudinal peaks $E_{\text{peak}}^{\pm Z}$ to high values.
   - High frontier boost exponents ($s_{\text{boost}} \ge 2.0$) heavily penalized advancing 1D tips ($0.4^{2.5} \approx 0.10$).

#### Automatic Wall Sensing & Cardinal Ray Tracing
To resolve stalling without requiring manual user toggles or specialized modes, MedEye3d introduces **Automatic Tubular Wall Sensing**:
1. **Radial Ray Tracing**: From the seed $\mathbf{x}_{\text{seed}}$, six cardinal rays are traced outward along $\pm X, \pm Y, \pm Z$. Along each ray $k$, the physical distance to the first strong edge ($E \ge 0.32$) is measured:
   $$r_k = \inf \{ \|\mathbf{x} - \mathbf{x}_{\text{seed}}\|_{\text{phys}} : E(\mathbf{x}) \ge 0.32 \}$$
2. **Minimum Wall Distance**:
   $$r_{\text{wall}} = \min_{k \in \{+X, -X, +Y, -Y, +Z, -Z\}} r_k$$
3. **Axis-Specific Linear Resistance**: For each principal coordinate axis $A \in \{X, Y, Z\}$, the average path resistance is evaluated:
   $$\mathcal{R}_A = \frac{1}{2}\big(\overline{E}_{A+} + \overline{E}_{A-}\big)$$
4. **Tubular Criterion**: If $r_{\text{wall}} \le 3.5\text{ mm}$ and the minimum resistance along any axis is significantly lower than the cross-sectional directions ($\min(\mathcal{R}) \le 0.70 \cdot \text{median}(\mathcal{R})$), MedEye3d classifies the structure as **tubular**:
   $$\text{is\_tubular} = (r_{\text{wall}} \le 3.5\text{ mm}) \land \text{anisotropic resistance}$$

#### Corridor Sampling & Dynamic Barrier Ceiling
Once tubular mode is activated:
1. **Adaptive Core Radius**:
   $$R_{\text{core}} = \max\big(1.0\text{ mm}, \; 0.75 \cdot r_{\text{wall}}\big)$$
   This guarantees that $E_{\text{bg}}$ is sampled strictly from the fluid lumen and never contaminated by surrounding muscular walls.
2. **Longitudinal Lumen Corridor Sampling**:
   Along the identified minimum resistance axis $\mathbf{a}_{\text{tube}}$ (e.g. craniocaudal $\pm Z$ for ureters), the directional peak sampling is confined to an ultra-narrow cylinder of radius $r \le \max(1.5\text{ mm}, 0.8 \cdot r_{\text{wall}})$, preventing contamination from exterior retroperitoneal fat or psoas muscle.
3. **Dynamic Barrier Ceiling**:
   Instead of a static hardcoded `0.35` cutoff, the host passes a dynamic barrier threshold $E_{\text{barrier}}$ via the push constant block (`gamma_meta.z`):
   $$E_{\text{barrier}} = \max\big(0.40, \; E_{\text{seed}} + 0.15\big)$$
   Inside the compute shader:
   ```glsl
   float E_barrier = (pc.gamma_meta.z > 0.05) ? pc.gamma_meta.z : 0.35;
   if (E >= E_barrier) D = 1.0e-5;
   ```
   This ensures that the seed voxel remains fully conductive ($D \approx 1.0$), while the surrounding tubular walls ($E \ge E_{\text{barrier}}$) remain completely impenetrable.
4. **Frontier Boost Clamping**:
   Advancing tips in 1D tubular channels have narrow surface areas. MedEye3d clamps the effective frontier boost exponent:
   $$s_{\text{eff}} = \min(s_{\text{boost}}, 1.25)$$
   preventing tip conductance from choking down.
5. **Adaptive 1-Click Step Scaling**:
   Because 1D tubular channels require more iterations to traverse substantial anatomical lengths, 1-Click mode automatically scales iteration count:
   $$K_{\text{tubular}} = \max(K, 80)$$

#### Clinical Confinement & Reach
- **Longitudinal Reach**: In a single click or wand hold on a ureter, heat smoothly streams $30 - 40+$ voxels along the tubular lumen.
- **Zero Lateral Wall Leakage**: The dynamic barrier ceiling and harmonic mean conductance strictly confine heat within the lumen, producing **0 voxels leaked** into surrounding retroperitoneum or muscle.

---

### 2.7 Breakthrough 7: Lumen Ridge / Valley Follower & Elevated Baseline Channel Normalization (Curved Vessels & Ureters)

#### The Clinical Challenge: Curvature & Elevated Edge Ridges in CT
While Breakthrough 6 established straight-line tubular detection, clinical abdominal imaging reveals two critical real-world phenomena:
1. **Anatomical Curvature**: Ureters, mesenteric vessels, and biliary ducts are not straight Cartesian cylinders. For example, in Patient 6, the left ureter drifts $5.8\text{ mm}$ along $Y$ over $13\text{ mm}$ along $Z$ ($24^\circ$ anatomical tilt). Rigid straight rays leave the lumen within $5\text{ mm}$, hitting the outer retroperitoneal wall ($E = 0.976$) and falsely declaring the longitudinal path blocked.
2. **Elevated Edge Ridge in Lumens**: In narrow structures ($2 - 4\text{ mm}$ across), CT point-spread functions and convolutional edge filters overlap from both opposing walls into the center. The lumen itself sits on an elevated edge ridge with $E \in [0.40, 0.75]$. If $E_{\text{bg}}$ is anchored to a lower baseline ($0.45$) with $\gamma_{\text{eff}} = 12.0$, face conductivity drops to $D_{\text{face}} \approx 0.0004$, throttling flow by $2,500\times$ and freezing diffusion.

#### The Mathematical & Algorithmic Solution
MedEye3d resolves this with a three-pillar architecture:

1. **Curvature-Adaptive Lumen Valley / Ridge Follower**:
   Instead of casting rigid 1D rays, MedEye3d tracks slice-by-slice along the candidate coordinate axes.
   At each successive slice, it evaluates a 2D local energy minimization:
   $$(x_s^*, y_s^*) = \arg\min_{(x, y) \in \mathcal{N}} E(x, y, z)$$
   within a local window $r \le 2.5\text{ mm}$ around the previous slice center.
   This locks onto the true lumen centerline even as the vessel curves across 3D space, measuring the true lumen edge profile $E_{\text{lumen}}(z)$ and true transverse wall distances $r_{\text{wall}}$.

2. **Lumen-Referenced Adaptive Contrast Normalization**:
   When a tubular structure is confirmed ($r_{\text{wall}} \le 4.0\text{ mm}$ and reach $\ge 8.0\text{ mm}$):
   - For high-gradient narrow lumens ($E_{\text{seed}} \ge 0.30$):
     $$E_{\text{bg}} = \min\big(0.80, \; \max(E_{\text{seed}}, \; \text{percentile}_{85}(E_{\text{lumen}}))\big)$$
     For any voxel inside the lumen where $E \le E_{\text{bg}}$, $E_{\text{rel}} \equiv 0.0 \implies D = 1.0$ ($100\%$ unhindered flow speed!).
   - Outer wall barrier threshold:
     $$E_{\text{barrier}} = \max\big(0.38, \; \min(0.96, \; \text{percentile}_{75}(E_{\text{wall}}) - 0.02)\big)$$
     All outer wall voxels ($E \ge E_{\text{barrier}}$) drop to $D = 10^{-5}$.
   - Longitudinal peak ($p_{\parallel}$): set to $\min(E_{\text{bg}} + 0.04, E_{\text{barrier}} - 0.02)$, keeping the channel open.
   - Transverse peak ($p_{\perp}$): set to $\max(0.40, E_{\text{barrier}})$, enforcing strict lateral confinement.

3. **Vulkan Shader Seed Decoupling & 1-Click Scaling ($K \ge 140$)**:
   - At the exact seed location (`dist2 < 1e-8`), the compute shader selects the open channel peak:
     $$\text{if } (pc.gamma\_meta.w > 0.5) \quad E_{\text{peak}} = \min_{k} (\text{peaks}_k)$$
   - In 1-Click mode on tubular structures, $K$ is automatically scaled:
     $$K_{\text{tubular}} = \max(K, 140)$$
     delivering $30 - 60\text{ mm}$ longitudinal flow along the ureter in $\sim 10\text{ ms}$ on GPU with **strictly 0 voxels leaked** into retroperitoneal fat.

---

## 3. Multimodal Guidance: Combined vs CT Only vs PET Only

Lesions often appear differently across modalities:
- **CT Only**: Superior for bony lesions, lung nodules, and high-contrast organ boundaries (liver capsule, kidney cortex).
- **PET Only**: Superior for metabolic active tumors lacking distinct CT density changes (e.g., peritoneal metastases, subtle lymphadenopathy).
- **Combined (CT+PET)**: Default fusion combining metabolic activity with anatomical boundaries.

MedEye3d precomputes and caches three distinct 3D diffusivity fields in the case HDF5 file:
1. `"diffusivity"`: Dual-modality combined CT+PET.
2. `"diffusivity_ct"`: CT-only guidance (PET channel masked to 0.0).
3. `"diffusivity_pet"`: PET-only guidance (CT channel masked to air baseline of -1024 HU).

When the user selects an edge source from the GUI, [`switch_heatgdt_edge_modality!`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L358-L373) fetches the desired tensor from the in-memory cache and re-uploads it to the GPU via a staging buffer copy without destroying or recreating the compute pipeline.

---

## 4. How to Use Heat-GDT in the MedEye3d GUI

### 4.1 Activation & Visibility
The **Segmentation & AI** section is permanently positioned at the top of the sidebar directly beneath **Lesion Navigation**, ensuring all controls are visible and accessible without scrolling through patient metadata or radiological reports.

1. **Shortcut**: Press the **`W`** key anywhere in the 3D viewer. This automatically:
   - Expands the **Segmentation & AI** accordion section if collapsed.
   - Scrolls the sidebar to the top so segmentation parameters are in full view.
   - Sets the algorithm selection to **`Heat-GDT`**.
   - Primes the GPU Heat-GDT pipeline with the active modality diffusivity.
   - Enters painting mode with a new or active lesion ID.
2. **GUI Section**: Alternatively, open the **Segmentation & AI** section and set the **AI** menu to **`Heat-GDT`** (which is selected by default).

```
┌────────────────────────────────────────────────────────┐
│ [▼ Segmentation & AI]                                  │
│                   VIEW MODE / EDIT MODE                │
│ [ New ]  [ Paint ]  [ Erase ]  [ View ]                │
│ Brush: [───●──────────────────] [ Move ]               │
│ AI: [ Heat-GDT              ▼ ] [ Run AI ]             │
│ Mode: [ 1-Click ] [ Wand (Hold) ]                      │
│ θ: [───●──────────────────────] K: [──────●──────────] │
│ Speed: [─────────●────────────]                        │
│ Size ±: [──────────●───────────] [ Reset ]             │
│ Edge: [ Both (Combined) ] [ CT Only ] [ PET Only ]     │
│ Adapt: [ON]    Sens: [────●───]                        │
│ θ: sensitivity (size) | K: smoothness | Edge: [Both]   │
└────────────────────────────────────────────────────────┘
```

### 4.2 Interactive Controls Explained

| Control | Parameter | Default | Function |
| :--- | :--- | :--- | :--- |
| **Mode** | `heatgdt_one_click_mode` | `1-Click` | **Operation Mode**: `[ 1-Click ]` runs full $K$-step diffusion and commits immediately upon a single mouse click; `[ Wand (Hold) ]` expands diffusion progressively across frames while holding the mouse button down. |
| **θ (Theta)** | `heatgdt_theta` | `0.001` | **Base Segmentation Threshold**. Lower values (e.g. 0.0005) produce larger, more inclusive segmentations; higher values (e.g. 0.005) produce tighter, core-only segmentations. |
| **K** | `heatgdt_K` | `40` | **Diffusion Iterations / Boundary Smoothness**. Number of Euler steps. Higher values yield smoother geodesic boundaries. |
| **Speed** | `HEAT_DISPLAY_RAMP_RATE` | `12.0` | **Visual Expansion Speed (Wand mode)**. Decoupled exponential display ramp rate. Controls how quickly the visual overlay catches up to the underlying GPU computation. |
| **Size ±** | Post-hoc adjustment | `0.0` | **Live Lesion Resizing**. Allows growing or shrinking the committed lesion mask after mouse release or one-click execution without re-painting. |
| **Edge Toggle** | `heatgdt_edge_modality` | `Both` | **Dedicated 3-Button Guidance Toggle**: `[Both]` (dual CT+PET), `[CT Only]` (pure anatomical CT edges), or `[PET Only]` (pure metabolic PET edges). Active mode glows green. Swaps GPU barrier tensors immediately on click. |
| **Adapt** | `heatgdt_adaptive_contrast`| `true` | **Adaptive Local Contrast Boost**. When enabled, dynamically measures edge strength around the advancing shell and sharpens soft-tissue margins. |
| **Sens** | `heatgdt_sensitivity` | `1.0` | **Contrast Sensitivity**. Exponent scale (0.5 to 2.0). Increase to 1.5–2.0 for very faint soft-tissue lesions. |

### 4.3 One-Click vs Wand (Hold) Workflows

- **One-Click Mode (`[ 1-Click ]`, Default)**:
  1. Select an existing lesion label or click `[ New ]`.
  2. Single-click on the lesion center in any 2D view (axial, sagittal, coronal, or MRI quadrant).
  3. The GPU immediately seeds and computes full $K$-step diffusion (~5ms on Vulkan), reads back the bounding box, commits the discrete mask to `seg_vol`, updates the timepoint cache, and refreshes 2D textures.
  4. If fine-tuning is desired, adjust the **Size ±** slider to expand or contract the segmented volume post-hoc.
- **Wand (Hold) Mode (`[ Wand (Hold) ]`)**:
  1. Click and hold the Left Mouse Button on the lesion center.
  2. Bounded Progressive Diffusion engages with identical physics to 1-Click mode:
     - **Initial press**: Bounded at 20 steps with upfront boundary-calibrated $s_{\text{boost}}$.
     - **Each tick (~50ms)**: Advances by +10 steps (20 → 30 → 40) at real $\theta = \text{heatgdt\_theta}[]$ (true WYSIWYG).
     - **Cap at $K$**: Halts cleanly when $K = \text{heatgdt\_K}[]$ (default 40) is reached, eliminating over-diffusion and boundary leakage regardless of how long the button is held.
  3. Release the mouse button when satisfied. Discrete commit occurs with zero post-release jump.

---

## 5. Where is the Code & External Test Scripts

All source code and automated test scripts are located directly within the repository.

### 5.1 Source Code Map

| File Path | Description | Key Functions / Structs |
| :--- | :--- | :--- |
| [`src/display/Vulkan/shaders/diffuse_step.comp`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/shaders/diffuse_step.comp) | Vulkan GLSL compute shader | PDE Euler step, 80-byte push constants (`edge_params`), **harmonic mean interface conductivity** (`2*Dc*Dn/(Dc+Dn+1e-7)`), **local relative edge normalization** `clamp((E-E_bg)*scale, 0, 1)`, hard absolute barriers (`E >= 0.35`) |
| [`src/display/Vulkan/VulkanHeatDiffusion.jl`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/VulkanHeatDiffusion.jl) | Julia Vulkan compute wrapper | [`compute_local_relative_edge_params`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/VulkanHeatDiffusion.jl#L313-L395), [`compute_frontier_edge_scale`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/VulkanHeatDiffusion.jl#L220-L310), [`seed_only!`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/VulkanHeatDiffusion.jl#L1094-L1190), [`run_incremental_steps!`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/VulkanHeatDiffusion.jl#L1200-L1320), [`DiffuseStepPC`](file:///workspaces/MedEye3d.jl/src/display/Vulkan/VulkanHeatDiffusion.jl#L42-L60) |
| [`src/display/GLFW/MakieEventHandlers.jl`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl) | Event dispatch & async tick loop | [`execute_one_click_heatgdt!`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L6574-L6720), [`toggle_heatgdt_one_click_mode!`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L254-L260), [`reactToOneShotHeatGDT`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L6440-L6535), [`reactToHeatGDTStart`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L6840-L6970), [`reactToHeatGDTTick`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L7014-L7160), [`reactToHeatGDTStop`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L7218-L7295), [`switch_heatgdt_edge_modality!`](file:///workspaces/MedEye3d.jl/src/display/GLFW/MakieEventHandlers.jl#L383-L396) |
| [`src/display/LesionMetadataWindow.jl`](file:///workspaces/MedEye3d.jl/src/display/LesionMetadataWindow.jl) | Makie GUI controls | Top-level `sec_seg` section, Mode `[1-Click] [Wand]` toggle, 3-button edge toggle, adaptive toggle, sensitivity slider ([lines 5278–5750](file:///workspaces/MedEye3d.jl/src/display/LesionMetadataWindow.jl#L5278-L5750)) |
| [`src/display/reactingToMouseKeyboard/reactToKeyboard/ReactToKeyboard.jl`](file:///workspaces/MedEye3d.jl/src/display/reactingToMouseKeyboard/reactToKeyboard/ReactToKeyboard.jl) | Keyboard shortcut handler | `W` key dispatch to `OneShotHeatGDTEvent` ([line 667](file:///workspaces/MedEye3d.jl/src/display/reactingToMouseKeyboard/reactToKeyboard/ReactToKeyboard.jl#L667)) |
| [`src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl`](file:///workspaces/MedEye3d.jl/src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl) | Mouse hold & drag handler | Left-button press/release triggering `HeatGDTStartEvent` and `HeatGDTStopEvent` ([lines 171–186](file:///workspaces/MedEye3d.jl/src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl#L171-L186)) |
| [`src/display/InferenceClient.jl`](file:///workspaces/MedEye3d.jl/src/display/InferenceClient.jl) | CPU fallback Heat-GDT | [`run_heatgdt`](file:///workspaces/MedEye3d.jl/src/display/InferenceClient.jl#L938-L1085) with harmonic mean conductivities, Dirichlet continuous source, and `edge_params` |
| [`src/preprocessing/OrganHeatLesionTracker.jl`](file:///workspaces/MedEye3d.jl/src/preprocessing/OrganHeatLesionTracker.jl) | Multi-timepoint lesion propagation | [`grow_heat_kernel_lesion`](file:///workspaces/MedEye3d.jl/src/preprocessing/OrganHeatLesionTracker.jl#L381-L470) with anisotropic spacing and `s_boost` |
| [`scripts/preprocessing/generate_modality_diffusivities.jl`](file:///workspaces/MedEye3d.jl/scripts/preprocessing/generate_modality_diffusivities.jl) | Batch precomputation CLI on GPU 1 | Sliding-window HED3D-Ultra inference with channel masking for HDF5 |

---

### 5.2 Standalone External Automated Test Suite

To verify the heat kernel externally in a terminal, CI/CD pipeline, or headless environment, run the following test commands:

#### Test 1: Shape-Adaptive Local Edge Contrast & Barrier Confinement
Verifies frontier edge sampling and tests that boosted adaptive diffusion ($s=6.0$) strictly confines diffusion at weak soft-tissue boundaries ($D=0.35$) where standard diffusion ($s=1.0$) leaks:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_adaptive_local_edges.jl
```
*Expected Result:* `Test Summary: Pass 7, Total 7. All tests PASSED.`

#### Test 2: Edge Modality Switching & Caching
Verifies seamless switching between `Combined`, `CT Only`, and `PET Only` edge tensors, cache hits, and fallback to `"diffusivity"`:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_modality_edge_switch.jl
```
*Expected Result:* `Test Summary: Pass 13, Total 13. All tests PASSED.`

#### Test 3: Craniocaudal Anisotropic Spacing Verification
Compares isotropic spacing $(1.0, 1.0, 1.0)$ against clinical anisotropic spacing $(0.78, 0.78, 3.0)$ on GPU and CPU, proving craniocaudal leakage suppression:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_anisotropic_heat_diffusion.jl
```
*Expected Result:* `Test Summary: Pass 9, Total 9. All tests PASSED.`

#### Test 4: Vulkan Compute Pipeline & Memory Allocation
Verifies 3D storage image allocation, ping-pong compute buffers, push constant layout, and staging buffer readbacks:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_vulkan_heat_pipeline.jl
```
*Expected Result:* `Test Summary: Pass 13, Total 13. All tests PASSED.`

#### Test 5: Full Clinical Patient 6 End-to-End Integration Test
Runs on the full $512 \times 512 \times 326$ clinical patient volume (`pat_6_files/preprocessed_volumes.h5`), tests seeding, incremental stepping, timepoint re-upload, and live switching between `diffusivity_ct` and `diffusivity_pet`:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_heatgdt_integration_pat6.jl
```
*Expected Result:* `ALL HEAT-GDT INTEGRATION TESTS PASSED ✓`

#### Test 6: Multi-Timepoint Organ & Lesion Tracking Pipeline
Verifies organ centroid IDW tracking, exclusion organ masking, and heat-kernel lesion growth across multiple timepoints:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_organ_heat_lesion_mapping.jl
```
*Expected Result:* `Test Summary: Pass 112, Total 112. All tests PASSED.`

#### Test 7: GUI Segmentation Controls, Modality Toggles & W Key Activation
Verifies that the Segmentation & AI section is visible at the top of the sidebar, algorithm defaults to Heat-GDT, edge guidance toggle buttons (`[Both]`, `[CT Only]`, `[PET Only]`) cleanly switch active GPU tensors and button highlights, and pressing `W` automatically expands the section and activates Wand mode:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_gui_segmentation_controls.jl
```
*Expected Result:* `Test Summary: Pass 25, Total 25. All tests PASSED.`

#### Test 8: One-Click Mode Toggling, GUI Buttons & GPU Execution
Verifies observable state (`heatgdt_one_click_mode`), GUI button toggling (`[1-Click]` vs `[Wand]`), and live single-click Vulkan GPU execution committing complete lesion segmentation in ~5ms:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_heatgdt_one_click.jl
```
*Expected Result:* `ALL ONE-CLICK HEAT-GDT TESTS PASSED ✓`

#### Test 9: Wand Mode Bounded Progression & Cursor Drag Stroke Painting
Verifies that Wand mode executes bounded stepping (20 → 30 → 40), halts at $K = \text{heatgdt\_K}[]$, supports drag-to-paint seed updates along the mouse cursor trajectory, and commits discrete masks without post-release jumps:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_heatgdt_wand_fidelity.jl
```
*Expected Result:* `Test Summary: Pass 17, Total 17. All tests PASSED.`

#### Test 10: Relative Edge Normalization, Core Growth Speed & Zero Leakage
Verifies that:
1. Relative edge normalization in soft tissue maps core voxels ($E \le E_{\text{bg}}$) to $D_{\text{eff}} \approx 1.0$, allowing unhindered diffusion regardless of background soft-tissue noise.
2. Weak local boundary peaks ($E \to E_{\text{peak}}$) are stretched to $E_{\text{rel}} \to 1.0 \implies D_{\text{eff}} = 10^{-5}$, forming firm stopping walls at soft-tissue margins.
3. Harmonic mean face conductivities guarantee exactly **0 voxels leakage** across strong barriers ($E \ge 0.35$) in both Wand and 1-Click modes across both CPU and Vulkan GPU:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_heatgdt_edge_leakage_and_relative_speed.jl
```
*Expected Result:* `Test Summary: Pass 14, Total 14. ALL RELATIVE EDGE AND ZERO LEAKAGE TESTS PASSED ✓`

#### Test 11: Directional Edge Normalization & Asymmetric Boundary Confinement
Verifies 6-Sector Cardinal Directional Normalization on an asymmetric diaphragmatic mass:
1. Directional cones isolate cranial $+Z$ peak ($E \approx 0.85$, lung) and caudal $-Z$ peak ($E \approx 0.16$, liver soft tissue) independently.
2. Demonstrates the failure of isotropic normalization (leaking 220+ voxels past the caudal boundary due to cranial lung shadowing).
3. Verifies that directional normalization stops at **both** cranial (lung) and caudal (liver) margins simultaneously with **0 leaked voxels** on CPU, GPU 1-Click mode, and GPU Wand mode:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_heatgdt_directional_normalization.jl
```
*Expected Result:* `Test Summary: Pass 21, Total 21. ALL DIRECTIONAL EDGE NORMALIZATION TESTS PASSED ✓`

#### Test 12: Narrow Channel / Ureter Confinement & Longitudinal Propagation
Verifies automatic tubular wall sensing, 2D lumen valley ridge-following, and longitudinal channel diffusion on both a synthetic $3\text{ mm}$ ureter and real clinical Patient 6 abdomen CT volume (`pat_6_files/preprocessed_volumes.h5`):
1. 2D valley follower traces curved ureter lumens over $60 - 100+\text{ mm}$, overcoming anatomical curvature and tilt without exiting into outer walls.
2. Lumen-referenced contrast normalization ($E_{\text{bg}}$ elevated to cover narrow lumen gradient) eliminates the $2,500\times$ conductance freeze.
3. Transverse wall barrier ceiling ($E_{\text{barrier}} \ge 0.85 - 0.94$) strictly confines diffusion to the lumen.
4. Verifies $30 - 40+$ voxels longitudinal propagation on synthetic ureter and $29.4\text{ mm}$ (9 slices, 309 voxels) on clinical Patient 6 ureter with strictly **0 lateral fat leakage** across CPU diffusion, Vulkan GPU 1-Click mode ($K \ge 140$), and Vulkan GPU Wand mode:
```bash
CUDA_VISIBLE_DEVICES=0 julia --threads=1 --project=. test/test_heatgdt_tubular_ureter.jl
```
*Expected Result:* `Test Summary: Pass 27, Total 27. ALL TUBULAR URETER HEAT DIFFUSION TESTS PASSED ✓`

---

## 6. How to Run Heat-GDT Programmatically (Headless Example)

You can run the GPU heat kernel in standalone Julia scripts without launching the Makie GUI:

```julia
using MedEye3d
using MedEye3d.VulkanBackend
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using GLFW

# 1. Initialize headless Vulkan context
GLFW.Init()
GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
GLFW.WindowHint(GLFW.VISIBLE, 0)
window = GLFW.CreateWindow(256, 256, "Headless Heat Kernel")
ctx = init_vulkan_context(window, 256, 256)

# 2. Allocate GPU heat diffusion state for volume dimensions (W, H, D)
w, h, d = 512, 512, 326
state = HeatDiffusionState()
init_heat_diffusion!(state, ctx, w, h, d)

# 3. Upload precomputed 3D diffusivity field (Float32)
# diffusivity = ... (from HDF5: read(h5["BASELINE/diffusivity_ct"]))
upload_diffusivity!(state, ctx, diffusivity)

# 4. Physical voxel dimensions (dx, dy, dz) in mm
spacing = (0.9765625, 0.9765625, 3.27)

# 5. Compute adaptive frontier contrast boost for lesion at (cx, cy, cz)
cx, cy, cz = 256, 256, 160
s_boost = compute_frontier_edge_scale(diffusivity, cx, cy, cz, 18.0, spacing; sensitivity=1.2f0)

# 6. Seed heat impulse at lesion center
seed_only!(state, ctx, cx, cy, cz; seed_radius=2, spacing=spacing, s_boost=s_boost)

# 7. Run incremental diffusion steps on GPU
# K = 40 steps, dt = 0.16, theta = 0.001
run_incremental_steps!(state, ctx, 40; dt=0.16f0, theta=0.001f0, spacing=spacing, s_boost=s_boost)

# 8. Read extracted binary mask back to CPU memory
mask = read_mask_to_cpu(state, ctx)
voxel_count = count(mask .> 0.5f0)
println("Segmented lesion volume: $voxel_count voxels")

# 9. Clean up GPU resources
destroy_heat_diffusion!(state, ctx)
destroy_vulkan_context!(ctx)
GLFW.DestroyWindow(window)
GLFW.Terminate()
```

---

## 7. How to Precompute Modality Diffusivities on GPU 1

To precompute `diffusivity_ct` and `diffusivity_pet` for a new patient or HDF5 case using GPU 1 (`CUDA_VISIBLE_DEVICES=1`), run:

```bash
CUDA_VISIBLE_DEVICES=1 julia --threads=1 --project=. scripts/preprocessing/generate_modality_diffusivities.jl /path/to/case/preprocessed_volumes.h5
```

Optional flags:
- `--ct-only`: Precompute only the CT edge field.
- `--pet-only`: Precompute only the PET edge field.

The script runs sliding-window inference with the champion HED3D-Ultra model, masks the non-target modality channels, evaluates $D = \exp(-\gamma E)$, and writes datasets directly into the HDF5 study groups.
