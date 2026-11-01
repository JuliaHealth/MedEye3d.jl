#!/usr/bin/env python3
"""
Single Reproducible Prostate CT-MRI Registration Pipeline.

Pipeline Workflow:
  1. Loads fixed CT and moving MRI volumes.
  2. Runs / uses TotalSegmentator for both CT (-ta total --fast) and MRI (-ta total_mr --fast).
  3. Computes initial Landmark Rigid Registration:
     - Uses paired anatomical fiducials (FCSV if available or computed from TotalSeg mask centroids).
     - SVD Procrustes computes optimal rigid rotation + translation.
  4. Runs Sliced Wasserstein Distance (SWD) surface registration on the prostate:
     - Extracts prostate surface point clouds via Marching Cubes.
     - Differentiable SWD optimization in PyTorch on CUDA/GPU to refine alignment.
  5. Resamples MRI (and masks) into fixed CT space.
  6. Evaluates registration quality (Mattes Mutual Information, Prostate Dice).
  7. Outputs:
     - Registered NIfTI volumes and masks.
     - ITK H5 transform files.
     - Self-contained, clean 3D Slicer MRB scene.
     - JSON metrics summary.

Usage:
  python3 register_ct_mri.py --ct <path_to_ct> --mri <path_to_mri> --output_dir <path_to_out>

Options:
  --ct_fcsv <path>      Optional pre-computed CT fiducials
  --mri_fcsv <path>     Optional pre-computed MRI fiducials
  --seg_cache <dir>     Cache directory for TotalSegmentator segmentations
  --force_totalseg      Force re-running TotalSegmentator even if cached
  --swd_epochs <int>    Number of SWD optimization epochs (default: 800)
  --seed <int>          Random seed for full reproducibility (default: 42)
"""

import os
import sys
import glob
import shutil
import zipfile
import tempfile
import argparse
import json
import numpy as np
import torch
import SimpleITK as sitk
from skimage import measure

# Default TotalSegmentator binary location
DEFAULT_TOTAL_SEG_BIN = shutil.which("TotalSegmentator") or "/usr/local/bin/TotalSegmentator"


# ==============================================================================
# Helper Functions: Fiducials & Transforms
# ==============================================================================

def parse_fcsv(fcsv_path):
    """Parse 3D Slicer FCSV landmark file into dict of label -> np.ndarray [x, y, z] in LPS."""
    landmarks = {}
    if not fcsv_path or not os.path.exists(fcsv_path):
        return landmarks
    with open(fcsv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            if len(parts) >= 12:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    label = parts[11].strip()
                    landmarks[label] = np.array([x, y, z])
                except (ValueError, IndexError):
                    continue
    return landmarks


def extract_mask_centroid(mask_image_path):
    """Compute physical centroid (LPS) of a binary segmentation mask."""
    if not os.path.exists(mask_image_path):
        return None
    img = sitk.ReadImage(mask_image_path)
    arr = sitk.GetArrayViewFromImage(img)
    if np.sum(arr) == 0:
        return None
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(img)
    if stats.HasLabel(1):
        return np.array(stats.GetCentroid(1))
    labels = stats.GetLabels()
    if labels:
        return np.array(stats.GetCentroid(labels[0]))
    return None


def extract_prostate_keypoints(label_image):
    """
    Extract 7 canonical anatomical landmark points from a 3D binary prostate segmentation:
      1. Centroid (mean physical coordinate)
      2. Apex (inferior pole / lowest Z in LPS)
      3. Base (superior pole / highest Z in LPS)
      4. Left lateral margin (maximum X in LPS)
      5. Right lateral margin (minimum X in LPS)
      6. Anterior margin (minimum Y in LPS)
      7. Posterior margin (maximum Y in LPS)
    All returned coordinates are in physical LPS (mm).
    """
    arr = sitk.GetArrayFromImage(label_image)
    if np.sum(arr) == 0:
        return {}
    z, y, x = np.where(arr > 0)
    def v2p(vx, vy, vz):
        return np.array(label_image.TransformContinuousIndexToPhysicalPoint([float(vx), float(vy), float(vz)]))
    return {
        "prostate_centroid": v2p(x.mean(), y.mean(), z.mean()),
        "prostate_apex": v2p(x[z == z.min()].mean(), y[z == z.min()].mean(), z.min()),
        "prostate_base": v2p(x[z == z.max()].mean(), y[z == z.max()].mean(), z.max()),
        "prostate_left": v2p(x.max(), y[x == x.max()].mean(), z[x == x.max()].mean()),
        "prostate_right": v2p(x.min(), y[x == x.min()].mean(), z[x == x.min()].mean()),
        "prostate_anterior": v2p(x[y == y.min()].mean(), y.min(), z[y == y.min()].mean()),
        "prostate_posterior": v2p(x[y == y.max()].mean(), y.max(), z[y == y.max()].mean())
    }


def compute_landmark_rigid_transform(ct_landmarks, mri_landmarks, names=None):
    """
    SVD Procrustes rigid alignment:
    Finds R (orthogonal 3x3 with det=+1) and t (3,) such that:
        T(p_ct) = R @ p_ct + t ~= p_mri (maps fixed CT -> moving MRI for resampling)
    """
    ct_pts = np.array(ct_landmarks, dtype=np.float64)
    mri_pts = np.array(mri_landmarks, dtype=np.float64)
    
    ct_centroid = np.mean(ct_pts, axis=0)
    mri_centroid = np.mean(mri_pts, axis=0)
    
    ct_centered = ct_pts - ct_centroid
    mri_centered = mri_pts - mri_centroid
    
    H = ct_centered.T @ mri_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T
    t = mri_centroid - R @ ct_centroid

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(R.flatten().tolist())
    transform.SetTranslation(t.tolist())

    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"  Landmark Rigid: rotation={angle:.2f}°, trans=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm")
    if names:
        for i, name in enumerate(names):
            err = np.linalg.norm(R @ ct_pts[i] + t - mri_pts[i])
            print(f"    - {name:16s}: residual={err:6.2f} mm")
    return transform, R, t, angle


# ==============================================================================
# Helper Functions: Surface Extraction & SWD Registration
# ==============================================================================

def extract_surface_points(label_image, num_points=500, seed=42):
    """Marching cubes -> physical surface point cloud."""
    arr = sitk.GetArrayFromImage(label_image)
    if np.sum(arr) == 0:
        return np.empty((0, 3))
    try:
        verts, _, _, _ = measure.marching_cubes(arr, level=0.5)
    except (RuntimeError, ValueError):
        return np.empty((0, 3))
    
    # In SimpleITK, voxel coords are (x, y, z); numpy array is (z, y, x)
    verts_xyz = verts[:, ::-1].copy()
    origin = np.array(label_image.GetOrigin())
    spacing = np.array(label_image.GetSpacing())
    direction = np.array(label_image.GetDirection()).reshape(3, 3)
    
    physical = (verts_xyz * spacing) @ direction.T + origin
    if len(physical) > num_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(physical), num_points, replace=False)
        physical = physical[idx]
    return physical


def swd_optimization_torch(ct_points, mri_points, initial_transform,
                           max_epochs=800, lr=0.1, seed=42):
    """
    SWD rigid registration using PyTorch on CUDA/CPU.
    initial_transform maps CT -> MRI.
    Returns a refined sitk.AffineTransform mapping CT -> MRI.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  SWD Optimization device: {device} ({max_epochs} epochs, lr={lr})")

    R_init = np.array(initial_transform.GetMatrix()).reshape(3, 3)
    t_init = np.array(initial_transform.GetTranslation())

    # Bring MRI points into CT space via inverse of initial transform
    R_inv = R_init.T
    t_inv = -R_inv @ t_init
    mri_in_ct = (R_inv @ mri_points.T).T + t_inv

    ct_tensor = torch.from_numpy(ct_points).float().to(device)
    mri_tensor = torch.from_numpy(mri_in_ct).float().to(device)

    # Residual transform parameters: rotation (logit-space, +/-5 deg) + translation
    rotation_params = torch.zeros(3, device=device, requires_grad=True)
    translation_params = torch.zeros(3, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([rotation_params, translation_params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_loss = float('inf')
    best_R = None
    best_t = None

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        rot_deg = (torch.sigmoid(rotation_params) - 0.5) * 10.0
        rot_rad = torch.deg2rad(rot_deg)
        rx, ry, rz = rot_rad
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        Rx = torch.stack([torch.stack([one, zero, zero]),
                          torch.stack([zero, cx, -sx]),
                          torch.stack([zero, sx, cx])])
        Ry = torch.stack([torch.stack([cy, zero, sy]),
                          torch.stack([zero, one, zero]),
                          torch.stack([-sy, zero, cy])])
        Rz = torch.stack([torch.stack([cz, -sz, zero]),
                          torch.stack([sz, cz, zero]),
                          torch.stack([zero, zero, one])])
        R = Rz @ Ry @ Rx
        mri_xf = (R @ mri_tensor.T).T + translation_params

        # Sliced Wasserstein Distance with 200 random 1D projections
        n_proj = 200
        proj = torch.randn(n_proj, 3, device=device)
        proj = proj / torch.norm(proj, dim=1, keepdim=True)
        p_ct = ct_tensor @ proj.T
        p_mri = mri_xf @ proj.T
        s_ct, _ = torch.sort(p_ct, dim=0)
        s_mri, _ = torch.sort(p_mri, dim=0)
        loss = torch.mean((s_ct - s_mri) ** 2)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            with torch.no_grad():
                best_R = R.cpu().numpy().copy()
                best_t = translation_params.cpu().numpy().copy()

    # Compose: T_final(p_ct) = R_init @ R_res^T @ p_ct - R_init @ R_res^T @ t_res + t_init
    R_final = R_init @ best_R.T
    t_final = -R_init @ best_R.T @ best_t + t_init

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(R_final.flatten().tolist())
    transform.SetTranslation(t_final.tolist())
    
    a_final = np.degrees(np.arccos(np.clip((np.trace(R_final) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"  SWD Best Loss: {best_loss:.6f}")
    print(f"  SWD Transform: rotation={a_final:.2f}°, trans=[{t_final[0]:.2f}, {t_final[1]:.2f}, {t_final[2]:.2f}] mm")
    return transform, R_final, t_final, a_final, best_loss


# ==============================================================================
# Helper Functions: Metrics & Resampling
# ==============================================================================

def resample_mri(mri_image, ct_image, transform, interpolator=sitk.sitkLinear):
    """Resample moving MRI into fixed CT space."""
    return sitk.Resample(mri_image, ct_image, transform,
                         interpolator, 0.0, mri_image.GetPixelID())


def compute_mi(ct_image, mri_image, transform):
    """Compute Mattes Mutual Information between fixed CT and transformed moving MRI."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05, seed=42)
    reg.SetInitialTransform(transform, inPlace=False)
    try:
        return float(reg.MetricEvaluate(sitk.Cast(ct_image, sitk.sitkFloat32),
                                        sitk.Cast(mri_image, sitk.sitkFloat32)))
    except Exception as e:
        print(f"  Warning: MI calculation failed: {e}")
        return 0.0


def compute_dice(mask1, mask2):
    """Compute Sørensen-Dice coefficient between two binary masks."""
    arr1 = sitk.GetArrayViewFromImage(mask1) > 0
    arr2 = sitk.GetArrayViewFromImage(mask2) > 0
    intersection = np.logical_and(arr1, arr2).sum()
    total = arr1.sum() + arr2.sum()
    return float(2.0 * intersection / total) if total > 0 else 0.0


# ==============================================================================
# Helper Functions: 3D Slicer MRB Packaging
# ==============================================================================

def strip_itk_metadata(image):
    """Remove ITK/NIfTI header keys that corrupt NRRD headers in 3D Slicer."""
    keys_to_remove = [k for k in image.GetMetaDataKeys()
                      if k.startswith("ITK_") or k in (
                          "aux_file", "bitpix", "cal_max", "cal_min",
                          "datatype", "descrip", "intent_code", "intent_name",
                          "intent_p1", "intent_p2", "intent_p3", "pixdim[0]",
                          "pixdim[1]", "pixdim[2]", "pixdim[3]", "pixdim[4]",
                          "pixdim[5]", "pixdim[6]", "pixdim[7]",
                          "qform_code", "qform_code_name", "qoffset_x",
                          "qoffset_y", "qoffset_z", "quatern_b", "quatern_c",
                          "quatern_d", "scl_inter", "scl_slope", "sform_code",
                          "sform_code_name", "srow_x", "srow_y", "srow_z",
                          "slice_code", "slice_duration", "slice_end",
                          "slice_start", "toffset", "vox_offset",
                          "xyzt_units", "dim_info",
                      ) or k.startswith("dim[") or k.startswith("srow_")
                      or k.startswith("qoffset_") or k.startswith("quatern_")
                      or k.startswith("pixdim[")]
    for k in keys_to_remove:
        image.EraseMetaData(k)
    return image


def create_slicer_mrb(output_path, volumes, transforms=None, fiducials=None,
                      scene_name="Scene", center_z=-870.0, center_y=140.0):
    """Create a fully compliant, self-contained 3D Slicer MRB scene file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scene_dir = os.path.join(tmpdir, scene_name)
        data_dir = os.path.join(scene_dir, "Data")
        os.makedirs(data_dir, exist_ok=True)

        mrml = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<MRML version="Slicer4.4" userTags="">',
            ' <Selection'
            '  id="vtkMRMLSelectionNodeSingleton" name="Selection"'
            '  hideFromEditors="true" selectable="true" selected="false"'
            '  singletonTag="Singleton"'
            '  activePlaceNodeClassName="vtkMRMLMarkupsFiducialNode"'
            ' ></Selection>',
            ' <Interaction'
            '  id="vtkMRMLInteractionNodeSingleton" name="Interaction"'
            '  hideFromEditors="true" selectable="true" selected="false"'
            '  singletonTag="Singleton"'
            '  currentInteractionMode="ViewTransform"'
            ' ></Interaction>'
        ]

        nid = 1
        vol_node_ids = []
        for vn, vi in volumes.items():
            fn = f"{vn}.nrrd"
            vol_clean = strip_itk_metadata(sitk.Image(vi))
            sitk.WriteImage(vol_clean, os.path.join(data_dir, fn), useCompression=True)

            vol_node_id = f"vtkMRMLScalarVolumeNode{nid}"
            vol_node_ids.append(vol_node_id)

            mrml.append(
                f' <VolumeArchetypeStorage'
                f'  id="vtkMRMLVolumeArchetypeStorageNode{nid}"'
                f'  name="{vn}" fileName="Data/{fn}"'
                f'  fileArcheType="Data/{fn}" freeSurferAnalyze="0"'
                f'  centerImage="0" useOrientationFromFile="1"'
                f'  singleFile="0"'
                f' ></VolumeArchetypeStorage>'
            )

            mrml.append(
                f' <VolumeDisplay'
                f'  id="vtkMRMLScalarVolumeDisplayNode{nid}"'
                f'  name="VolumeDisplay"'
                f'  hideFromEditors="true" selectable="true" selected="false"'
                f'  color="0.9 0.9 0.3"'
                f'  selectedColor="1 0 0"'
                f'  ambient="0" diffuse="1" specular="0" power="1"'
                f'  opacity="1" visibility="true"'
                f'  scalarVisibility="false" vectorVisibility="false"'
                f'  tensorVisibility="false" interpolateTexture="false"'
                f'  scalarRangeFlag="UseData" scalarRange="0 100"'
                f'  colorNodeID="vtkMRMLColorTableNodeGrey"'
                f'  autoWindowLevel="1" interpolate="1"'
                f'  applyThreshold="0" autoThreshold="0"'
                f'  upperThreshold="32767" lowerThreshold="-32768"'
                f' ></VolumeDisplay>'
            )

            spacing = vi.GetSpacing()
            origin = vi.GetOrigin()
            direction = vi.GetDirection()
            # Convert LPS origin to RAS for Slicer MRML (negate x, y)
            ras_origin = (-origin[0], -origin[1], origin[2])
            d = list(direction)
            ijk2ras = [
                -d[0], -d[1], -d[2],
                -d[3], -d[4], -d[5],
                d[6],  d[7],  d[8]
            ]
            dir_str = "   ".join([str(v) for v in ijk2ras])
            mrml.append(
                f' <Volume'
                f'  id="{vol_node_id}" name="{vn}"'
                f'  hideFromEditors="false" selectable="true" selected="false"'
                f'  references="display:vtkMRMLScalarVolumeDisplayNode{nid};'
                f'storage:vtkMRMLVolumeArchetypeStorageNode{nid};"'
                f'  userTags=""'
                f'  spacing="{spacing[0]} {spacing[1]} {spacing[2]}"'
                f'  origin="{ras_origin[0]} {ras_origin[1]} {ras_origin[2]}"'
                f'  voxelVectorType="undefined"'
                f'  ijkToRASDirections="{dir_str} "'
                f' ></Volume>'
            )
            nid += 1

        bg_vol = vol_node_ids[0] if vol_node_ids else ""
        fg_vol = vol_node_ids[-1] if len(vol_node_ids) > 1 else ""
        for view_name, view_label, orientation, color, slice_matrix in [
            ("Red", "R", "Axial", "0.952941 0.290196 0.2",
             f"-1 0 0 0 0 1 0 {center_y:.1f} 0 0 1 {center_z:.1f} 0 0 0 1"),
            ("Green", "G", "Coronal", "0.431373 0.690196 0.294118",
             f"-1 0 0 0 0 0 1 {center_z:.1f} 0 1 0 {center_y:.1f} 0 0 0 1"),
            ("Yellow", "Y", "Sagittal", "0.882353 0.882353 0.294118",
             f"0 0 -1 0 -1 0 0 {center_y:.1f} 0 1 0 {center_z:.1f} 0 0 0 1"),
        ]:
            mrml.append(
                f' <Slice'
                f'  id="vtkMRMLSliceNode{view_name}" name="{view_name}"'
                f'  hideFromEditors="false" selectable="true" selected="false"'
                f'  singletonTag="{view_name}"'
                f'  attributes="MappedInLayout:1"'
                f'  layoutLabel="{view_label}" layoutName="{view_name}"'
                f'  active="false" visibility="true"'
                f'  backgroundColor="0 0 0" backgroundColor2="0 0 0"'
                f'  layoutColor="{color}"'
                f'  orientationMarkerType="none" orientationMarkerSize="medium"'
                f'  rulerType="none" rulerColor="white"'
                f'  AxisLabels="L;R;P;A;I;S"'
                f'  fieldOfView="350 350 1"'
                f'  dimensions="256 256 1"'
                f'  xyzOrigin="0 0 0"'
                f'  sliceResolutionMode="1"'
                f'  uvwExtents="350 350 1" uvwDimensions="256 256 1" uvwOrigin="0 0 0"'
                f'  activeSlice="0" layoutGridRows="1" layoutGridColumns="1"'
                f'  sliceToRAS="{slice_matrix}"'
                f'  orientationMatrixAxial="-1 0 0 0 1 0 0 0 1"'
                f'  orientationMatrixSagittal="0 0 -1 -1 0 0 0 1 0"'
                f'  orientationMatrixCoronal="-1 0 0 0 0 1 0 1 0"'
                f'  orientation="{orientation}" defaultOrientation="{orientation}"'
                f'  orientationReference="{orientation}"'
                f'  jumpMode="1" sliceVisibility="false"'
                f'  widgetVisibility="false" widgetOutlineVisibility="true"'
                f'  useLabelOutline="false"'
                f' ></Slice>'
            )
            refs = ""
            if bg_vol:
                refs += f"backgroundVolume:{bg_vol};"
            if fg_vol:
                refs += f"foregroundVolume:{fg_vol};"
            mrml.append(
                f' <SliceComposite'
                f'  id="vtkMRMLSliceCompositeNode{view_name}"'
                f'  name="SliceComposite" hideFromEditors="true"'
                f'  selectable="true" selected="false"'
                f'  singletonTag="{view_name}"'
                f'  references="{refs}"'
                f'  compositing="0" foregroundOpacity="0.5"'
                f'  labelOpacity="1" linkedControl="1"'
                f'  fiducialVisibility="1" fiducialLabelVisibility="1"'
                f'  layoutName="{view_name}"'
                f'  doPropagateVolumeSelection="1"'
                f' ></SliceComposite>'
            )

        if transforms:
            for tn, td in transforms.items():
                fn = f"{tn}.h5"
                fp = os.path.join(data_dir, fn)
                if isinstance(td, sitk.Transform):
                    sitk.WriteTransform(td, fp)
                elif isinstance(td, str) and os.path.exists(td):
                    shutil.copy(td, fp)
                else:
                    continue
                mrml.append(
                    f' <TransformStorage id="vtkMRMLTransformStorageNode{nid}"'
                    f'  name="{tn}" fileName="Data/{fn}"'
                    f' ></TransformStorage>'
                )
                mrml.append(
                    f' <LinearTransform id="vtkMRMLLinearTransformNode{nid}"'
                    f'  name="{tn}"'
                    f'  references="storage:vtkMRMLTransformStorageNode{nid};"'
                    f' ></LinearTransform>'
                )
                nid += 1

        if fiducials:
            for fn_name, pts in fiducials.items():
                fn = f"{fn_name}.fcsv"
                fp = os.path.join(data_dir, fn)
                with open(fp, 'w') as f:
                    f.write("# Markups fiducial file version = 4.11\n")
                    f.write("# CoordinateSystem = LPS\n")
                    f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
                    for i, pt in enumerate(pts):
                        x, y, z = pt[0], pt[1], pt[2]
                        label = pt[3] if len(pt) > 3 else f"pt_{i}"
                        f.write(f"vtkMRMLMarkupsFiducialNode_{i},{x},{y},{z},0,0,0,1,1,1,0,{label},\n")
                mrml.append(
                    f' <MarkupsFiducialStorage id="vtkMRMLMarkupsFiducialStorageNode{nid}"'
                    f'  name="{fn_name}" fileName="Data/{fn}"'
                    f' ></MarkupsFiducialStorage>'
                )
                mrml.append(
                    f' <MarkupsFiducial id="vtkMRMLMarkupsFiducialNode{nid}"'
                    f'  name="{fn_name}"'
                    f'  references="storage:vtkMRMLMarkupsFiducialStorageNode{nid};"'
                    f' ></MarkupsFiducial>'
                )
                nid += 1

        mrml.append('</MRML>')
        mrml_path = os.path.join(scene_dir, f"{scene_name}.mrml")
        with open(mrml_path, 'w') as f:
            f.write('\n'.join(mrml))

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(scene_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, tmpdir)
                    zf.write(fpath, arcname)
        print(f"  Created 3D Slicer MRB: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


# ==============================================================================
# Main Pipeline Function
# ==============================================================================

def run_registration(
    ct_path,
    mri_path,
    output_dir,
    ct_fcsv=None,
    mri_fcsv=None,
    totalseg_bin=DEFAULT_TOTAL_SEG_BIN,
    seg_cache_dir=None,
    force_totalseg=False,
    swd_epochs=800,
    swd_lr=0.1,
    swd_points=500,
    seed=42,
    scene_name="Prostate_Registration",
    skip_mrb=False
):
    """
    Executes complete CT-MRI registration pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("PROSTATE CT-MRI REGISTRATION PIPELINE")
    print(f"  Fixed CT:    {ct_path}")
    print(f"  Moving MRI:  {mri_path}")
    print(f"  Output Dir:  {output_dir}")
    print("=" * 75)

    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT volume not found: {ct_path}")
    if not os.path.exists(mri_path):
        raise FileNotFoundError(f"MRI volume not found: {mri_path}")

    # Set random seeds for deterministic execution
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Load CT and MRI
    print("\n[Step 1/7] Loading Volumes...")
    ct_image = sitk.ReadImage(ct_path)
    mri_image = sitk.ReadImage(mri_path)
    print(f"  CT:  Size={ct_image.GetSize()}, Spacing={[round(s, 2) for s in ct_image.GetSpacing()]}, Type={ct_image.GetPixelIDTypeAsString()}")
    print(f"  MRI: Size={mri_image.GetSize()}, Spacing={[round(s, 2) for s in mri_image.GetSpacing()]}, Type={mri_image.GetPixelIDTypeAsString()}")

    # 2. TotalSegmentator Segmentations
    print("\n[Step 2/7] Segmenting Volumes with TotalSegmentator...")
    if not seg_cache_dir:
        seg_cache_dir = os.path.join(output_dir, "segmentations")
    os.makedirs(seg_cache_dir, exist_ok=True)

    ct_seg_dir = os.path.join(seg_cache_dir, "ct")
    mri_seg_dir = os.path.join(seg_cache_dir, "mri")
    os.makedirs(ct_seg_dir, exist_ok=True)
    os.makedirs(mri_seg_dir, exist_ok=True)

    ct_prostate_path = os.path.join(ct_seg_dir, "prostate.nii.gz")
    mri_prostate_path = os.path.join(mri_seg_dir, "prostate.nii.gz")

    # Check for pre-existing masks in case directories
    scans_ct_dir = os.path.dirname(ct_path)
    if not os.path.exists(ct_prostate_path):
        for candidate in [
            os.path.join(scans_ct_dir, "CT_Prostate_Mask.nii.gz"),
            os.path.join(scans_ct_dir, "prostate.nii.gz")
        ]:
            if os.path.exists(candidate) and os.path.getsize(candidate) > 1000:
                shutil.copy(candidate, ct_prostate_path)
                print(f"  Loaded pre-existing CT prostate mask from {candidate}")
                break

    scans_mri_dir = os.path.dirname(mri_path)
    if not os.path.exists(mri_prostate_path):
        for candidate in [
            os.path.join(scans_mri_dir, "MRI_Prostate_Mask.nii.gz"),
            os.path.join(scans_mri_dir, "prostate.nii.gz")
        ]:
            if os.path.exists(candidate) and os.path.getsize(candidate) > 1000:
                shutil.copy(candidate, mri_prostate_path)
                print(f"  Loaded pre-existing MRI prostate mask from {candidate}")
                break

    need_ct_seg = force_totalseg or not (os.path.exists(ct_prostate_path) and os.path.getsize(ct_prostate_path) > 1000)
    need_mri_seg = force_totalseg or not (os.path.exists(mri_prostate_path) and os.path.getsize(mri_prostate_path) > 1000)

    roi_subset_arg = "--roi_subset prostate urinary_bladder femur_left femur_right hip_left hip_right"

    if need_ct_seg:
        print(f"  Running TotalSegmentator on CT: {totalseg_bin} -i '{ct_path}' -o '{ct_seg_dir}' -ta total --fast {roi_subset_arg}")
        cmd_ct = f"{totalseg_bin} -i '{ct_path}' -o '{ct_seg_dir}' -ta total --fast {roi_subset_arg} --quiet"
        ret = os.system(cmd_ct)
        if ret != 0:
            print(f"  Warning: TotalSegmentator CT exited with code {ret}")
    else:
        print(f"  Using cached CT segmentations: {ct_seg_dir}")

    if need_mri_seg:
        print(f"  Running TotalSegmentator on MRI: {totalseg_bin} -i '{mri_path}' -o '{mri_seg_dir}' -ta total_mr --fast {roi_subset_arg}")
        cmd_mri = f"{totalseg_bin} -i '{mri_path}' -o '{mri_seg_dir}' -ta total_mr --fast {roi_subset_arg} --quiet"
        ret = os.system(cmd_mri)
        if ret != 0:
            print(f"  Warning: TotalSegmentator MRI exited with code {ret}")
    else:
        print(f"  Using cached MRI segmentations: {mri_seg_dir}")

    # 3. Landmark Identification & Initial Alignment
    print("\n[Step 3/7] Landmark Identification & Initial Rigid Alignment...")
    # Check for FCSV files if not explicitly provided
    if not ct_fcsv:
        scans_dir = os.path.dirname(ct_path)
        candidates = glob.glob(os.path.join(scans_dir, "*FIXED*.fcsv"))
        if candidates:
            ct_fcsv = candidates[0]
    if not mri_fcsv:
        scans_dir = os.path.dirname(mri_path)
        candidates = glob.glob(os.path.join(scans_dir, "*MOVING*.fcsv"))
        if candidates:
            mri_fcsv = candidates[0]

    ct_landmarks = {}
    mri_landmarks = {}
    if ct_fcsv and mri_fcsv and os.path.exists(ct_fcsv) and os.path.exists(mri_fcsv):
        print(f"  Loading fiducials from FCSV files:\n    CT:  {ct_fcsv}\n    MRI: {mri_fcsv}")
        ct_landmarks = parse_fcsv(ct_fcsv)
        mri_landmarks = parse_fcsv(mri_fcsv)

    # If no FCSV or incomplete landmarks, compute from TotalSegmentator masks
    key_labels = ['femur_left', 'femur_right', 'hip_left', 'hip_right', 'prostate', 'urinary_bladder']
    common_landmarks = sorted(set(ct_landmarks.keys()) & set(mri_landmarks.keys()))
    if len(common_landmarks) < 3:
        print("  Extracting paired anatomical centroids from TotalSegmentator segmentations...")
        ct_landmarks = {}
        mri_landmarks = {}
        for lbl in key_labels:
            c_ct = extract_mask_centroid(os.path.join(ct_seg_dir, f"{lbl}.nii.gz"))
            c_mri = extract_mask_centroid(os.path.join(mri_seg_dir, f"{lbl}.nii.gz"))
            if c_ct is not None and c_mri is not None:
                ct_landmarks[lbl] = c_ct
                mri_landmarks[lbl] = c_mri
        common_landmarks = sorted(set(ct_landmarks.keys()) & set(mri_landmarks.keys()))

    use_prostate_kps = False
    if len(common_landmarks) >= 3:
        ct_pts_list = [ct_landmarks[l] for l in common_landmarks]
        mri_pts_list = [mri_landmarks[l] for l in common_landmarks]
        landmark_transform, R_lm, t_lm, angle_lm = compute_landmark_rigid_transform(
            ct_pts_list, mri_pts_list, common_landmarks)
        if angle_lm > 30.0:
            print(f"  [Notice] Initial rotation angle ({angle_lm:.1f}°) indicates FOV slab truncation of bone landmarks.")
            use_prostate_kps = True
    else:
        use_prostate_kps = True

    if use_prostate_kps and os.path.exists(ct_prostate_path) and os.path.exists(mri_prostate_path):
        print("  Using 3D canonical prostate keypoints (centroid, apex, base, lateral, A/P margins) for rigid init...")
        ct_pro_img_temp = sitk.ReadImage(ct_prostate_path)
        mri_pro_img_temp = sitk.ReadImage(mri_prostate_path)
        ct_kps = extract_prostate_keypoints(ct_pro_img_temp)
        mri_kps = extract_prostate_keypoints(mri_pro_img_temp)
        if len(ct_kps) >= 3 and len(mri_kps) >= 3:
            kp_names = sorted(set(ct_kps.keys()) & set(mri_kps.keys()))
            ct_pts_list = [ct_kps[k] for k in kp_names]
            mri_pts_list = [mri_kps[k] for k in kp_names]
            landmark_transform, R_lm, t_lm, angle_lm = compute_landmark_rigid_transform(
                ct_pts_list, mri_pts_list, kp_names)
            common_landmarks = kp_names
            ct_landmarks = ct_kps
            mri_landmarks = mri_kps

    # 4. Sliced Wasserstein Distance (SWD) Prostate Surface Alignment
    print("\n[Step 4/7] Sliced Wasserstein Distance (SWD) Prostate Surface Refinement...")
    swd_transform = landmark_transform
    R_swd, t_swd, angle_swd, swd_loss = R_lm, t_lm, angle_lm, None

    if os.path.exists(ct_prostate_path) and os.path.exists(mri_prostate_path):
        ct_pro_img = sitk.ReadImage(ct_prostate_path)
        mri_pro_img = sitk.ReadImage(mri_prostate_path)
        ct_pts = extract_surface_points(ct_pro_img, num_points=swd_points, seed=seed)
        mri_pts = extract_surface_points(mri_pro_img, num_points=swd_points, seed=seed)
        print(f"  Prostate surface points: CT={len(ct_pts)}, MRI={len(mri_pts)}")

        if len(ct_pts) > 10 and len(mri_pts) > 10:
            min_n = min(len(ct_pts), len(mri_pts))
            rng = np.random.RandomState(seed)
            ct_eq = ct_pts[rng.choice(len(ct_pts), min_n, replace=False)]
            mri_eq = mri_pts[rng.choice(len(mri_pts), min_n, replace=False)]
            swd_transform, R_swd, t_swd, angle_swd, swd_loss = swd_optimization_torch(
                ct_eq, mri_eq, landmark_transform,
                max_epochs=swd_epochs, lr=swd_lr, seed=seed)
        else:
            print("  Warning: Prostate surface contains insufficient points; using Landmark transform.")
    else:
        print("  Warning: Prostate segmentation masks missing; using Landmark transform.")

    # 5. Resampling Volumes & Masks
    print("\n[Step 5/7] Resampling Moving Volumes and Masks into CT Space...")
    mri_reg_landmark = resample_mri(mri_image, ct_image, landmark_transform, sitk.sitkLinear)
    mri_reg_swd = resample_mri(mri_image, ct_image, swd_transform, sitk.sitkLinear)

    ct_prostate_img = sitk.ReadImage(ct_prostate_path) if os.path.exists(ct_prostate_path) else None
    mri_prostate_img = sitk.ReadImage(mri_prostate_path) if os.path.exists(mri_prostate_path) else None
    mri_pro_reg_lm = None
    mri_pro_reg_swd = None
    if mri_prostate_img is not None:
        mri_pro_reg_lm = resample_mri(mri_prostate_img, ct_image, landmark_transform, sitk.sitkNearestNeighbor)
        mri_pro_reg_swd = resample_mri(mri_prostate_img, ct_image, swd_transform, sitk.sitkNearestNeighbor)

    # 6. Quantitative Evaluation Metrics
    print("\n[Step 6/7] Computing Quantitative Evaluation Metrics...")
    mi_unaligned = compute_mi(ct_image, mri_image, sitk.AffineTransform(3))
    mi_landmark = compute_mi(ct_image, mri_image, landmark_transform)
    mi_swd = compute_mi(ct_image, mri_image, swd_transform)

    dice_unaligned = float('nan')
    dice_landmark = float('nan')
    dice_swd = float('nan')
    if ct_prostate_img is not None and mri_prostate_img is not None:
        mri_pro_unaligned = resample_mri(mri_prostate_img, ct_image, sitk.AffineTransform(3), sitk.sitkNearestNeighbor)
        dice_unaligned = compute_dice(ct_prostate_img, mri_pro_unaligned)
        dice_landmark = compute_dice(ct_prostate_img, mri_pro_reg_lm)
        dice_swd = compute_dice(ct_prostate_img, mri_pro_reg_swd)

    print(f"  Mattes Mutual Information (more negative = better similarity):")
    print(f"    - Unaligned:     {mi_unaligned:8.4f}")
    print(f"    - Landmark Init: {mi_landmark:8.4f}")
    print(f"    - Landmark+SWD:  {mi_swd:8.4f}")
    if not np.isnan(dice_swd):
        print(f"  Prostate Sørensen-Dice Overlap:")
        print(f"    - Unaligned:     {dice_unaligned:8.4f}")
        print(f"    - Landmark Init: {dice_landmark:8.4f}")
        print(f"    - Landmark+SWD:  {dice_swd:8.4f}")

    # 7. Saving Outputs (NIfTI, H5 Transforms, 3D Slicer MRB, JSON)
    print("\n[Step 7/7] Saving Outputs...")

    # A. NIfTI outputs
    nifti_dir = os.path.join(output_dir, "nifti")
    os.makedirs(nifti_dir, exist_ok=True)
    mri_swd_nii_path = os.path.join(nifti_dir, "MRI_registered_to_CT_SWD.nii.gz")
    mri_lm_nii_path = os.path.join(nifti_dir, "MRI_registered_to_CT_Landmark_Init.nii.gz")
    sitk.WriteImage(mri_reg_swd, mri_swd_nii_path)
    sitk.WriteImage(mri_reg_landmark, mri_lm_nii_path)
    print(f"  Saved NIfTI: {mri_swd_nii_path}")
    print(f"  Saved NIfTI: {mri_lm_nii_path}")

    if ct_prostate_img is not None:
        ct_pro_nii_path = os.path.join(nifti_dir, "CT_Prostate_Mask.nii.gz")
        sitk.WriteImage(ct_prostate_img, ct_pro_nii_path)
    if mri_pro_reg_swd is not None:
        mri_pro_swd_nii = os.path.join(nifti_dir, "MRI_Prostate_Mask_registered_SWD.nii.gz")
        sitk.WriteImage(mri_pro_reg_swd, mri_pro_swd_nii)

    # B. Transforms
    transforms_dir = os.path.join(output_dir, "transforms")
    os.makedirs(transforms_dir, exist_ok=True)
    lm_tfm_path = os.path.join(transforms_dir, "Transform_Landmark_Init.h5")
    swd_tfm_path = os.path.join(transforms_dir, "Transform_Landmark_SWD.h5")
    sitk.WriteTransform(landmark_transform, lm_tfm_path)
    sitk.WriteTransform(swd_transform, swd_tfm_path)
    # Also write .tfm files for compatibility with MedEye3d / ITK
    swd_tfm_itk_path = os.path.join(transforms_dir, "Transform_Landmark_SWD.tfm")
    followup_tfm_path = os.path.join(transforms_dir, "Transform_FollowUp_to_Baseline_1.tfm")
    sitk.WriteTransform(swd_transform, swd_tfm_itk_path)
    sitk.WriteTransform(swd_transform, followup_tfm_path)
    print(f"  Saved Transform: {swd_tfm_path}")
    print(f"  Saved Transform: {swd_tfm_itk_path}")

    # C. Fiducials for MRB
    center_z = -870.0
    center_y = 140.0
    if "prostate" in ct_landmarks:
        center_z = float(ct_landmarks["prostate"][2])
        center_y = float(-ct_landmarks["prostate"][1])  # RAS: Y_ras = -Y_lps

    fiducials = {
        "CT_Fiducials": [(p[0], p[1], p[2], f"CT_{n}") for n, p in ct_landmarks.items()],
        "MRI_Fiducials": [(p[0], p[1], p[2], f"MRI_{n}") for n, p in mri_landmarks.items()]
    }

    # D. 3D Slicer MRB package
    mrb_path = os.path.join(output_dir, f"{scene_name}.mrb")
    if not skip_mrb:
        mrb_volumes = {
            "CT": ct_image,
            "MRI_Original": mri_image,
            "MRI_Landmark_Init": mri_reg_landmark,
            "MRI_Landmark_SWD_Prostate": mri_reg_swd,
        }
        mrb_transforms = {
            "T_Landmark_Init": landmark_transform,
            "T_Landmark_SWD": swd_transform,
        }
        create_slicer_mrb(
            mrb_path,
            mrb_volumes,
            transforms=mrb_transforms,
            fiducials=fiducials,
            scene_name=scene_name,
            center_z=center_z,
            center_y=center_y
        )

    # E. Summary JSON
    summary = {
        "fixed_ct": ct_path,
        "moving_mri": mri_path,
        "landmarks_used": common_landmarks,
        "transforms": {
            "Landmark_Init": {
                "rotation_deg": float(angle_lm),
                "translation_mm": [float(x) for x in t_lm],
                "matrix_3x3": R_lm.tolist(),
                "MI": float(mi_landmark),
                "prostate_dice": float(dice_landmark) if not np.isnan(dice_landmark) else None
            },
            "Landmark_SWD_Prostate": {
                "rotation_deg": float(angle_swd),
                "translation_mm": [float(x) for x in t_swd],
                "matrix_3x3": R_swd.tolist(),
                "swd_best_loss": float(swd_loss) if swd_loss is not None else None,
                "MI": float(mi_swd),
                "prostate_dice": float(dice_swd) if not np.isnan(dice_swd) else None
            }
        },
        "metrics": {
            "MI": {
                "unaligned": float(mi_unaligned),
                "landmark_init": float(mi_landmark),
                "landmark_swd": float(mi_swd)
            },
            "prostate_dice": {
                "unaligned": float(dice_unaligned) if not np.isnan(dice_unaligned) else None,
                "landmark_init": float(dice_landmark) if not np.isnan(dice_landmark) else None,
                "landmark_swd": float(dice_swd) if not np.isnan(dice_swd) else None
            }
        },
        "output_files": {
            "mrb": mrb_path,
            "registered_mri_swd_nifti": mri_swd_nii_path,
            "registered_mri_landmark_nifti": mri_lm_nii_path,
            "transform_swd_h5": swd_tfm_path,
            "transform_landmark_h5": lm_tfm_path
        }
    }
    json_path = os.path.join(output_dir, "registration_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved JSON Summary: {json_path}")

    print("\n" + "=" * 75)
    print("REGISTRATION PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  MRB:   {mrb_path}")
    print(f"  NIfTI: {mri_swd_nii_path}")
    print(f"  JSON:  {json_path}")
    print("=" * 75)
    return summary


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single Reproducible Prostate CT-MRI Landmark + SWD Registration Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ct", required=True, help="Path to fixed CT volume (.nii.gz)")
    parser.add_argument("--mri", required=True, help="Path to moving MRI volume (.nii.gz)")
    parser.add_argument("--output_dir", default="./registration_output", help="Output directory")
    parser.add_argument("--ct_fcsv", default=None, help="Optional pre-computed CT fiducials (.fcsv)")
    parser.add_argument("--mri_fcsv", default=None, help="Optional pre-computed MRI fiducials (.fcsv)")
    parser.add_argument("--totalseg_bin", default=DEFAULT_TOTAL_SEG_BIN, help="TotalSegmentator executable binary path")
    parser.add_argument("--seg_cache", default=None, help="Directory to store or load cached TotalSegmentator masks")
    parser.add_argument("--force_totalseg", action="store_true", help="Force re-running TotalSegmentator even if cached")
    parser.add_argument("--swd_epochs", type=int, default=800, help="Number of SWD optimization epochs")
    parser.add_argument("--swd_lr", type=float, default=0.1, help="Learning rate for SWD optimization")
    parser.add_argument("--swd_points", type=int, default=500, help="Number of surface points sampled for SWD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--scene_name", default="Prostate_CT_MRI_Registration", help="Base name for the Slicer MRB scene")

    args = parser.parse_args()

    run_registration(
        ct_path=args.ct,
        mri_path=args.mri,
        output_dir=args.output_dir,
        ct_fcsv=args.ct_fcsv,
        mri_fcsv=args.mri_fcsv,
        totalseg_bin=args.totalseg_bin,
        seg_cache_dir=args.seg_cache,
        force_totalseg=args.force_totalseg,
        swd_epochs=args.swd_epochs,
        swd_lr=args.swd_lr,
        swd_points=args.swd_points,
        seed=args.seed,
        scene_name=args.scene_name
    )


if __name__ == "__main__":
    main()
