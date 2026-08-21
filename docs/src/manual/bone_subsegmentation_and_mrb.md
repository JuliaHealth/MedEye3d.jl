# Bone Subsegmentation & MRB Scene Integration

MedEye3d.jl includes specialized anatomical preprocessing tools for quantitative radionuclide therapy assessment and 3D Slicer MRB scene compatibility.

---

## 1. Bone Subsegmentation (Surface vs Marrow)

Module: `src/preprocessing/BoneSubsegmentation.jl`

In nuclear medicine dosimetry (e.g. $^{177}\text{Lu}$-PSMA, $^{225}\text{Ac}$), distinguishing whether a metastatic lesion resides in the cortical bone surface or the trabecular bone marrow is critical for toxicity modeling.

### Processing Pipeline
1. **Bone Mask Extraction**: TotalSegmentator skeleton labels are extracted into a binary bone volume.
2. **Cortical Surface Extraction**: A 3D morphological erosion/boundary operator extracts the outer cortical surface (1-2 voxel thickness).
3. **Trabecular Marrow Extraction**: The internal cavity is identified as the trabecular bone marrow compartment.
4. **Lesion Overlap Quantification**:
   $$\text{Surface Voxels} = \sum (\text{Lesion} \cap \text{Bone Surface})$$
   $$\text{Marrow Voxels} = \sum (\text{Lesion} \cap \text{Bone Marrow})$$

### Classification Categories
- **`PURE_SURFACE`**: Lesion strictly confined to the cortical bone boundary.
- **`PURE_MARROW`**: Lesion strictly located inside the bone marrow cavity.
- **`MIXED_TRANSMURAL`**: Lesion expanding across both marrow and cortical surface.
- **`SOFT_TISSUE_NON_BONE`**: Lesion located outside skeletal structures.

---

## 2. 3D Slicer MRB Scene Ingestion

MedEye3d directly parses native 3D Slicer Medical Reality Bundles (`.mrb`):

- **`extract_mrb.py` / `convert_to_hdf5.jl`**: Unpacks MRB archives, parses MRML scene graphs, and aligns multi-segment `.seg.nrrd` files with corresponding CT and PET volumes.
- **Native Loading (`scripts/app/run_interactive_mrb.jl`)**: Loads DICOM / NIfTI series and scene hierarchies directly via `MedImages.jl`.
