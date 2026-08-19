import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import SimpleITK as sitk

def apply_transforms_and_save(mrml_path, out_dir):
    tree = ET.parse(mrml_path)
    root = tree.getroot()

    # Get transforms
    transforms = {}
    for tnode in root.findall(".//LinearTransform"):
        tid = tnode.get("id")
        matrix_str = tnode.get("matrixTransformToParent")
        if matrix_str:
            vals = list(map(float, matrix_str.split()))
            matrix = np.array(vals).reshape(4, 4).T
            
            parent_id = ""
            refs = tnode.get("references", "").split(";")
            for ref in refs:
                if ref.startswith("transform:"):
                    parent_id = ref.split(":")[1]
            
            transforms[tid] = {"matrix": matrix, "parent": parent_id}

    # Get storages
    storages = {}
    for snode in root.findall(".//VolumeArchetypeStorage") + root.findall(".//SegmentationStorage"):
        tid = snode.get("id")
        fname = snode.get("fileName")
        if fname:
            storages[tid] = fname

    mrml_dir = os.path.dirname(mrml_path)
    os.makedirs(out_dir, exist_ok=True)

    # Process volumes and segmentations
    nodes = root.findall(".//Volume") + root.findall(".//Segmentation")
    for vnode in nodes:
        name = vnode.get("name")
        refs = vnode.get("references", "").split(";")
        
        storage_id = ""
        transform_id = ""
        for ref in refs:
            if ref.startswith("storage:"):
                storage_id = ref.split(":")[1]
            elif ref.startswith("transform:"):
                transform_id = ref.split(":")[1]
                
        if storage_id in storages:
            rel_file = storages[storage_id]
            full_file = os.path.join(mrml_dir, rel_file)
            
            if os.path.isfile(full_file):
                print(f"Converting {name} from {full_file}")
                # Read with SimpleITK
                img = sitk.ReadImage(full_file)
                
                # Apply transform
                current_tid = transform_id
                T_RAS = np.eye(4)
                while current_tid and current_tid in transforms:
                    t_info = transforms[current_tid]
                    T_RAS = t_info["matrix"] @ T_RAS
                    current_tid = t_info["parent"]
                    
                if not np.allclose(T_RAS, np.eye(4)):
                    L = np.diag([-1.0, -1.0, 1.0, 1.0])
                    T_LPS = L @ T_RAS @ L
                    
                    # Convert to affine transform
                    affine = sitk.AffineTransform(3)
                    affine.SetMatrix(T_LPS[:3, :3].flatten().tolist())
                    affine.SetTranslation(T_LPS[:3, 3].tolist())
                    
                    # Inverse the transform since SimpleITK Resample uses mapping from output to input
                    affine = affine.GetInverse()
                    
                    # Resample the image to new space
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(img)
                    resampler.SetTransform(affine)
                    
                    if "seg" in rel_file.lower() or "lesion" in name.lower() or "mask" in name.lower():
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    else:
                        resampler.SetInterpolator(sitk.sitkLinear)
                        
                    img = resampler.Execute(img)
                
                # Write to nifti
                out_file = os.path.join(out_dir, name.replace(" ", "_").replace("/", "_") + ".nii.gz")
                sitk.WriteImage(img, out_file)
                print(f"Saved {out_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_mrb_to_nifti.py <path_to_mrml> <output_dir>")
        sys.exit(1)
    
    apply_transforms_and_save(sys.argv[1], sys.argv[2])
