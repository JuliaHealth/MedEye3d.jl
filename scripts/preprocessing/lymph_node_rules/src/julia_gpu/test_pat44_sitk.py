import SimpleITK as sitk
case_dir = "data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44"
img = sitk.ReadImage(f"{case_dir}/segmentations/autochthon_left.nii.gz")
print("Direction:", img.GetDirection())
from src.legacy_python.utils import get_orientation_code
print("Orientation:", get_orientation_code(img.GetDirection()))
