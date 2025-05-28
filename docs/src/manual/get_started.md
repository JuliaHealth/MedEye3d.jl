## Image Orientation Guidelines

MedImages.jl is optimized for images in **LPS (Left-Posterior-Superior)** orientation for consistent processing and analysis. If your medical images are in a different orientation, we recommend converting them to LPS before using MedImages.jl.

### Quick Orientation Converter

To help you convert your images to LPS orientation, we provide a convenient script that handles both NIfTI files and DICOM series.

#### Setup Instructions

**Note:** Ensure you are not in a virtual environment and have Python 3 installed on your system.

##### For Windows and macOS
```julia
using Pkg
Pkg.add("PyCall")
Pkg.add("ArgParse")

using Downloads
Downloads.download("https://gist.githubusercontent.com/divital-coder/6d2dc6868ee5f8427cf719f54213227d/raw/a3b50aa734c662e0a4e57be6e29f69fc22921d03/reorient_to_lps.jl", "./reorient_to_lps.jl")
python_path = Sys.which("python")
if isnothing(python_path)
  error("Python not found in PATH, please install python or add it to your PATH")
else
   ENV["PYTHON"] = python_path
end   
run(`pip install SimpleITK`)
Pkg.build("PyCall")
```

##### For Arch Linux
```julia
using Pkg
Pkg.add("PyCall")
Pkg.add("ArgParse")

using Downloads
Downloads.download("https://gist.githubusercontent.com/divital-coder/6d2dc6868ee5f8427cf719f54213227d/raw/a3b50aa734c662e0a4e57be6e29f69fc22921d03/reorient_to_lps.jl", "./reorient_to_lps.jl")
python_path = Sys.which("python")
if isnothing(python_path)
  error("Python not found in PATH, please install python or add it to your PATH")
else
   ENV["PYTHON"] = python_path
end   
run(`yay -S python-simpleitk`)
Pkg.build("PyCall")
```

#### Usage Examples

##### Convert a NIfTI File
```bash
julia reorient_to_lps.jl input_image.nii.gz output_lps.nii.gz
```

##### Convert a DICOM Series
```bash
julia reorient_to_lps.jl /path/to/dicom/folder/ output_lps.nii.gz
```

##### Additional Options
```bash
# Enable verbose output
julia reorient_to_lps.jl brain.nii.gz brain_lps.nii.gz --verbose

# Force overwrite existing files
julia reorient_to_lps.jl brain.nii.gz brain_lps.nii.gz --force
```

### Converting Back to DICOM Format

**Important:** The orientation converter outputs NIfTI files only. If you need your processed images back in DICOM format, you can use the [`ITKIOWrapper.jl`](https://github.com/JuliaHealth/ITKIOWrapper.jl) package for conversion.

#### Installing ITKIOWrapper.jl

```julia
using Pkg
Pkg.add("ITKIOWrapper")
```

#### Converting NIfTI to DICOM Series

After processing your LPS-oriented NIfTI file with MedImages.jl, convert it back to DICOM:

```julia
using ITKIOWrapper

# Convert processed NIfTI back to DICOM series
# This creates a directory with DICOM files
dicom_nifti_conversion("processed_lps_image.nii.gz", "./output_dicom_series", true)
```

#### Complete Workflow Example

Here's a typical workflow for DICOM users:

```julia
using MedImages, ITKIOWrapper

# 1. Convert original DICOM to LPS-oriented NIfTI (using the orientation script)
# julia reorient_to_lps.jl /path/to/original/dicom/ lps_oriented.nii.gz

# 2. Process with MedImages.jl
medimage = MedImages.load_image("lps_oriented.nii.gz")
# ... perform your analysis and modifications ...

# 3. Save processed image (if modified)
# save_image(modified_medimage, "processed_image.nii.gz")

# 4. Convert back to DICOM series if needed
dicom_nifti_conversion("processed_image.nii.gz", "./final_dicom_series", true)
```

#### Advanced DICOM Conversion Options

For more control over the DICOM output, you can also create images programmatically:

```julia
using ITKIOWrapper

# Load your processed NIfTI image
img = load_image("processed_lps_image.nii.gz")
metadata = load_spatial_metadata(img)
voxel_data = load_voxel_data(img, metadata)

# Save as DICOM series with full control
save_image(voxel_data, metadata, "./custom_dicom_output", true)
```

### Summary

1. **Input DICOM** → Convert to LPS NIfTI using the orientation script
2. **Process** → Use MedImages.jl for analysis and modifications
3. **Output DICOM** → Use ITKIOWrapper.jl to convert back to DICOM format

Once your images are in LPS orientation, you can use them seamlessly with MedImages.jl for optimal performance and compatibility, and easily convert between NIfTI and DICOM formats as needed for your workflow.