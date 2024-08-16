using Conda, PyCall, Pkg
using Statistics
Conda.pip_interop(true)
Conda.pip("install", "SimpleITK")
Conda.pip("install", "h5py")
sitk = pyimport("SimpleITK")
np = pyimport("numpy")

path = "D:/mingw_installation/home/hurtbadly/Downloads/ct_soft_pat_3_sudy_0.nii.gz"
# Step 2: Read the NIfTI image
image = sitk.ReadImage(path)

# Step 3: Define the new spacing
new_spacing = [1.0, 2.0, 3.0]

# Step 4: Compute the new size
original_size = size(np.array(sitk.GetArrayViewFromImage(image)))
original_spacing = image.GetSpacing()
new_size = [round(Int, original_size[i] * (original_spacing[i] / new_spacing[i])) for i in 1:3]

# Convert new_size to a Python list of uint32 using PyVector
new_size_py = PyVector(np.array(new_size, dtype=np.uint32))

# Step 5: Define the resampling parameters
resample = sitk.ResampleImageFilter()
resample.SetOutputSpacing(new_spacing)
resample.SetSize(new_size_py)
resample.SetInterpolator(sitk.sitkLinear)
resample.SetOutputDirection(image.GetDirection())
resample.SetOutputOrigin(image.GetOrigin())

# Step 6: Resample the image
resampled_image = resample.Execute(image)

# Step 7: Save or use the resampled image
sitk.WriteImage(resampled_image, "D:/mingw_installation/home/hurtbadly/Downloads/resampled_image.nii.gz")
