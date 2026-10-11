using MedImages, CUDA
function gpu_resample_to_image(im_fixed::MedImages.MedImage, im_moving::MedImages.MedImage, interp::MedImages.Interpolator_enum)
    gpu_fixed_vox = CuArray(Float32.(im_fixed.voxel_data))
    gpu_moving_vox = CuArray(Float32.(im_moving.voxel_data))
    gpu_fixed = MedImages.update_voxel_data(im_fixed, gpu_fixed_vox)
    gpu_moving = MedImages.update_voxel_data(im_moving, gpu_moving_vox)
    gpu_result = MedImages.resample_to_image(gpu_fixed, gpu_moving, interp)
    cpu_result_vox = Array(gpu_result.voxel_data)
    cpu_result = MedImages.update_voxel_data(gpu_result, cpu_result_vox)
    
    CUDA.unsafe_free!(gpu_fixed_vox)
    CUDA.unsafe_free!(gpu_moving_vox)
    CUDA.unsafe_free!(gpu_result.voxel_data)
    CUDA.reclaim()
    
    return cpu_result
end
