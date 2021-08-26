
"""
module for handling dicom files adapted from 
    https://github.com/JuliaHealth/DICOM.jl/issues/68
"""
module DicomManage

export load_dicom,extract_pixeldata,sort_slices,get_gridpoints,interpolate_to

using DICOM,Interpolations
function load_dicom(dir)
    dcms = dcmdir_parse(dir)
    loaded_dcms = Dict()
    # 'dcms' could contain data for different series, so we have to filter by series
    unique_series = unique([dcm.SeriesInstanceUID for dcm in dcms])
    for (idx, series) in enumerate(unique_series)
        dcms_in_series = filter(dcm -> dcm.SeriesInstanceUID == series, dcms)
        pixeldata = extract_pixeldata(dcms_in_series)
        loaded_dcms[idx] = (; pixeldata = pixeldata, dcms = dcms_in_series)
    end
    return loaded_dcms
end



function extract_pixeldata(dcm_array)
    if length(dcm_array) == 1
        return only(dcm_array).PixelData
    else
        return cat([dcm.PixelData for dcm in dcm_array]...; dims = 3)
    end
end


# This ensures that first slice has lowest position, and last slice has highest position
# This assumption is required for `get_gridpoints()` to return correct slice locations
function sort_slices(scan)
    slicelocations = [dcm.SliceLocation for dcm in scan.dcms]
    if issorted(slicelocations)
        return scan
    elseif issorted(slicelocations; rev = true)
        pixeldata = reverse(scan.pixeldata; dims = 3)
        dcms = reverse(scan.dcms)
        return (; pixeldata, dcms)        
    end
    error("Could not sort scan")
end

function get_gridpoints(scan)
    @assert ndims(scan.pixeldata) == 3
    dcm = first(scan.dcms)
    # [!] I might have gotten the x and y backwards below
    # but this might not matter if nx = ny
    (ox, oy, oz) = dcm.ImagePositionPatient
    (dx, dy) = dcm.PixelSpacing
    dz = dcm.SliceThickness
    (nx, ny, nz) = size(scan.pixeldata)
    x = ox:dx:ox+dx*(nx-1)
    y = oy:dy:oy+dy*(ny-1)
    z = oz:dz:oz+dz*(nz-1)
    return (x, y, z)
end

function interpolate_to(input, target)
    targetgrid = get_gridpoints(target)
    inputgrid = get_gridpoints(input)
    itp = LinearInterpolation(inputgrid, input.pixeldata, extrapolation_bc = Line())
    return itp((targetgrid)...)
end

end#module