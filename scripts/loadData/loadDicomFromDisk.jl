using DrWatson
using Dicom
@quickactivate "Probabilistic medical segmentation"

```@doc
Loading Dicom data  into julia Arrays
```
module loadDicomFromDisk

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



end # module