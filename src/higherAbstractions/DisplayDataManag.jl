module DisplayDataManag

using ColorTypes, MedImages, ModernGL, GLFW, FreeTypeAbstraction, Base.Threads, Observables, Statistics
using ..ForDisplayStructs
export retrieveVoxelArray, getDisplayedData, setDisplayedData, depositVoxelArray

"""
Simplfication function :exposed to the user
"""
function getDisplayedData(
    medEyeStruct::MainMedEye3d,
    textNumb::Union{Int32,Vector{Int32}}
)

    initialVoxelData = map(num -> ones(medEyeStruct.voxelArrayTypes[num], medEyeStruct.voxelArrayShapes[num]), textNumb)
    initialVoxelData = typeof(textNumb) == Int32 ? [initialVoxelData] : initialVoxelData

    displayedTextureInfo = DisplayedVoxels(activeNumb=textNumb, voxelData=initialVoxelData)

    for voxelArray in displayedTextureInfo.voxelData
        voxelArray[:, :, :] .-= 2 #subtracting 2 so we get all -1s
    end
    put!(medEyeStruct.channel, displayedTextureInfo)
    totalSleep = 0
    @info "Program going in sleep"
    while ((mean(displayedTextureInfo.voxelData[1]) == -1) || totalSleep > 5.0)
        sleep(0.1)
        totalSleep += 0.0
    end
    @info "Program out of sleep"
    return typeof(textNumb) == Int32 ? displayedTextureInfo.voxelData[1] : displayedTextureInfo.voxelData
end



"""
function invoked on on_next
Since the voxel array modification feature is only provided in single image display,
the switchIndex attribute of the stateData will be defaulted to 1
"""
function retrieveVoxelArray(
    activeTexture::DisplayedVoxels,
    stateData::Vector{StateDataFields}
)
    stateData = stateData[stateData[1].switchIndex]

    if typeof(activeTexture.activeNumb) == Int32
        activeTexture.voxelData[1][:, :, :] = stateData.onScrollData.dataToScroll[activeTexture.activeNumb].dat
    else
        for numb in activeTexture.activeNumb
            activeTexture.voxelData[numb][:, :, :] = stateData.onScrollData.dataToScroll[numb].dat
        end
    end
    # activeTexture.voxelData[1][:, :, :] = stateData.onScrollData.dataToScroll[activeTexture.activeNumb].dat
end



"""
Simplification function :exposed to the user
"""
function setDisplayedData(
    medEyeStruct::MainMedEye3d,
    displayData::Union{Vector{Array{Float32,3}},Array{Float32,3}}
)
    customVoxels = typeof(displayData) == Array{Float32,3} ? CustomDisplayedVoxels(voxelData=[displayData]) : CustomDisplayedVoxels(voxelData=displayData)
    put!(medEyeStruct.channel, customVoxels)

    # sleep(2)
    # put!(medEyeStruct.channel, customVoxels.scrollDat)
end


"""
Function for the deposition of the modified voxels on the screen.
Since the voxel array modification feature is only provided in single image display,
the switchIndex attribute of the stateData will be defaulted to 1
"""
function depositVoxelArray(
    modifiedData::CustomDisplayedVoxels,
    stateData::Vector{StateDataFields}
)
    stateData = stateData[stateData[1].switchIndex]
    for (index, modifArray) in enumerate(modifiedData.voxelData)
        stateData.onScrollData.dataToScroll[index].dat = modifArray
    end

    # modifiedData.scrollDat = stateData.onScrollData
end

end # end of DisplayDataManag.jl
