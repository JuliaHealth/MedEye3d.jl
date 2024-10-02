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

    while ((mean(displayedTextureInfo.voxelData[1]) == -1) || totalSleep > 5.0)
        sleep(0.1)
        totalSleep += 0.0
    end
    return typeof(textNumb) == Int32 ? displayedTextureInfo.voxelData[1] : displayedTextureInfo.voxelData
end



"""
function invoked on on_next
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



function depositVoxelArray(
    modifiedData::CustomDisplayedVoxels,
    stateData::Vector{StateDataFields}
)
    stateData = stateDate[stateData[1].switchIndex]
    for (index, modifArray) in enumerate(modifiedData.voxelData)
        stateData.onScrollData.dataToScroll[index].dat = modifArray
    end

    # modifiedData.scrollDat = stateData.onScrollData
end

end # end of DisplayDataManag.jl
