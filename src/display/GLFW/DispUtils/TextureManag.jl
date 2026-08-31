"""
TextureManag — Vulkan texture management.
Replaces OpenGL texture operations with Vulkan equivalents via VulkanTextures module.
"""
module TextureManag
using Base: Float16
using GLFW
using Base.Threads, Logging, Setfield
using ..OpenGLDisplayUtils, ..ForDisplayStructs, ..Uniforms, ..CustomFragShad, ..DataStructs, ..DisplayWords, ..StructsManag
export activateTextures, addTextToTexture, initializeTextures, createTexture, getProperGL_TEXTURE, updateImagesDisplayed, updateTexture, assignUniformsAndTypesToMasks, setZoomPanUniforms


"""
Upload data to a given texture via Vulkan staging buffer.
Replaces OpenGL glTexSubImage2D.
"""
function updateTexture(::Type{Tt}, data::AbstractArray, textSpec::TextureSpec, xoffset::Int, yoffset::Int, widthh::Int32, heightt::Int32) where {Tt}
    # In Vulkan backend, this is handled by the consumer loop which calls
    # VulkanTextures.update_vulkan_texture! after finding the matching VkTexture
    # The texture spec just stores the data reference; actual upload happens in consumer
end


"""
Create a texture (stub — Vulkan textures are created during coordinateDisplay init).
"""
function createTexture(juliaDataType::Type{juliaDataTyp}, width::Int32, height::Int32, GL_RType::UInt32=UInt32(0), OpGlType=UInt32(0); forceNearest::Bool=false) where {juliaDataTyp}
    return Ref(UInt32(0))
end


"""
Initialize textures — sets up TextureSpec metadata without OpenGL calls.
"""
function initializeTextures(listOfTextSpecs, calcDimStruct::CalcDimsStruct)::Vector{TextureSpec}
    res = Vector{TextureSpec}()
    for (ind, textSpec) in enumerate(listOfTextSpecs)
        index = ind - 1
        actTextrureNumb = UInt32(index)  # Simple index, no GL_TEXTURE enum needed
        push!(res, setproperties(textSpec, (
            ID=Ref(UInt32(ind)),
            actTextrureNumb=actTextrureNumb,
            associatedActiveNumer=index,
            colorMask=RGBA(textSpec.color.r, textSpec.color.g, textSpec.color.b, 1.0)
        )))
    end
    return res
end


"""
Activate textures — no-op in Vulkan (descriptor sets handle binding).
"""
function activateTextures(listOfTextSpecs::Vector{TextureSpec})::Vector{TextureSpec}
    return listOfTextSpecs
end


"""
Get texture unit index (stub, not used by Vulkan).
"""
function getProperGL_TEXTURE(index::Int)::UInt32
    return UInt32(index)
end


"""
Set zoom/pan uniforms — no-op in Vulkan (uses push constants).
"""
function setZoomPanUniforms(forDispObj, calcDims)
    # No-op: Vulkan uses push constants for zoom/pan in VulkanRender.render_frame!
end


"""
Crosshair display — no-op, crosshair rendering removed.
"""
function crosshairDisplay(crosshair, mainRect, forDisplayConstants)
    # No-op: crosshair rendering not yet ported to Vulkan
end

"""
Update images displayed — Vulkan texture upload via forDisplayObjects.vulkanCtx.

For Vulkan: finds matching VkTexture by name and calls update_vulkan_texture!
"""
function updateImagesDisplayed(
    singleSliceDat::SingleSliceDat,
    forDisplayConstants::forDisplayObjects,
    wordsDispObj::ForWordsDispStruct,
    calcDimStruct::CalcDimsStruct,
    valueForMaskToSett::valueForMasToSetStruct,
    crosshair::GlShaderAndBufferFields,
    mainRect::GlShaderAndBufferFields,
    displayMode::DisplayMode)

    modulelistOfTextSpecs = forDisplayConstants.listOfTextSpecifications
    ctx = forDisplayConstants.vulkanCtx

    if ctx === nothing
        return  # No Vulkan context, skip
    end

    for updateDat in singleSliceDat.listOfDataAndImageNames
        # Find matching VkTexture by name
        for (i, vk_tex) in enumerate(forDisplayConstants.vulkanTextures)
            if hasproperty(vk_tex, :name) && vk_tex.name == updateDat.name
                try
                    # Convert data to Float32 for Vulkan R32_SFLOAT format
                    upload_data = Float32.(updateDat.dat)
                    # Use VulkanTextures module from VulkanBackend
                    # Access via the module hierarchy
                    Base.invokelatest(
                        getfield(parentmodule(parentmodule(@__MODULE__)), :VulkanBackend).VulkanTextures.update_vulkan_texture!,
                        ctx, vk_tex, upload_data
                    )
                catch e
                    @warn "Vulkan texture upload failed for $(updateDat.name): $e"
                end
                break
            end
        end
    end
end

"""
Assign uniforms and types to masks — sets up TextureSpec metadata.
In Vulkan, this just does the type mapping without OpenGL uniform queries.
"""
function assignUniformsAndTypesToMasks(textSpecs::Vector{TextureSpec{Float32}}, shader_program::UInt32)
    return map(x -> setProperOpenGlTypes(x), textSpecs)
end


"""
Set Julia data type → OpenGL type mapping on TextureSpec.
Kept for compatibility, though Vulkan uses FORMAT enums instead.
"""
function setProperOpenGlTypes(textSpec::TextureSpec)::TextureSpec
    # Just return the spec as-is — Vulkan format mapping is handled elsewhere
    return textSpec
end

"""
Add text to texture — no-op, text rendering removed.
"""
function addTextToTexture(wordsDispObj::ForWordsDispStruct, lines::Vector{SimpleLineTextStruct}, calcDimStruct::CalcDimsStruct)
    return nothing
end

end #..TextureManag
