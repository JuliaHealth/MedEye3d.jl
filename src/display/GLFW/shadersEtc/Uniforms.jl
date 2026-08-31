"""
Uniforms stub — OpenGL uniform management removed.
Vulkan uses UBO (Uniform Buffer Objects) via VulkanPipeline.update_ubo!.
TextureSpec fields are modified directly and the consumer loop reads them.
"""
module Uniforms
using StaticArrays, Dictionaries, Parameters, ColorTypes
using ..ForDisplayStructs

export isMaskDiffViss, changeMainTextureContribution, changeTextureContribution
export coontrolMinMaxUniformVals, createStructsDict, setCTWindow, setMaskColor
export setTextureVisibility, setTypeOfMainSampler!, setAllowedIDs!
export setTextureContribution

# All these functions are no-ops in the Vulkan backend.
# TextureSpec fields are modified directly by event handlers,
# and the consumer loop calls VulkanPipeline.update_ubo! to push them to the GPU.

function setMaskColor(color::RGB, uniformsStore::MaskTextureUniforms)
    # No-op: color read from TextureSpec.colorMask by UBO packer
end

function setTextureVisibility(isvisible::Bool, uniformsStore::TextureUniforms)
    # No-op: visibility read from TextureSpec.isVisible by UBO packer
end

function coontrolMinMaxUniformVals(textur::TextureSpec)
    # No-op: min/max read from TextureSpec.minAndMaxValue by UBO packer
end

function changeTextureContribution(textur::TextureSpec, change::Float32)
    newValue = textur.maskContribution + change
    if (newValue >= 0 && newValue <= 1)
        textur.maskContribution = newValue
    end
end

function setTextureContribution(textur::TextureSpec, value::Float32)
    textur.maskContribution = clamp(value, 0.0f0, 1.0f0)
end

function changeMainTextureContribution(textur::TextureSpec, change::Float32, stateObject::StateDataFields)
    newValue = textur.maskContribution + change
    if (newValue >= 0 && newValue <= 1)
        textur.maskContribution = newValue
    end
end

function setAllowedIDs!(textur::TextureSpec, ids::Vector{Int})
    textur.allowedIDs = Float32.(ids)
end

end # module Uniforms
