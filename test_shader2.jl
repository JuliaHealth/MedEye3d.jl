using MedEye3d
using MedEye3d.ShadersAndVerticies
using MedEye3d.ForDisplayStructs

# Create dummy TextureSpecs
textures = TextureSpec{Float32}[]
push!(textures, TextureSpec{Float32}(name="CT", isMainImage=true, isContinuusMask=true))
push!(textures, TextureSpec{Float32}(name="PET", isMainImage=true, isContinuusMask=true))
push!(textures, TextureSpec{Float32}(name="Anatomy", isMainImage=false, isContinuusMask=false, isMultiDiscreteMask=true))

source = MedEye3d.CustomFragShad.createCustomFramgentShader(textures, "r")
println("Shader code generated successfully (length: ", length(source), ")")
