using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.CustomFragShad
using ColorTypes

textureSpec_img = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-100, 200])
)
textureSpec_mask = TextureSpec{Float32}(
    name="Mask",
    isMainImage=false,
    color=RGB(1.0, 0.0, 0.0),
    maskContribution=Float32(1.0)
)
shader = CustomFragShad.createCustomFramgentShader([textureSpec_img, textureSpec_mask], "red")
println(shader)
