
using Pkg

using Documenter
using MedEye3d

makedocs(
    sitename = "MedEye3d",
    format = Documenter.HTML(),
    modules = [MedEye3d
    ,MedEye3d.SegmentationDisplay
    ,MedEye3d.ReactingToInput
    ,MedEye3d.ReactOnKeyboard
    ,MedEye3d.ReactOnMouseClickAndDrag
    ,MedEye3d.ReactToScroll
    ,MedEye3d.PrepareWindow
    ,MedEye3d.TextureManag
    ,MedEye3d.DisplayWords
    ,MedEye3d.Uniforms
    ,MedEye3d.ShadersAndVerticiesForText
    ,MedEye3d.ShadersAndVerticies
    ,MedEye3d.OpenGLDisplayUtils
    ,MedEye3d.CustomFragShad
    ,MedEye3d.PrepareWindowHelpers
    ,MedEye3d.StructsManag
    ,MedEye3d.ForDisplayStructs
    ,MedEye3d.DataStructs
    ,MedEye3d.BasicStructs
    ,MedEye3d.ModernGlUtil
    ,MedEye3d.MaskDiffrence
    ,MedEye3d.KeyboardVisibility
    ,MedEye3d.OtherKeyboardActions
    ,MedEye3d.WindowControll
    ,MedEye3d.ChangePlane
    
    
    ]
)

