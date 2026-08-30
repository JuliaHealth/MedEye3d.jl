module ForDisplayStructs
using Base: Int32, isvisible
export MouseStruct, parameter_type, Mask, TextureSpec, forDisplayObjects, StateDataFields, KeyboardStruct, KeyInputFields, TextureUniforms, MaskTextureUniforms, ForWordsDispStruct, MainMedEye3d
export DisplayedVoxels, CustomDisplayedVoxels, DisplayMode, SingleImage, MultiImage, QuadImage, GlShaderAndBufferFields
export DoubleClickEvent
using ColorTypes, Parameters, Observables, ModernGL, GLFW, Dictionaries, FreeTypeAbstraction, Observables
using ..DataStructs

struct ToggleSyncScroll end
export ToggleSyncScroll


"""
Display mode of the MedEye3d visualizer layout.
- `SingleImage`: A single full-screen 2D slicing plane.
- `MultiImage`: Two side-by-side 2D slices for comparative viewing.
- `QuadImage`: A four-panel layout (typically Axial, Coronal, Sagittal, and a 4th custom view like pure PET or a Compare mode timepoint).
"""
@enum DisplayMode begin
  SingleImage
  MultiImage
  QuadImage
end


"""
    Mask{arrayType}

Data structure defining an overlay mask to be displayed on top of the main radiological image.

# Fields
- `path::String`: Path to the mask in the HDF5 backend (if applicable).
- `maskId::Int64`: Unique identifier associated with this mask.
- `maskName::String`: Name of the anatomical structure or annotation class (e.g., "Liver", "Lesion").
- `maskArrayObs::Observable{Array{arrayType}}`: An active Observable wrapping the 3D voxel array holding the mask data, updating dynamically upon drawing/erasing.
- `colorRGBA::RGBA`: The designated rendering color of the mask.
"""
@with_kw struct Mask{arrayType}
  path::String 
  maskId::Int64 
  maskName::String 
  maskArrayObs::Observable{Array{arrayType}} 
  colorRGBA::RGBA 
end

"""
Abstract type holding references to OpenGL uniform variables used to control textures dynamically in shaders.
"""
abstract type TextureUniforms end


"""
    MaskTextureUniforms <: TextureUniforms

Holds reference pointers to OpenGL uniforms for a specific mask or layer.
Stores references needed for rapid GL context updates (visibility, blending, clip ranges) during interactive rendering without costly CPU-GPU sync overhead.
"""
@with_kw mutable struct MaskTextureUniforms <: TextureUniforms
  samplerName::String = ""
  samplerRef::Int32 = Int32(0)
  colorsMaskRef::Int32 = Int32(0) 
  isVisibleRef::Int32 = Int32(0)
  maskMinValue::Float32 = Float32(0)
  maskMAxValue::Float32 = Float32(0)
  maskRangeValue::Float32 = Float32(0)
  maskContribution::Float32 = Float32(0)
  allowedIDsRef::Vector{Int32} = fill(Int32(-1), 16)
  allowedIDCountRef::Int32 = Int32(-1)
end


"""
    TextureSpec{T}

Defines the specification and configuration parameters for an OpenGL texture layer (main volume or overlay mask).

# Fields
- `name::String`: Human-readable identifier.
- `numb::Int32`: Optional numeric ID for keyboard shortcut toggling (0-9).
- `isMainImage::Bool`: Flags if this layer is the base structural image (e.g., CT).
- `isNuclearMask::Bool`: Flags if this is a functional imaging layer (e.g., PET) needing specialized crossfade blending.
- `isContinuusMask::Bool`: Flags continuous color mapping over a value range.
- `isMultiDiscreteMask::Bool`: Maps discrete integer labels to `colorSet` palettes.
- `color::RGB`: The primary color mapping (for binary masks or single-hue overlays).
- `strokeWidth::Int32`: Thickness of interactive paint/erase strokes on this mask.
- `isEditable::Bool`: Enables live mouse annotation interactions.
- `GL_Rtype::UInt32`: OpenGL texture format (e.g. `GL_R8UI`).
- `OpGlType::UInt32`: OpenGL data type (e.g. `GL_UNSIGNED_BYTE`).
"""
@with_kw mutable struct TextureSpec{T}
  name::String = ""
  numb::Int32 = -1
  whichCreated::Int32 = -1
  isMainImage::Bool = false
  isNuclearMask::Bool = false
  isContinuusMask::Bool = false
  isMultiDiscreteMask::Bool = false
  color::RGB = RGB(0.0, 0.0, 0.0)
  colorSet::Vector{RGB} = []
  strokeWidth::Int32 = Int32(3)
  isEditable::Bool = false
  GL_Rtype::UInt32 = UInt32(0)
  OpGlType::UInt32 = UInt32(0)
  actTextrureNumb::UInt32 = UInt32(0)
  associatedActiveNumer::Int64 = Int64(0)
  ID::Base.RefValue{UInt32} = Ref(UInt32(0))
  isVisible::Bool = true
  uniforms::TextureUniforms = MaskTextureUniforms()
  minAndMaxValue::Vector{T} = []#entry one is minimum possible value for this mask, and second entry is maximum possible value for this mask
  maskContribution::Float32 = 1.0 # controlls contribution  of given mask to the overall image - maximum value is 1 minimum 0 if we have 3 masks and all control contribution is set to 1 and all are visible their corresponding influence to pixel color is 33%
  studyType::String = "" #type of the study - for example CT, MRI, PET, SPECT
end

#utility function to check type associated
parameter_type(::Type{TextureSpec{T}}) where {T} = T
parameter_type(x::TextureSpec) = parameter_type(typeof(x))

"""
given Vector of TextureSpecs
it creates dictionary where keys are associated names
and values are indicies where they are found in a list
"""
function getLocationDict(listt)::Dictionary{String,Int64}
  return Dictionary(map(it -> it.name, listt), collect(eachindex(listt)))

end#getLocationDict


"""
Defined in order to hold constant objects needed to display images
listOfTextSpecifications::Vector{TextureSpec} = [TextureSpec()]
window = []
vertex_shader::UInt32 =1
fragment_shader::UInt32=1
shader_program::UInt32=1
vbo::UInt32 =1 #vertex buffer object id
ebo::UInt32 =1 #element buffer object id
mainImageUniforms::MainImageUniforms = MainImageUniforms()# struct with references to main image
TextureIndexes::Dictionary{String, Int64}=Dictionary{String, Int64}()  #gives a way of efficient querying by supplying dictionary where key is a name we are intrested in and a key is index where it is located in our array
numIndexes::Dictionary{Int32, Int64} =Dictionary{Int32, Int64}() # a way for fast query using assigned numbers
gslsStr::String="" # string giving information about used openg gl gsls version
windowControlStruct::WindowControlStruct=WindowControlStruct()# holding data usefull to controll display window


"""
@with_kw mutable struct forDisplayObjects
  listOfTextSpecifications::Vector{TextureSpec} = [TextureSpec()]
  window = []
  vertex_shader::UInt32 = 1
  fragment_shader::UInt32 = 1
  shader_program::UInt32 = 1
  vbo::UInt32 = 1 #vertex buffer object id
  ebo::UInt32 = 1 #element buffer object id
  imageUniforms::MaskTextureUniforms = MaskTextureUniforms() #we can pass all texture uniforms here, ideally we would like to make it a vector
  TextureIndexes::Dictionary{String,Int64} = Dictionary{String,Int64}()  #gives a way of efficient querying by supplying dictionary where key is a name we are intrested in and a key is index where it is located in our array
  numIndexes::Dictionary{Int32,Int64} = Dictionary{Int32,Int64}() # a way for fast query using assigned numbers
  gslsStr::String = "" # string giving information about used openg gl gsls version
  windowControlStruct::WindowControlStruct = WindowControlStruct()# holding data usefull to controll display window
  isFastScroll::Bool = false # set by f letter to true and by s to normal - slow
  imagePos::Int64 = 1
  isSyncScrollOn::Bool = true # toggled by pressing c
  isCrosshairVisible::Bool = false # toggled by Makie GUI
  uvScaleRef::Int32 = Int32(-1)   # uniform location for vec2 uvScale (GPU zoom)
  uvOffsetRef::Int32 = Int32(-1)  # uniform location for vec2 uvOffset (GPU pan)
end


"""
Holding necessery data to display text  - like font related
"""
@with_kw struct ForWordsDispStruct
    fontFace::Union{FTFont, Nothing} = begin
        try
            # Check if we should disable fonts
            if haskey(ENV, "JULIA_FREETYPE_NO_FONTCONFIG")
                nothing
            else
                FTFont()
            end
        catch e
            @warn "Font initialization failed, disabling text rendering: $e"
            nothing
        end
    end
  textureSpec::TextureSpec = TextureSpec{UInt8}() # texture specification of texture used to display text
  fragment_shader_words::UInt32 = 1 #reference to fragment shader used to display text
  vbo_words::Base.RefValue{UInt32} = Ref(UInt32(1)) #reference to vertex buffer object used to display text
  shader_program_words::UInt32 = 1
end


"""
Holding necessery data to controll keyboard shortcuts

isCtrlPressed::Bool = false# left - scancode 37 right 105 - Int32
isShiftPressed::Bool = false # left - scancode 50 right 62- Int32
isAltPressed::Bool= false# left - scancode 64 right 108- Int32
isEnterPressed::Bool= false# scancode 36
isTAbPressed::Bool= false#
isSpacePressed::Bool= false#
isF1Pressed::Bool= false
isF2Pressed::Bool= false
isF3Pressed::Bool= false

lastKeysPressed::Vector{String}=[] # last pressed keys - it listenes to keys only if ctrl/shift or alt is pressed- it clears when we release those case or when we press enter
#informations about what triggered sending this particular struct to the  actor
mostRecentScanCode ::GLFW.Key=GLFW.KEY_KP_4
mostRecentKeyName ::String=""
mostRecentAction ::GLFW.Action= GLFW.RELEASE

"""
@with_kw mutable struct KeyboardStruct
  isCtrlPressed::Bool = false# left - scancode 37 right 105 - Int32
  isShiftPressed::Bool = false # left - scancode 50 right 62- Int32
  isAltPressed::Bool = false# left - scancode 64 right 108- Int32
  isEnterPressed::Bool = false# scancode 36
  isTAbPressed::Bool = false#
  isSpacePressed::Bool = false#
  isF1Pressed::Bool = false
  isF2Pressed::Bool = false
  isF3Pressed::Bool = false
  isF4Pressed::Bool = false
  isF5Pressed::Bool = false
  isF6Pressed::Bool = false
  isF7Pressed::Bool = false
  isF8Pressed::Bool = false
  isF9Pressed::Bool = false
  isPlusPressed::Bool = false
  isMinusPressed::Bool = false
  isZPressed::Bool = false
  isFPressed::Bool = false
  isSPressed::Bool = false
  isCPressed::Bool = false
  lastKeysPressed::Vector{String} = [] # last pressed keys - it listenes to keys only if ctrl/shift or alt is pressed- it clears when we release those case or when we press enter
  #informations about what triggered sending this particular struct to the  actor
  mostRecentScanCode::Int32 = Int32(GLFW.KEY_KP_4)
  mostRecentKeyName::String = ""
  mostRecentAction::GLFW.Action = GLFW.RELEASE

end

"""
Holding necessery data to controll mouse interaction
"""
@with_kw mutable struct MouseStruct
  isLeftButtonDown::Bool = false # true if left button was pressed and not yet released
  isRightButtonDown::Bool = false# true if right button was pressed and not yet released
  lastCoordinates::Vector{CartesianIndex{2}} = [] # list of accumulated mouse coordinates
  actualWindowWidth::Int = 0  # actual GLFW window content area width
  actualWindowHeight::Int = 0  # actual GLFW window content area height
end#MouseStruct

"""
Event fired when a double left-click is detected.
Dispatched as a dedicated type via on_next! — same pattern as KeyInputFields, MouseStruct, etc.
"""
@with_kw mutable struct DoubleClickEvent
  x::Int = 0                    # cursor x at time of click
  y::Int = 0                    # cursor y at time of click
  actualWindowWidth::Int = 0
  actualWindowHeight::Int = 0
end#DoubleClickEvent


"""
Structure for handling key input
"""
@with_kw struct KeyInputFields
  scancode::Int32
  action::GLFW.Action
end


"""
Fields for the display of crosshair on screen
"""
@with_kw mutable struct GlShaderAndBufferFields
  shaderProgram::UInt32 = UInt32(1)
  fragmentShader::UInt32 = UInt32(1)
  vao::Union{Ref{UInt32},Int32} = Int32(1)
  vbo::Union{Ref{UInt32},Int32} = Int32(1)
  ebo::Union{Ref{UInt32},Int32} = Int32(1)
end



"""
Actor that is able to store a state to keep needed data for proper display

  currentDisplayedSlice::Int=1 # stores information what slice number we are currently displaying
    mainForDisplayObjects:: forDisplayObjects=forDisplayObjects() # stores objects needed to  display using OpenGL and GLFW
    onScrollData::FullScrollableDat = FullScrollableDat()
    textureToModifyVec::Vector{TextureSpec}=[] # texture that we want currently to modify - if list is empty it means that we do not intend to modify any texture
    isSliceChanged::Bool= false # set to true when slice is changed set to false when we start interacting with this slice - thanks to this we know that when we start drawing on one slice and change the slice the line would star a new on new slice
    textDispObj::ForWordsDispStruct =ForWordsDispStruct()# set of objects and constants needed for text diplay
    currentlyDispDat::SingleSliceDat =SingleSliceDat() # holds the data displayed or in case of scrollable data view for accessing it
    calcDimsStruct::CalcDimsStruct=CalcDimsStruct()   #data for calculations of necessary constants needed to calculate window size , mouse position ...
    valueForMasToSet::valueForMasToSetStruct=valueForMasToSetStruct() # value that will be used to set  pixels where we would interact with mouse
    lastRecordedMousePosition::CartesianIndex{3} = CartesianIndex(1,1,1) # last position of the mouse  related to right click - usefull to know onto which slice to change when dimensions of scroll change
    forUndoVector::AbstractArray=[] # holds lambda functions that when invoked will  undo last operations
    maxLengthOfForUndoVector::Int64 = 10 # number controls how many step at maximum we can get back
    isBusy::Base.Threads.Atomic{Bool}= Threads.Atomic{Bool}(0) # used to indicate by some functions that actor is busy and some interactions should be ceased


"""
@with_kw mutable struct StateDataFields
  currentDisplayedSlice::Int = 1 # stores information what slice number we are currently displaying
  mainForDisplayObjects::forDisplayObjects = forDisplayObjects() # stores objects needed to  display using OpenGL and GLFW
  onScrollData::FullScrollableDat = FullScrollableDat()
  textureToModifyVec::Vector{TextureSpec} = [] # texture that we want currently to modify - if list is empty it means that we do not intend to modify any texture
  isSliceChanged::Bool = false # set to true when slice is changed set to false when we start interacting with this slice - thanks to this we know that when we start drawing on one slice and change the slice the line would star a new on new slice
  textDispObj::ForWordsDispStruct = ForWordsDispStruct()# set of objects and constants needed for text diplay
  currentlyDispDat::SingleSliceDat = SingleSliceDat() # holds the data displayed or in case of scrollable data view for accessing it
  calcDimsStruct::CalcDimsStruct = CalcDimsStruct()   #data for calculations of necessary constants needed to calculate window size , mouse position ...
  valueForMasToSet::valueForMasToSetStruct = valueForMasToSetStruct() # value that will be used to set  pixels where we would interact with mouse
  lastRecordedMousePosition::CartesianIndex{3} = CartesianIndex(1, 1, 1) # last position of the mouse  related to right click - usefull to know onto which slice to change when dimensions of scroll change
  lastPanDragCoords::Vector{CartesianIndex{2}} = [] # last position of the mouse for right click drag panning
  lastPaintCoords::Vector{CartesianIndex{2}} = [] # last position of the mouse for continuous brush painting/erasing
  forUndoVector::AbstractArray = [] # holds lambda functions that when invoked will  undo last operations
  maxLengthOfForUndoVector::Int64 = 15 # number controls how many step at maximum we can get back
  fieldKeyboardStruct::KeyboardStruct = KeyboardStruct()
  displayMode::DisplayMode = SingleImage
  imagePosition::Int64 = 1
  switchIndex::Int = 1
  mainRectFields::GlShaderAndBufferFields = GlShaderAndBufferFields()
  crosshairFields::GlShaderAndBufferFields = GlShaderAndBufferFields()
  textFields::GlShaderAndBufferFields = GlShaderAndBufferFields()
  spacingsValue::Union{Vector{Tuple{Float64,Float64,Float64}},Tuple{Float64,Float64,Float64}} = [(1.0, 1.0, 1.0)]
  originValue::Union{Vector{Tuple{Float64,Float64,Float64}},Tuple{Float64,Float64,Float64}} = [(1.0, 1.0, 1.0)]
  supervoxelFields::GlShaderAndBufferFields = GlShaderAndBufferFields()
  supervoxelVertAndInd::Dict{String,Vector} = Dict("supervoxel_vertices" => [], "supervoxel_indices" => [])
  
  moveLesionMode::Bool = false
  movingLesionID::Int = 0
  movingLesionOriginalCoords::Vector{CartesianIndex{3}} = CartesianIndex{3}[]
  movingLesionStartTex::Tuple{Float64, Float64} = (0.0, 0.0)
  movingLesionLastDelta::CartesianIndex{3} = CartesianIndex(0,0,0)
  movingLesionOriginalBGs::Vector{UInt16} = UInt16[]
  movingLesionSourceName::String = ""   # which array name ("Mask", "manualModif") the lesion was found in
  allSupervoxels::Dict{Int, Dict{Int, Dict{String,Any}}} = Dict{Int, Dict{Int, Dict{String, Any}}}()
end

"""
Structure for MainMedEye3d, initialized with keyword arguments in coordinateDisplay (initialization function)
"""
@with_kw mutable struct MainMedEye3d
  channel::Base.Channel{Any}
  voxelArrayShapes::Vector{Tuple{Int64,Int64,Int64}} = Vector{Tuple}()
  voxelArrayTypes::Vector{Any} = Vector{Any}()
  textDispObj::ForWordsDispStruct = ForWordsDispStruct()# set of objects and constants needed for text diplay
  states::Vector{StateDataFields} = Vector{StateDataFields}()
  displayMode::DisplayMode = SingleImage
end


"""
Struct for holding information necessary for attaining the texture along with its voxel data
"""
@with_kw mutable struct DisplayedVoxels
  activeNumb::Union{Vector{Int32},Int32} = Int32(1)
  voxelData::Vector{Array{Float32,3}} = Vector{Array{Float32,3}}()
end


"""
Struct for holding the user generated voxel data
"""
@with_kw mutable struct CustomDisplayedVoxels
  voxelData::Vector{Array{Float32,3}} = Vector{Array{Float32,3}}()
  # scrollDat::FullScrollableDat = FullScrollableDat()
end
end #module
