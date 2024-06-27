module ForDisplayStructs
using Base: Int32, isvisible
export MouseStruct, parameter_type, Mask, TextureSpec, forDisplayObjects, StateDataFields, KeyboardStruct, TextureUniforms, MainImageUniforms, MaskTextureUniforms, ForWordsDispStruct, MainMedEye3d
export ScrollChannelDataStruct
using ColorTypes, Parameters, Observables, ModernGL, GLFW, Dictionaries, FreeTypeAbstraction, ..DataStructs


"""
data needed for definition of mask  - data that will be displayed over main image
this struct is parametarized by type of 3 dimensional array that will be used  to store data
"""
@with_kw struct Mask{arrayType}
  path::String #path to this file in Hdf5
  maskId::Int64 #unique associated with id taken from Hdf5 file system
  maskName::String #unique for class not unique for instance for example it can be name of the organ that will be segmented - need to be unique in instance but across instances needs to be named the same
  maskArrayObs::Observable{Array{arrayType}} # observable array used to store information that will be displayed over main image
  colorRGBA::RGBA #associated RGBA  that will be displayed based on the values in maskArrayObs
end
"""
hold reference numbers that will be used to access and modify given uniform value in a shader
"""
abstract type TextureUniforms end


"""
hold reference numbers that will be used to access and modify given uniform value
In order to have easy fast access to the values set the most recent values will also be stored inside
In order to improve usability  we will also save with what data type this mask is associated
for example Int, uint, float etc
"""
@with_kw struct MaskTextureUniforms <: TextureUniforms
  samplerName::String = ""#name of the sampler - mainly for debugging purposes
  samplerRef::Int32 = Int32(0) #reference to sampler of the texture
  colorsMaskRef::Int32 = Int32(0) #reference to uniform holding color of this mask
  isVisibleRef::Int32 = Int32(0)# reference to uniform that points weather we
  maskMinValue::Int32 = Int32(0)# minimum value associated with possible value of mask
  maskMAxValue::Int32 = Int32(0)# maximum value associated with possible value of mask
  maskRangeValue::Int32 = Int32(0)# range of values associated with possible value of mask
  maskContribution::Int32 = Int32(0)# controlls contribution  of given mask to the overall image - maximum value is 1 minimum 0 if we have 3 masks and all control contribution is set to 1 and all are visible their corresponding influence to pixel color is 33%
end

"""
Holding references to ..Uniforms used to controll main image
"""
@with_kw struct MainImageUniforms <: TextureUniforms
  samplerName::String = ""#name of the sampler - mainly for debugging purposes
  samplerRef::Int32 = Int32(0) #reference to   sampler of the texture
  isVisibleRef::Int32 = Int32(0)# reference to uniform that points weather we
  # ..Uniforms controlling windowing
  min_shown_white::Int32 = Int32(0)
  max_shown_black::Int32 = Int32(0)
  displayRange::Int32 = Int32(0)
  # ..Uniforms controlling  displaying masks diffrence
  isMaskDiffrenceVis::Int32 = Int32(0)
  mainImageContribution::Float32 = 1.0 # controlls contribution  of given mask to the overall image - maximum value is 1 minimum 0 if we have 3 masks and all control contribution is set to 1 and all are visible their corresponding influence to pixel color is 33%

end

"""
Holding the data needed to create and  later reference the textures

name::String=""               #human readable name by which we can reference texture
numb::Int32 =-1               #needed to enable swithing between textures generally convinient when between 0-9; needed only if texture is to be modified by mouse input
whichCreated::Int32 =-1       #marks which one this texture was when created - so first in list second ... - needed for convinient accessing ..Uniforms in shaders
isMainImage ::Bool = false  #true if this texture represents main image
isNuclearMask ::Bool = false # used for example in case of nuclear imagiing studies
isContinuusMask ::Bool = false # in case of masks if mask is continuus color display we set multiple colors in a vector
color::RGB = RGB(0.0,0.0,0.0) #needed in case for the masks in order to establish the range of colors we are intrested in in case of binary mask there is no point to supply more than one color (supply Vector with length = 1)
colorSet::Vector{RGB}=[]    #set of colors that can be used for mask with continous values
strokeWidth::Int32 =Int32(3)#marking how thick should be the line that is left after acting with the mouse ...
isEditable::Bool =false     #if true we can modify given  texture using mouse interaction
GL_Rtype::UInt32 =UInt32(0)           #GlRtype - for example GL_R8UI or GL_R16I
OpGlType ::UInt32 =UInt32(0)          #open gl type - for example GL_UNSIGNED_BYTE or GL_SHORT
actTextrureNumb ::UInt32 =UInt32(0)          #usefull to be able to activate the texture using GL_Activetexture - with proper open GL constant
associatedActiveNumer ::Int64 =Int64(0)          #usefull to be able to activate the texture using GL_Activetexture - with proper open GL constant
ID::Base.RefValue{UInt32} = Ref(UInt32(0))   #id of Texture
isVisible::Bool= true       #if false it should be invisible
uniforms::TextureUniforms=MaskTextureUniforms()# holds values needed to control ..Uniforms in a shader
minAndMaxValue::Vector{T} = []#entry one is minimum possible value for this mask, and second entry is maximum possible value for this mask

"""
@with_kw mutable struct TextureSpec{T}
  name::String = ""               #human readable name by which we can reference texture
  numb::Int32 = -1               #needed to enable swithing between textures generally convinient when between 0-9; needed only if texture is to be modified by mouse input
  whichCreated::Int32 = -1       #marks which one this texture was when created - so first in list second ... - needed for convinient accessing ..Uniforms in shaders
  isMainImage::Bool = false  #true if this texture represents main image
  isNuclearMask::Bool = false # used for example in case of nuclear imagiing studies
  isContinuusMask::Bool = false # in case of masks if mask is continuus color display we set multiple colors in a vector
  color::RGB = RGB(0.0, 0.0, 0.0) #needed in case for the masks in order to establish the range of colors we are intrested in in case of binary mask there is no point to supply more than one color (supply Vector with length = 1)
  colorSet::Vector{RGB} = []    #set of colors that can be used for mask with continous values
  strokeWidth::Int32 = Int32(3)#marking how thick should be the line that is left after acting with the mouse ...
  isEditable::Bool = false     #if true we can modify given  texture using mouse interaction
  GL_Rtype::UInt32 = UInt32(0)           #GlRtype - for example GL_R8UI or GL_R16I
  OpGlType::UInt32 = UInt32(0)          #open gl type - for example GL_UNSIGNED_BYTE or GL_SHORT
  actTextrureNumb::UInt32 = UInt32(0)          #usefull to be able to activate the texture using GL_Activetexture - with proper open GL constant
  associatedActiveNumer::Int64 = Int64(0)          #usefull to be able to activate the texture using GL_Activetexture - with proper open GL constant
  ID::Base.RefValue{UInt32} = Ref(UInt32(0))   #id of Texture
  isVisible::Bool = true       #if false it should be invisible
  uniforms::TextureUniforms = MaskTextureUniforms()# holds values needed to control ..Uniforms in a shader
  minAndMaxValue::Vector{T} = []#entry one is minimum possible value for this mask, and second entry is maximum possible value for this mask
  maskContribution::Float32 = 1.0 # controlls contribution  of given mask to the overall image - maximum value is 1 minimum 0 if we have 3 masks and all control contribution is set to 1 and all are visible their corresponding influence to pixel color is 33%
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
  mainImageUniforms::MainImageUniforms = MainImageUniforms()# struct with references to main image
  TextureIndexes::Dictionary{String,Int64} = Dictionary{String,Int64}()  #gives a way of efficient querying by supplying dictionary where key is a name we are intrested in and a key is index where it is located in our array
  numIndexes::Dictionary{Int32,Int64} = Dictionary{Int32,Int64}() # a way for fast query using assigned numbers
  gslsStr::String = "" # string giving information about used openg gl gsls version
  windowControlStruct::WindowControlStruct = WindowControlStruct()# holding data usefull to controll display window
  isFastScroll::Bool = false # set by f letter to true and by s to normal - slow
end


"""
Holding necessery data to display text  - like font related
"""
@with_kw struct ForWordsDispStruct
  fontFace::FTFont = FTFont(Ptr{FreeTypeAbstraction.FreeType.__JL_FT_FaceRec_}(), false) # font we will use to display text
  textureSpec::TextureSpec = TextureSpec{UInt8}() # texture specification of texture used to display text
  fragment_shader_words::UInt32 = 1 #reference to fragment shader used to display text
  vbo_words::Base.RefValue{UInt32} = Ref(UInt32(1)) #reference to vertex buffer object used to display text
  shader_program_words::UInt32 = 1

end #ForWordsDispStruct


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
@with_kw struct KeyboardStruct
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
  isPlusPressed::Bool = false
  isMinusPressed::Bool = false
  isZPressed::Bool = false
  isFPressed::Bool = false
  isSPressed::Bool = false
  lastKeysPressed::Vector{String} = [] # last pressed keys - it listenes to keys only if ctrl/shift or alt is pressed- it clears when we release those case or when we press enter
  #informations about what triggered sending this particular struct to the  actor
  mostRecentScanCode::GLFW.Key = GLFW.KEY_KP_4
  mostRecentKeyName::String = ""
  mostRecentAction::GLFW.Action = GLFW.RELEASE

end
"""
Holding necessery data to controll mouse interaction
"""
@with_kw struct MouseStruct
  isLeftButtonDown::Bool = false # true if left button was pressed and not yet released
  isRightButtonDown::Bool = false# true if right button was pressed and not yet released
  lastCoordinates::Vector{CartesianIndex{2}} = [] # list of accumulated mouse coordinates
end#MouseStruct


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
  forUndoVector::AbstractArray = [] # holds lambda functions that when invoked will  undo last operations
  maxLengthOfForUndoVector::Int64 = 15 # number controls how many step at maximum we can get back
end

"""
Structure for MainMedEye3d, initialized with keyword arguments in coordinateDisplay (initialization function)
"""
@with_kw mutable struct MainMedEye3d
channel :: Base.Channel{Any}
end


end #module

