
using DrWatson
@quickactivate "Probabilistic medical segmentation"

module ForDisplayStructs
using Base: Int32, isvisible
export Mask,TextureSpec,forDisplayObjects, ActorWithOpenGlObjects, KeyboardStruct,TextureUniforms,MainImageUniforms, MaskTextureUniforms

using ColorTypes,Parameters,Observables,ModernGL,GLFW,Rocket, Dictionaries


```@doc
data needed for definition of mask  - data that will be displayed over main image 
this struct is parametarized by type of 3 dimensional array that will be used  to store data
```
@with_kw  struct Mask{arrayType}
  path::String #path to this file in Hdf5
  maskId::Int64 #unique associated with id taken from Hdf5 file system
  maskName::String #unique for class not unique for instance for example it can be name of the organ that will be segmented - need to be unique in instance but across instances needs to be named the same
  maskArrayObs::Observable{Array{arrayType}} # observable array used to store information that will be displayed over main image
  colorRGBA::RGBA #associated RGBA  that will be displayed based on the values in maskArrayObs
end
```@doc
hold reference numbers that will be used to access and modify given uniform value in a shader
```
abstract type TextureUniforms end


```@doc
hold reference numbers that will be used to access and modify given uniform value
In order to have easy fast access to the values set the most recent values will also be stored inside
In order to improve usability  we will also save with what data type this mask is associated 
for example Int, uint, float etc
```
@with_kw struct MaskTextureUniforms <: TextureUniforms
samplerName::String =""#name of the sampler - mainly for debugging purposes
samplerRef ::Int32 =Int32(0) #reference to sampler of the texture
colorsMaskRef ::Int32 =Int32(0) #reference to uniform holding color of this mask
isVisibleRef::Int32 =Int32(0)# reference to uniform that points weather we 
end

```@doc
Holding references to uniforms used to controll main image
```
@with_kw struct MainImageUniforms<: TextureUniforms
samplerName::String =""#name of the sampler - mainly for debugging purposes
samplerRef ::Int32 =Int32(0) #reference to   sampler of the texture
isVisibleRef::Int32 =Int32(0)# reference to uniform that points weather we 
# uniforms controlling windowing
min_shown_white::Int32 =Int32(0)
max_shown_black::Int32 =Int32(0)
displayRange::Int32 =Int32(0)
end

```@doc
Holding the data needed to create and  later reference the textures
```
@with_kw struct TextureSpec
  name::String                #human readable name by which we can reference texture
  numb::Int32 =-1             #needed to enable swithing between textures generally convinient when between 0-9; needed only if texture is to be modified by mouse input
  dataType::Type              #type of data that will be used Int/uint/float
  isMainImage ::Bool = false  #true if this texture represents main image
  isTextTexture ::Bool = false  #true if this texture represents texture that is supposed to hold text
  isSecondaryMain ::Bool = false # true if it holds some other important information that is not mask - used for example in case of nuclear imagiing studies
  isContinuusMask ::Bool = false # in case of masks if mask is continuus color display will be a bit diffrent
  color::RGB = RGB(0.0,0.0,0.0) #needed in case for the masks in order to establish the range of colors we are intrested in in case of binary mask there is no point to supply more than one color (supply Vector with length = 1)
  colorSet::Vector{RGB}=[]    #set of colors that can be used for mask with continous values
  strokeWidth::Int32 =Int32(3)#marking how thick should be the line that is left after acting with the mouse ... 
  isEditable::Bool =false     #if true we can modify given  texture using mouse interaction
  widthh::Int32 =Int32(0)     #width of texture
  heightt::Int32 =Int32(0)    #height of the texture
  slicesNumber::Int = 0       #number of slices available
  GL_Rtype::UInt32 =UInt32(0)           #GlRtype - for example GL_R8UI or GL_R16I
  OpGlType ::UInt32 =UInt32(0)          #open gl type - for example GL_UNSIGNED_BYTE or GL_SHORT
  ID::Base.RefValue{UInt32} = Ref(UInt32(0))   #id of Texture
  isVisible::Bool= true       #if false it should be invisible 
  uniforms::TextureUniforms=MaskTextureUniforms()# holds values needed to control uniforms in a shader
  #used in case this is main image and we want to set window
  min_shown_white::Int32 =400
  max_shown_black::Int32 =-200
end

```@doc
Defined in order to hold constant objects needed to display images 
```
@with_kw mutable struct forDisplayObjects    
  listOfTextSpecifications::Vector{TextureSpec} = []
  window = []
  vertex_shader::UInt32 =1
  fragment_shader::UInt32=1
  shader_program::UInt32=1
  stopListening::Base.Threads.Atomic{Bool}= Threads.Atomic{Bool}(0)# enables unlocking GLFW context for futher actions
  stopExecution::Base.Threads.Atomic{Bool}= Threads.Atomic{Bool}(0)#it will halt ability to display image for futher display in order to keep OpenGL from blocking - optional to set
  vbo::UInt32 =1 #vertex buffer object id
  ebo::UInt32 =1 #element buffer object id
  #imageDims = texture dimensions
  imageTextureWidth::Int32=1
  imageTextureHeight::Int32=1
  #windowDims
  windowWidth::Int32=1
  windowHeight::Int32=1
  #number of available slices - needed for scrolling needs
  slicesNumber::Int32=1
  mainImageUniforms::MainImageUniforms = MainImageUniforms()# struct with references to main image
end



```@doc
Actor that is able to store a state to keep needed data for proper display
```
mutable struct ActorWithOpenGlObjects <: NextActor{Any}
    currentDisplayedSlice::Int # stores information what slice number we are currently displaying
    mainForDisplayObjects::Main.ForDisplayStructs.forDisplayObjects # stores objects needed to  display using OpenGL and GLFW
    onScrollData::Vector{Tuple{String, Array{T, 3} where T}}
    textureToModifyVec::Vector{TextureSpec} # texture that we want currently to modify - if list is empty it means that we do not intend to modify any texture
    isSliceChanged::Bool # set to true when slice is changed set to false when we start interacting with this slice - thanks to this we know that when we start drawing on one slice and change the slice the line would star a new on new slice
    ActorWithOpenGlObjects() = new(1,forDisplayObjects(),[],[],false)
end

```@doc
Holding necessery data to controll keyboard shortcuts```
@with_kw struct KeyboardStruct
  isCtrlPressed::Bool # left - scancode 37 right 105 - Int32
  isShiftPressed::Bool  # left - scancode 50 right 62- Int32
  isAltPressed::Bool# left - scancode 64 right 108- Int32
  lastKeyPressed::String # last pressed key 
end

end #module

