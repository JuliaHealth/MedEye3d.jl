
using DrWatson
@quickactivate "Probabilistic medical segmentation"

module ForDisplayStructs
using Base: Int32
export Mask
export TextureSpec
export forDisplayObjects
export ActorWithOpenGlObjects

using ColorTypes
using Parameters
using Observables
using ModernGL
using GLFW
using Rocket

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
Holding the data needed to create and  later reference the textures
```
@with_kw struct TextureSpec
  name::String # human readable name by which we can reference texture
  numb::Int32 =-1# needed to enable swithing between textures generally convinient when between 0-9; needed only if texture is to be modified by mouse input
  colors::Vector{RGB}=[]# needed in case for the masks in order to establish the range of colors we are intrested in in case of binary mask there is no point to supply more than one color (supply Vector with length = 1)
  strokeWidth::Int32 =Int32(3)#marking how thick should be the line that is left after acting with the mouse ... 
  isEditable::Bool =false #if true we can modify given  texture using mouse interaction
  widthh::Int32 =Int32(0)  # width of texture
  heightt::Int32 =Int32(0)  #height of the texture
  slicesNumber::Int = 0 #number of slices available
  GL_Rtype::UInt32 #GlRtype - for example GL_R8UI or GL_R16I
  OpGlType ::UInt32 #open gl type - for example GL_UNSIGNED_BYTE or GL_SHORT
  samplName::String #name of the specified sampler in fragment shader  - critical is that in case of using floats we use sampler in case of integers isampler and in case of unsigned integers usampler 
  ID::Base.RefValue{UInt32} = Ref(UInt32(0))   #id of Texture
  isVisible::Bool= true # if false it should be invisible 
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
  stopListening::Base.Threads.Atomic{Bool}= Threads.Atomic{Bool}(0)
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
end




```@doc
Actor that is able to store a state to keep needed data for proper display
```
mutable struct ActorWithOpenGlObjects <: NextActor{Any}
    currentDisplayedSlice::Int # stores information what slice number we are currently displaying
    mainForDisplayObjects::Main.ForDisplayStructs.forDisplayObjects # stores objects needed to  display using OpenGL and GLFW
    onScrollData::Vector{Tuple{String, Array{T, 3} where T}}
    textureToModifyVec::Vector{TextureSpec} # texture that we want currently to modify - if list is empty it means that we do not intend to modify any texture
    ActorWithOpenGlObjects() = new(1,forDisplayObjects(),[],[])
end




end #module

