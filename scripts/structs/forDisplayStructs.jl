
using DrWatson
@quickactivate "Probabilistic medical segmentation"

module ForDisplayStructs
export Mask
export TextureSpec
export forDisplayObjects

using ColorTypes
using Parameters
using Observables
using ModernGL
using GLFW
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
struct TextureSpec
  name::String # human readable name by which we can reference texture
  widthh::Int # width of texture
  heightt::Int #height of the texture
  slicesNumber::Int #number of slices available
  GL_Rtype::UInt32 #GlRtype - for example GL_R8UI or GL_R16I
  OpGlType ::UInt32 #open gl type - for example GL_UNSIGNED_BYTE or GL_SHORT
  samplName::String #name of the specified sampler in fragment shader 
  ID   #id of Texture

end

```@doc
Defined in order to hold constant objects needed to display images 
```
@with_kw struct forDisplayObjects    
  listOfTextSpecifications::Vector{TextureSpec} = []
  window = []
  vertex_shader::UInt32 =1
  fragment_shader::UInt32=1
  shader_program::UInt32=1
  stopListening::Base.Threads.Atomic{Bool}= Threads.Atomic{Bool}(0)
end



end #module

