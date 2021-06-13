
using DrWatson
@quickactivate "Probabilistic medical segmentation"

module ForDisplayStructs
export Mask
using ColorTypes
using Parameters
using Observables

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


end #module

