
using DrWatson
@quickactivate "Probabilistic medical segmentation"

```@doc
data needed for definition of mask  - data that will be displayed over main image 
this struct is parametarized by type of 3 dimensional array that will be used  to store data
```
struct Mask{arrayType}
  maskId::Int64 #unique associated with id taken from Hdf5 file system
  maskName::String #unique for class not unique for instance for example it can be name of the organ that will be segmented - need to be unique in instance but across instances needs to be named the same
  maskArrayObs::arrayType # observable array used to store information that will be displayed over main image
  colorRGBA #associated RGBA  that will be displayed based on the values in maskArrayObs
end
