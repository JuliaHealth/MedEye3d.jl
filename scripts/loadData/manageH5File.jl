```@doc
module for controlling data stored in Hdf5 filesystem

```
module h5manag


using DrWatson
@quickactivate "Probabilistic medical segmentation"
using HDF5
using Main.ForDisplayStructs
using ColorTypes
using Parameters
using Observables


const pathToHd5 = DrWatson.datadir("hdf5Main", "mainHdDataBaseLiver07.hdf5")
const g = h5open(pathToHd5, "r+")

```@doc
getting example study in a form of 3 dimensional array
```
function getExample() ::Array{Number, 3}
     read(g["trainingScans/liver-orig001"]["liver-orig001"])
end

```@doc
getting example mask in a form of 3 dimensional array
```
function getExampleLabels() ::Array{Number, 3}
    read(g["trainingLabels"]["liver-seg001"]["liver-seg001"])
end


```@doc
getting subgroups of given group
```
myGetGroupsNames(file::HDF5.File) = keys(file)


```@doc
creating or retrieving Mask that will store some data about the instance- data will be stored in a subfolder related to the patient we are analyzing in array of the same dimensions as main array in this subfolder
name - name of mask - it should be unique in patient but not among patients - for example name may be liver in single patient the is one liver but in all patients mask with this name can be present
path - path in main hdf5 filesytem to get to a target subfolder example : "trainingScans/liver-orig001"
dims - dimensions of main array
color - RGBA value that will be used in displayed mask
parameter - type of observable 3 dimensional array used
return Mask struct with observable 3d array linked to appropriate HDF5 dataset
```
function getOrCreateMaskData(::Type{arrayType}, name::String,path::String,dims::Tuple{Int, Int, Int}, color::RGBA  ) :: Main.ForDisplayStructs.Mask{arrayType}     where{arrayType}
#initializing mask Array Observable
maskArrOut = Observable(Array{arrayType}(UndefInitializer(), dims))
maskId= 0    
#1) we need  to establish weather we have a mask of this name in given path
if name in keys(g[path]) #where g is our dataset
#2a) if we have this mas already created we load the data  from it into array
    obj = g[path][name]
    maskArrOut[]= read(obj)
    maskId= obj.id
#2b) if such file is not present we put array of given dimensions and save it into given path dataset of given name
else
    g[path][name] = maskArrOut[]
    maskId= g[path][name].id
end    
#5) the struct Mask with appropriate name will be return with given name created/retrieved array, id of HDF5 file anhd given color
return Main.ForDisplayStructs.Mask{arrayType}(
            path, #path
            maskId, #maskId
            name, #maskName
            Array{arrayType}(UndefInitializer(), dims), #maskArrayObs
            color  #colorRGBA 
            )

  
end



```@doc
saving to hdf5 data from mask object
```
function saveMaskDataC!(::Type{arrayType},mask :: Main.ForDisplayStructs.Mask{arrayType}) where{arrayType}
    g[mask.path][mask.maskName][:,:,:] = mask.maskArrayObs[]
 end


end # manag


