```@doc
module for controlling data stored in Hdf5 filesystem

```

using DrWatson
@quickactivate "Probabilistic medical segmentation"
using HDF5
include(DrWatson.scriptsdir("structs","forDisplayStructs.jl"))

const pathToHd5 = DrWatson.datadir("hdf5Main", "mainHdDataBaseLiver07.hdf5")
const g = h5open(pathToHd5, "w")

```@doc
getting example study in a form of 3 dimensional array
```
function getExample() ::Array{Number, 3}
     read(g["trainingScans"]["liver-orig001"]["liver-orig001"])
end

```@doc
getting example mask in a form of 3 dimensional array
```
function getExampleLabels() ::Array{Number, 3}
    read(g["trainingLabels"]["liver-seg001"]["liver-seg001"])
end


"trainingScans"
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
function getOrCreateMaskData(name::String,path::String,dims::Tuple{Int, Int, Int}, color::RGBA ) :: Mask{arrayType} where{arrayType}
#1) we need  to establish weather we have a mask of this name in given path
if name in myGetGroupsNames(g[path]) #where g is our dataset
#2a) if we have this mas already created we load the data  from it into array
    
#2b) if such file is not present we create array of given dimensions and save it into given path dataset of given name

#3) we put the array into observable

#4) we register that on the change of the array the data will be asynchronously persisted in the hdf5 dataset  of given path and given name

#5) the struct Mask with appropriate name will be return with given name created/retrieved array, id of HDF5 file anhd given color

end    



function getOrCreateMaskData(dims::Tuple{Int, Int, Int}) :: Observable{arrayType} where{arrayType}


# Attributes can be created using

# attributes(parent)[name] = value