```@doc
module for controlling data stored in Hdf5 filesystem

```
module manageH5File 

using DrWatson
@quickactivate "Probabilistic medical segmentation"
using HDF5


const pathToHd5 = DrWatson.datadir("hdf5Main", "mainHdDataBaseLiver07.hdf5")
const g = h5open(pathToHd5, "r")

```@doc
getting example study in a form of 3 dimensional array
```
function getExample() ::Array{Number, 3}
    arr=[]
        for obj in g["testScans"]
            arr= read( obj)
            break
        end
    return arr
end

end