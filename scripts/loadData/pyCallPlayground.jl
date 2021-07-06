using Base: String

using DrWatson
@quickactivate "Probabilistic medical segmentation"
using Conda
using PyCall

Conda.pip_interop(true)
Conda.pip("install", "SimpleITK")
Conda.pip("install", "h5py")

sitk = pyimport("SimpleITK")
np= pyimport("numpy")
h5py= pyimport("h5py")




mainHdfFolder = DrWatson.datadir("hdf5Main")
mainHdfFile = DrWatson.datadir("hdf5Main", "mainHdDataBaseLiver07.hdf5")



f = h5py.File(mainHdfFile, "w")
trainingScans = f.create_group("trainingScans")
trainingLabels = f.create_group("trainingLabels")
testScans = f.create_group("testScans")



#given directory it gives all mhd file names concateneted with path - to get full file path and second in subarray will be file name
function getListOfMhdFromFolder(folderPath::String) ::Vector{Vector{AbstractString}}
    return readdir(folderPath) |>
    (arr)-> filter((str)-> occursin(".mhd",str), arr) |>
    (arr)-> map(str-> [split(str,".")[1], joinpath(folderPath,str) ],arr)
end

#Return intensity of all voxels and physical location of pixels of each x,y and z axis so the result will be 2 dimensional array
function getPhysicalLocsandIntesities( image)
    pixels = np.array(sitk.GetArrayViewFromImage(image))
    
    tuples=CartesianIndices(pixels)     |>
    (cartInds)->Tuple.(cartInds) |> # changing into tuples to make it work with sitk
    (x)-> map((t)->(t[1]-1, t[2]-1, t[3]-1  ) ,x) # python is 0 based ...
    
    locs =  [tuples[:,1,1], tuples[1,:,1],tuples[1,1,:]  ]  |>#defining location of pixels that we are intrested in location
    (axes) -> map(axis-> map(tupl-> image.TransformIndexToPhysicalPoint(tupl), axis )  ,axes) # looking for physical locations of 
    return (pixels,locs)
end



function addGroups(group,folderPath)
    for shortArr in getListOfMhdFromFolder(folderPath)
        
        innerGroup= group.create_group(shortArr[0])
        dat = sitk.ReadImage(shortArr[1])|>
            (img)-> getPhysicalLocsandIntesities(img)
        innerGroup.create_dataset(shortArr[0], data=dat[1])
        innerGroup.create_dataset(shortArr[0]+"PhysLocs", data=dat[2])
        
        print("*")
        print(shortArr[0])
    end
end

pathToTrainingScans
pathToTrainingLabels
pathToTestScans

addGroups(trainingScans,pathToTrainingScans)
addGroups(trainingLabels,pathToTrainingLabels)
addGroups(testScans,pathToTestScans)

f.close()










image = sitk.ReadImage(dirOfExample)

pixels = np.array(sitk.GetArrayViewFromImage(image))
    
tuples=CartesianIndices(pixels)     |>
(cartInds)->Tuple.(cartInds) |> # changing into tuples to make it work with sitk
(x)-> map((t)->(t[1]-1, t[2]-1, t[3]-1  ) ,x) # python is 0 based ...
locs =  [tuples[:,1,1], tuples[1,:,1],tuples[1,1,:]  ]  

axes =  [tuples[:,1,1], tuples[1,:,1],tuples[1,1,:]  ] #defining location of pixels that we are intrested in location
map(axis-> map(tupl-> image.TransformIndexToPhysicalPoint(tupl), axis )  ,axes)

image.TransformIndexToPhysicalPoint(axes[1][5])










getPhysicalLocsandIntesities(image)

pixels = np.array(sitk.GetArrayViewFromImage(image))

image.TransformIndexToPhysicalPoint((0,0,0))
cartInds = CartesianIndices(pixels)[:,1,1]



dirOfExample = DrWatson.datadir("liverPrimData","training-scans" ,"scan","liver-orig001.mhd" )
np.array(sitk.GetArrayViewFromImage(sitk.ReadImage(dirOfExample)))[1,:,1]



pp = DrWatson.datadir("liverPrimData","training-scans" ,"scan")
getListOfMhdFromFolder(pp)

