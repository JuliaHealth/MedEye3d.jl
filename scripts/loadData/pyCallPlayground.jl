using Base: String

using DrWatson
@quickactivate "JuliaMed3d"
using Conda
using PyCall
using Pkg
Pkg.build("HDF5")
using HDF5
Conda.pip_interop(true)
Conda.pip("install", "SimpleITK")
Conda.pip("install", "h5py")

sitk = pyimport("SimpleITK")
np= pyimport("numpy")



mainHdfFolder = DrWatson.datadir("hdf5Main")
mainHdfFile = DrWatson.datadir("hdf5Main", "mainHdDataBaseLiver07.hdf5")


f = h5open(mainHdfFile, "w")

trainingScans = create_group(f, "trainingScans")
trainingLabels =  create_group(f, "trainingLabels")   
testScans = create_group(f, "testScans")        



#given directory it gives all mhd file names concateneted with path - to get full file path and second in subarray will be file name
function getListOfMhdFromFolder(folderPath::String) ::Vector{Vector{AbstractString}}
    return readdir(folderPath) |>
    (arr)-> filter((str)-> occursin(".mhd",str), arr) |>
    (arr)-> map(str-> [split(str,".")[1], joinpath(folderPath,str) ],arr)
end

#Return intensity of all voxels and physical location of pixels of each x,y and z axis so the result will be 2 dimensional array
function getPhysicalLocsandIntesities( image, type::Type)
    pixels = np.array(sitk.GetArrayViewFromImage(image))
    pixels = convert(Array{type, 3},pixels)
    pixels=  permutedims(pixels, (3,2,1))
    sizz=size(pixels)
    for i in 1:sizz[1]
        pixels[i,:,:] =  reverse(pixels[i,:,:])
     end# 

   for i in 1:sizz[2]
    pixels[:,i,:] =  reverse(pixels[:,i,:])
   end# 


    tuples=CartesianIndices(pixels)     |>
    (cartInds)->Tuple.(cartInds) |> # changing into tuples to make it work with sitk
    (x)-> map((t)->(t[1]-1, t[2]-1, t[3]-1  ) ,x) # python is 0 based ...
 
    axes =  [tuples[:,1,1], tuples[1,:,1],tuples[1,1,:]  ] #defining location of pixels that we are intrested in location
    locs= map(axis-> map(tupl-> maximum([image.TransformIndexToPhysicalPoint(tupl)...]), axis )  ,axes) # looking for physical locations of voxels but we are intrested only in non 0 locations ...
    return (pixels,locs)
end



function addGroups(group,folderPath, type::Type)
    for shortArr in getListOfMhdFromFolder(folderPath)
        img = sitk.ReadImage(shortArr[2])
        dat = getPhysicalLocsandIntesities(img,type)       
        spacings = img.GetSpacing()
        spaCingList = [spacings[1],spacings[2],spacings[3]]
        innerGroup= create_group(group, shortArr[1])
        write(innerGroup, shortArr[1],dat[1]) 
        write(innerGroup, shortArr[1]*"PhysLocX" ,dat[2][1]) 
        write(innerGroup, shortArr[1]*"PhysLocY" ,dat[2][2]) 
        write(innerGroup, shortArr[1]*"PhysLocZ" ,dat[2][3]) 
        write(innerGroup, shortArr[1]*"Spacing" ,spaCingList) 
        
        print("*")
        print(shortArr[1])
    end
end

pathToTrainingScans =DrWatson.datadir("liverPrimData","training-scans" ,"scan" )
pathToTrainingLabels =DrWatson.datadir("liverPrimData","training-labels" ,"label" )
pathToTestScans =DrWatson.datadir("liverPrimData","test-scans" ,"scan" )


addGroups(trainingScans,pathToTrainingScans, Int16)
addGroups(trainingLabels,pathToTrainingLabels, UInt8)
addGroups(testScans,pathToTestScans, Int16)

close(f)



########PET CT upload












dirOfExample = DrWatson.datadir("liverPrimData","training-scans" ,"scan","liver-orig001.mhd" )
pixels = np.array(np.uint16,sitk.GetArrayViewFromImage(image))
pathToTrainingScans =DrWatson.datadir("liverPrimData","training-scans" ,"scan" )














shortArr= getListOfMhdFromFolder(pathToTrainingScans)[8]
group = trainingScans        


innerGroup= create_group(group, shortArr[1])
img = sitk.ReadImage(shortArr[2])
dat = getPhysicalLocsandIntesities(img,Int16)


datt =sitk.GetArrayViewFromImage(img)
jDat = Int16.(np.array(datt))

sizeof(np.array(datt))
sizeof(jDat)

d = create_dataset(innerGroup, "aaa", Int)

b = create_dataset(innerGroup, "bdgy", jDat,dtype= Int16)


typeof(jDat)
convert( Matrix{Int32} ,datt )

reduce(hcat,dat)

size(dat)

write(innerGroup, shortArr[1],dat[1]) 
write(innerGroup, shortArr[1]*"PhysLocX" ,dat[2][1]) 
write(innerGroup, shortArr[1]*"PhysLocY" ,dat[2][2]) 
write(innerGroup, shortArr[1]*"PhysLocZ" ,dat[2][3]) 


hcat(dat[2][1],dat[2][2],dat[2][3] )

print("*")
print(shortArr[1])













getPhysicalLocsandIntesities(image)

pixels = np.array(sitk.GetArrayViewFromImage(image))

image.TransformIndexToPhysicalPoint((0,0,0))
cartInds = CartesianIndices(pixels)[:,1,1]



dirOfExample = DrWatson.datadir("liverPrimData","training-scans" ,"scan","liver-orig001.mhd" )
np.array(sitk.GetArrayViewFromImage(sitk.ReadImage(dirOfExample)))[1,:,1]



pp = DrWatson.datadir("liverPrimData","training-scans" ,"scan")
getListOfMhdFromFolder(pp)

