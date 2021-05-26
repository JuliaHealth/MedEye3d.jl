using DrWatson
@quickactivate "Probabilistic medical segmentation"
DrWatson.greet()

using manageH5File


print(manageH5File.getExample())

# using HDF5

# pathToHd5 = datadir("hdf5Main", "mainHdDataBaseLiver07.hdf5")
# g = h5open(pathToHd5, "r")


# print(g["testScans"][1])
# arr=[]
# for obj in g["testScans"]
#   arr= read( obj)
#   break
# end
# arr


