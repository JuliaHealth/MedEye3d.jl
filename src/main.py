# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import SimpleITK as sitk
import matplotlib.pyplot as plt

import os
import sys
import h5py
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
pathToTrainingScans = 'C:\\Users\\jakub\\Downloads\\training-scans\\scan'
pathToTrainingLabels= 'C:\\Users\\jakub\\Downloads\\training-labels\\label'
pathToTestScans = 'C:\\Users\\jakub\\Downloads\\training-scans\\scan'

#takes mhd file on its basis loads from raw pixels data
def loadArrayFromFile(path):
    mr_image = sitk.ReadImage(path)
    return sitk.GetArrayViewFromImage(mr_image)
#given directory it gives all mhd file names concateneted with path - to get full file path and second in subarray will be file name
def getListOfMhdFromFolder(folderPath):
    arr = os.listdir(folderPath)
    arr= list(filter(lambda a: '.mhd' in a, arr))
    arr = list(map(lambda str: [str.split(".")[0], folderPath+"\\"+str ],arr))
    return arr


f = h5py.File('C:\\Users\\jakub\\OneDrive\\Documents\\GitHub\\probabilisticSegmentation\\Probabilistic medical segmentation\data\\hdf5Main\\mainHdDataBaseLiver07.hdf5', 'w')
trainingScans = f.create_group("trainingScans")
trainingLabels = f.create_group("trainingLabels")
testScans = f.create_group("testScans")


enumerated=[]
out=[]

#Return physical location of each pixel - will be used to calculate real distances in an image
def getPhysicalLocs (data, image) :
    enumerated = list(np.ndenumerate(np.array(data)))
    sh = data.shape
    out = np.full((sh[0], sh[1], sh[2], 3), np.nan, dtype="uint16")

    for tupl in enumerated:
        tuplLoc = tupl[0]
        phys = image.TransformIndexToPhysicalPoint(tuplLoc)
        out[tuplLoc[0], tuplLoc[1], tuplLoc[2]] = np.array([phys[0], phys[1], phys[2]]).astype("uint16")
    return out



def addGroups (group,folderPath):
    for shortArr in getListOfMhdFromFolder(folderPath):
        innerGroup= group.create_group(shortArr[0])
        img = sitk.ReadImage(shortArr[1])
        data = sitk.GetArrayFromImage(img)
        innerGroup.create_dataset(shortArr[0], data=data)
        innerGroup.create_dataset(shortArr[0]+"PhysLocs", data=getPhysicalLocs(data, img))
        print("*")
        print(shortArr[0])



addGroups(trainingScans,pathToTrainingScans)
addGroups(trainingLabels,pathToTrainingLabels)
addGroups(testScans,pathToTestScans)

f.close()
