######### Defining Helper Functions and imports

#]add ColorTypes,DataTypesBasic,Dictionaries,FreeType,FreeTypeAbstraction,GLFW,GeometryTypes,
using Pkg
Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")

Pkg.add("CUDA")

Pkg.add("Rocket")

Pkg.add(Pkg.PackageSpec(;name="ColorTypes", version="0.11.4"))
Pkg.add(Pkg.PackageSpec(;name="DataTypesBasic", version="2.0.3"))
Pkg.add(Pkg.PackageSpec(;name="Dictionaries", version="0.3.25"))
Pkg.add(Pkg.PackageSpec(;name="FreeType", version="4.1.0"))
Pkg.add(Pkg.PackageSpec(;name="FreeTypeAbstraction", version="0.10.0"))
Pkg.add(Pkg.PackageSpec(;name="GLFW", version="3.4.1"))
Pkg.add(Pkg.PackageSpec(;name="GeometryTypes", version="0.8.5"))
Pkg.add(Pkg.PackageSpec(;name="Match", version="1.2.0"))
Pkg.add(Pkg.PackageSpec(;name="ModernGL", version="1.1.5"))
Pkg.add(Pkg.PackageSpec(;name="Observables", version="0.5.4"))
Pkg.add(Pkg.PackageSpec(;name="Parameters", version="0.12.3"))
Pkg.add(Pkg.PackageSpec(;name="Setfield", version="1.1.1"))
Pkg.add(Pkg.PackageSpec(;name="HDF5", version="0.17.1"))
Pkg.add(Pkg.PackageSpec(;name="StaticArrays", version="1.5.0"))
Pkg.add(Pkg.PackageSpec(;name="MedEval3D", version="0.2.0"))
Pkg.add("Conda")
Pkg.add("PythonCall")
Pkg.add("CondaPkg")
Pkg.add("Libz")
using Pkg

Pkg.add(["ProgressMeter","StaticArrays","BSON","Distributed","Flux","Hyperopt","Colors"
,"Plots","Distributions","Clustering","IrrationalConstants","ParallelStencil"
,"HDF5","CoordinateTransformations","NIfTI","Images"])


# CondaPkg.add_pip("SimpleITK", version="")

using Pkg
# Pkg.add(url="https://github.com/jakubMitura14/Rocket.jl.git")
# Pkg.add("Rocket")
# directories of PET/CT Data - from not published (yet) dataset - single example available from https://wwsi365-my.sharepoint.com/:f:/g/personal/s9956jm_ms_wwsi_edu_pl/Eq3cL7Md5bhPvnUlFLAMKZAB3nsbl6Q18fG96iVajvnNqA?e=bzX68X
dirOfExample ="/workspaces/MedEye3d.jl/docs/src/data/Example_ct.nii.gz"
dirOfExamplePET ="/workspaces/MedEye3d.jl/docs/src/data/petb.nii.gz"

# @spawn :interactive
#I use Simple ITK as most robust
using MedEye3d,CondaPkg,Pkg,PythonCall, Base.Threads

# Conda.pip_interop(true)
# Conda.pip("install", "SimpleITK")
# Conda.pip("install", "h5py")
# CondaPkg.add("SimpleITK")
# CondaPkg.add("numpy")
sitk = pyimport("SimpleITK")
np= pyimport("numpy")

import MedEye3d
import MedEye3d.ForDisplayStructs
import MedEye3d.ForDisplayStructs.TextureSpec
using ColorTypes
import MedEye3d.SegmentationDisplay

import MedEye3d.DataStructs.ThreeDimRawDat
import MedEye3d.DataStructs.DataToScrollDims
import MedEye3d.DataStructs.FullScrollableDat
import MedEye3d.ForDisplayStructs.KeyboardStruct
import MedEye3d.ForDisplayStructs.MouseStruct
import MedEye3d.ForDisplayStructs.ActorWithOpenGlObjects
import MedEye3d.OpenGLDisplayUtils
import MedEye3d.DisplayWords.textLinesFromStrings
import MedEye3d.StructsManag.getThreeDims
using Rocket
import  Rocket.AsyncSchedulerInstance
import  Rocket.AsyncSchedulerDataMessage


using MedPipe3D
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedEye3d.visualizationFromHdf5, MedEye3d.distinctColorsSaved
using CUDA
using HDF5,Colors,Rocket
using CoordinateTransformations,Images

using NIfTI,LinearAlgebra,DICOM


function Rocket.__process_channeled_message(instance::AsyncSchedulerInstance{D}, message::AsyncSchedulerDataMessage{D}, actor) where D
   # print("oooooooooooo")
   on_next!(actor, message.data)
end


Pkg.instantiate()

#add ProgressMeter StaticArrays BSON Distributed Flux Hyperopt Colors Plots Distributions Clustering IrrationalConstants ParallelStencil HDF5

#directory where we want to store our HDF5 that we will use
pathToHDF5="/root/data/smallDataSet.hdf5"
# data_dir = "/home/data/Task09_Spleen/"
data_dir = "/root/data_decathlon/Task09_Spleen"
fid = h5open(pathToHDF5, "w")



dicom_path="/workspaces/MedEye3d.jl/docs/src/ScalarVolume_0"

# Read the DICOM file
dicom_data = DICOM.dcmdir_parse(dicom_path)

# Extract the image data
image_data = dicom_data["PixelData"]

# Extract the spatial metadata
spacing = (dicom_data["PixelSpacing"][1], dicom_data["PixelSpacing"][2], dicom_data["SliceThickness"])
origin = (dicom_data["ImagePositionPatient"][1], dicom_data["ImagePositionPatient"][2], dicom_data["ImagePositionPatient"][3])
direction = (dicom_data["ImageOrientationPatient"][1:3], dicom_data["ImageOrientationPatient"][4:6])




# Channel{T=Any}(func::Function, size=0; taskref=nothing, spawn=false)
# Base.Channel(size=0; taskref=nothing, spawn=false)


# Base.Channel{Char}(1, spawn=true)





#representing number that is the patient id in this dataset
patentNum = 3
patienGroupName=string(patentNum)
z=7# how big is the area from which we collect data to construct probability distributions
klusterNumb = 5# number of clusters - number of probability distributions we will use
#directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below

train_labels = map(fileEntry-> joinpath(data_dir,"labelsTr",fileEntry),readdir(joinpath(data_dir,"labelsTr"); sort=true))
train_images = map(fileEntry-> joinpath(data_dir,"imagesTr",fileEntry),readdir(joinpath(data_dir,"imagesTr"); sort=true))


#zipping so we will have tuples with image and label names
zipped= collect(zip(train_images,train_labels))
zipped=map(tupl -> (replace(tupl[1], "._" => ""), replace(tupl[2], "._" => "")),zipped)
tupl=zipped[patentNum]





# size=0
# Channel{Int}(size, spawn=true)

path_curr=tupl[1]
# function load_by_julia(path)

# Load the image from path
img = niread(path_curr)

# Get the pixel data array
data = img.raw

# Get the spacing
spacing = img.header.pixdim[2:4]


img.header.orientation
# Get the direction
direction = img.orientation

# Get the origin
origin = img.header.qoffset_x, img.header.qoffset_y, img.header.qoffset_z

    # return data, spacing, direction, origin
# end    
load_by_julia(tupl[1])


nii = niread(path_curr)

nii.header.srow_x
nii.header.srow_y
nii.header.srow_z

NIfTI.orientation(nii.header)

direction = nii.header.srow_x[1:3], nii.header.srow_y[1:3], nii.header.srow_z[1:3]


hdr=nii.header

dx = hdr.pixdim[2]
dy = hdr.pixdim[3]
# aka qfac left handedness
if hdr.pixdim[1] < 0  
    dz = -hdr.pixdim[4]
else
    dz = hdr.pixdim[4]
end
b = hdr.quatern_b
c = hdr.quatern_c
d = hdr.quatern_d
b2 = b*b
c2 = c*c
d2 = d*d
a = 1 - b2 - c2 - d2
if a < 1.e-7
    a = 1 / sqrt(b2 + c2 + d2)
    b *= a
    c *= a
    d *= a       # normalize (b,c,d) vector
    a = zero(a)  # a = 0 ==> 180 degree rotation
else
    a = sqrt(a)   # angle = 2*arccos(a)
end

NIfTI._dir2ori(
            (a*a+b*b-c*c-d*d)*dx,  (2*b*c-2*a*d)*dy,      (2*b*d+2*a*c)*dz,
            (2*b*c+2*a*d)*dx,      (a*a+c*c-b*b-d*d)*dy,  (2*c*d-2*a*b)*dz,
            (2*b*d-2*a*c)*dx,      (2*c*d+2*a*b)*dy,      (a*a+d*d-c*c-b*b)*dz,
        )


        xi, xj, xk, yi, yj, yk, zi, zj, zk=       (a*a+b*b-c*c-d*d)*dx,  (2*b*c-2*a*d)*dy,      (2*b*d+2*a*c)*dz,
        (2*b*c+2*a*d)*dx,      (a*a+c*c-b*b-d*d)*dy,  (2*c*d-2*a*b)*dz,
        (2*b*d-2*a*c)*dx,      (2*c*d+2*a*b)*dy,      (a*a+d*d-c*c-b*b)*dz


    # Normalize column vectors to get unit vectors along each ijk-axis
    # normalize i axis
    val = sqrt(xi*xi + yi*yi + zi*zi)
    if val == 0
        error("Invalid rotation directions.")
    end
    xi /= val
    yi /= val
    zi /= val

    # normalize j axis
    val = sqrt(xj*xj + yj*yj + zj*zj)
    if val == 0
        error("Invalid rotation directions.")
    end
    xj /= val
    yj /= val
    zj /= val

    # orthogonalize j axis to i axis, if needed
    val = xi*xj + yi*yj + zi* zj  # dot product between i and j
    if abs(val) > .0001
        xj -= val*xi
        yj -= val*yi
        zj -= val*zi

        val = sqrt(xj*xj + yj*yj + zj*zj)  # must renormalize
        if val == 0
            error("The first and second dimensions cannot be parallel.")
        end
        xj /= val
        yj /= val
        zj /= val
    end

    # normalize k axis; if it is zero, make it the cross product i x j
    val = sqrt(xk*xk + yk*yk + zk*zk)
    if val == 0
        xk = yi*zj-zi*yj
        yk = zi*xj-zj*xi
        zk = xi*yj-yi*xj
    else
        xk = xk/val
        yk = yk/val
        zk = zk/val
    end

    # orthogonalize k to i
    val = xi*xk + yi*yk + zi*zk  # dot product between i and k
    if abs(val) > 0.0001
        xk -= val*xi
        yk -= val*yi
        zk -= val*zi

        # must renormalize
        val = sqrt(xk*xk + yk*yk + zk*zk)
        if val == 0
            return 0  # I think this is suppose to be an error output
        end
        xk /= val
        yk /= val
        zk /= val
    end

    # orthogonalize k to j */
    val = xj*xk + yj*yk + zj*zk  # dot product between j and k
    if abs(val) > 0.0001
        xk -= val*xj
        yk -= val*yj
        zk -= val*zj

        val = sqrt(xk*xk + yk*yk + zk*zk)
        if val == 0
            return 0  # bad
        end
        xk /= val
        yk /= val
        zk /= val
    end

    # at this point Q is the rotation matrix from the (i,j,k) to (x,y,z) axes
    detQ = NIfTI._det(xi, xj, xk, yi, yj, yk, zi, zj, zk)
    # if( detQ == 0.0 ) return ; /* shouldn't happen unless user is a DUFIS */

    # Build and test all possible +1/-1 coordinate permutation matrices P;
    # then find the P such that the rotation matrix M=PQ is closest to the
    # identity, in the sense of M having the smallest total rotation angle.

    # Despite the formidable looking 6 nested loops, there are
    # only 3*3*3*2*2*2 = 216 passes, which will run very quickly.
    vbest = -666
    ibest = pbest=qbest=rbest= 1
    jbest = 2
    kbest = 3
    for (i, j, k) in ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
        for p in (-1, 1)           # p,q,r are -1 or +1
            for q in (-1, 1)       # and go into rows 1,2,3
                for r in (-1, 1)
                    p11, p12, p13 = NIfTI._nval_other_zero(i, p)
                    p21, p22, p23 = NIfTI._nval_other_zero(j, q)
                    p31, p32, p33 = NIfTI._nval_other_zero(k, r)
                    #=
                    P[1,i] = p
                    P[2,j] = q
                    P[3,k] = r
                    detP = det(P)  # sign of permutation
                    =#
                    detP = NIfTI._det(p11, p12, p13, p21, p22, p23, p31, p32, p33)
                    # doesn't match sign of Q
                    if detP * detQ >= 0.0
                        # angle of M rotation = 2.0 * acos(0.5 * sqrt(1.0 + trace(M)))
                        # we want largest trace(M) == smallest angle == M nearest to I
                        val = NIfTI._mul_trace(
                            p11, p12, p13, p21, p22, p23, p31, p32, p33,
                            xi, xj, xk, yi, yj, yk, zi, zj, zk
                        )
                        if val > vbest
                            vbest = val
                            ibest = i
                            jbest = j
                            kbest = k
                            pbest = p
                            qbest = q
                            rbest = r
                        end
                    end
                end
            end
        end
    end

    ibest,pbest,jbest,qbest,kbest,rbest
# function get_dir_nifti(path)
#     nii = niread(path)
#     direction = nii.header.srow_x[1:3], nii.header.srow_y[1:3], nii.header.srow_z[1:3]
#     return tuple(direction[1][1], direction[2][1], direction[3][1], direction[1][2], direction[2][2], direction[3][2], direction[1][3], direction[2][3], direction[3][3])
# end

# function get_dir_nifti(path)
#     nii = niread(path)
#     affine =  NIfTI.get_sform(nii.header)
#     direction = (affine[1, 1:3], affine[2, 1:3], affine[3, 1:3])
#     return direction
# end
using LinearAlgebra

function get_dir_nifti(path)
    nii = niread(path)
    affine =  NIfTI.get_sform(nii.header)
    direction = normalize(affine)
    return tuple(direction...)
end

orientation
get_dir_nifti(path_curr)
# function test_get_dir_nifti(path)
#     expected_direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
#     return get_dir_nifti(path) == expected_direction
# end
# test_get_dir_nifti(path_curr)

function getSimpleItkObject()
    return sitk
end

sitk = getSimpleItkObject()
reader = sitk.ImageSeriesReader()
im=sitk.ReadImage(path_curr)
im.GetSpacing()
im.GetDirection()



