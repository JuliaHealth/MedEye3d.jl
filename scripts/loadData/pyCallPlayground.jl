using Conda
Conda.add("SimpleITK")
using PyCall
sitk = pyimport("SimpleITK")
pyimport_conda("SimpleITK", "SimpleITK")
add SimpleITK