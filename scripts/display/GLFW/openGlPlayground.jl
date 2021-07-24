using Distributed: length

using DrWatson
@quickactivate "Probabilistic medical segmentation"

using GLFW: Window
using BenchmarkTools: minimum
using StaticArrays

dirToWorkerNumbs = DrWatson.scriptsdir("mainPipeline","processesDefinitions","workerNumbers.jl")
include(dirToWorkerNumbs)
using Main.workerNumbers
using Distributed


include(DrWatson.scriptsdir("display","GLFW","modernGL","fromGlMakie.jl"))
include(DrWatson.scriptsdir("display","GLFW","modernGL","textureManag.jl"))
include(DrWatson.scriptsdir("display","GLFW","modernGL","coordinateDisplay.jl"))

include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/structs/forDisplayStructs.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/loadData/manageH5File.jl")


scripts/display/GLFW/coordinateDisplay.jl
using Main.h5manag




#data source
exampleDat = Int16.(Main.h5manag.getExample())
exampleLabels = UInt8.(Main.h5manag.getExampleLabels())
dims = size(exampleDat)
widthh=dims[2]
heightt=dims[3]

#prepared =  prepareForDisplayOfTransverses(exampleDat, dims)



#############
#order of texture uploads  is important and texture 0 should be last binded as far as I get it 
stopListening[]=true
glClearColor(0.0, 0.0, 0.1 , 1.0)

#update labels
updateTexture(Int16,widthh,heightt,exampleLabels[200,:,:], trueLabels,stopListening,pboId, DATA_SIZE,GL_UNSIGNED_BYTE)
#update main image
updateTexture(Int16,widthh,heightt,exampleDat[200,:,:], mainTexture,stopListening,pboId, DATA_SIZE, GL_SHORT)
basicRender()
stopListening[]= false

############################# control scrolling
GLFW.SetScrollCallback(window, (_, xoff, yoff) -> begin 
print(yoff)
end  )


# stopListening[]=true

# updateTexture(Int16,widthh,heightt,modifyData(exampleDat, Int(yoff+currentSlice)), previousTexture,stopListening,pboId, DATA_SIZE)

# stopListening[]= false





controllScrollingDoc = """
controll swithing the slices while scrolling
"""
@doc controllScrollingDoc
function controllScrolling(  yoff::Int)
    print(Int(yoff+currentSlice))
    #stopListening[]=true
    updateTexture(Int16,widthh,heightt,modifyData(exampleDat, Int(yoff+currentSlice)), previousTexture,stopListening,pboId, DATA_SIZE)
	#stopListening[]= false

end





##############
bufSize = 32
name = zeros(UInt8, bufSize)
buflen = Ref{GLsizei}(0)
size1 = Ref{GLint}(0)
type = Ref{GLenum}()
#glGetActiveUniform(fragment_shader, 3, bufSize, buflen, size1, type, name)
glGetUniformfv(fragment_shader, 3, bufSize, buflen, size1, type, name)
String(name)


glGetUniformLocation(shader_program,"msk0")