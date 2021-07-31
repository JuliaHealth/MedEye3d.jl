using Base: Int16

     using DrWatson
     @quickactivate "Probabilistic medical segmentation"
     
     using Setfield
     using GLFW
     using ModernGL
     using ColorTypes

     include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/structs/forDisplayStructs.jl")
     include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/loadData/manageH5File.jl")
     using Main.h5manag

     include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindowHelpers.jl"))
     include(DrWatson.scriptsdir("display","GLFW","modernGL","OpenGLDisplayUtils.jl"))
     include(DrWatson.scriptsdir("display","GLFW","shadersEtc","ShadersAndVerticies.jl"))

     include(DrWatson.scriptsdir("display","GLFW","shadersEtc","Uniforms.jl"))

     
     include(DrWatson.scriptsdir("display","GLFW","modernGL","TextureManag.jl") )
     include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindow.jl"))
     

 

     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactToScroll.jl") )
     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactOnMouseClickAndDrag.jl") )

     include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl") )
     include(DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl"))




using Revise 
using Main.SegmentationDisplay
#data about textures we want to create
using  Main.ForDisplayStructs
using Parameters

# list of texture specifications, important is that main texture - main image should be specified first
#Order is important !
listOfTexturesToCreate = [
Main.ForDisplayStructs.TextureSpec(
    name = "grandTruthLiverLabel",
    dataType= UInt8,
    strokeWidth = 5,
    color = RGB(1.0,0.0,0.0),
   ),
Main.ForDisplayStructs.TextureSpec(
    name = "mainForModificationsTexture1",
    dataType= UInt8,
    color = RGB(0.0,1.0,0.0),
   ),
    Main.ForDisplayStructs.TextureSpec(
    name = "mainForModificationsTexture2",
    dataType= UInt8,
    color = RGB(0.0,0.0,1.0),
     )      
    ,Main.ForDisplayStructs.TextureSpec(
    name= "mainCTImage",
    dataType= Int16,)  
      ]
   
   
      listOfTexturesToCreate
    include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl"))
    using Main.ReactingToInput

    segmPath = DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl")
    include(segmPath)
    

    using Main.ForDisplayStructs
    using Main.ReactToScroll
    using Main.SegmentationDisplay
    using Rocket
    
    
#data source
# exampleDat = Int16.(Main.h5manag.getExample())
# exampleLabels = UInt8.(Main.h5manag.getExampleLabels())
# dims = size(exampleDat)
# widthh=dims[2]
# heightt=dims[3]
# slicesNumb= dims[1]

#  slice = 200
#   listOfDataAndImageNamesSlice = [("grandTruthLiverLabel",exampleLabels[slice,:,:]),("mainCTImage",exampleDat[slice,:,:] )]


#     listOfDataAndImageNames = [("grandTruthLiverLabel",exampleLabels),("mainCTImage",exampleDat)]
  
    
#     imagedims=dims
#     imageWidth = dims[2]
#     imageHeight = dims[3]
#configuring
   # Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,512,512, 1000,800)

 
    # Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)

    # Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,200 )


    # window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
    # stopListening = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.stopListening
    
  #  textSpec = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
    # textSpecB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]



    Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,40,40, 1000,800)



   ### playing with uniforms
   program = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.shader_program

    



###### main data ...



mainMaskDummy = zeros(UInt8,10,40,40)
ctDummy = ones(Int16,10,40,40)# will give white background for testing 
    listOfDataAndImageNames = [("grandTruthLiverLabel",mainMaskDummy),("mainCTImage",ctDummy ) ]
    #,("mainForModificationsTexture2",zeros(Int8,10,512,512))

    Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)
    slicee = 3
    listOfDataAndImageNamesSlice = [("grandTruthLiverLabel",mainMaskDummy[slicee,:,:]) ,("mainCTImage",ctDummy[slicee,:,:] )]


    Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,3 )
    textSpec = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
    push!(Main.SegmentationDisplay.mainActor.actor.textureToModifyVec, textSpec)

   # Main.SegmentationDisplay.mainActor.actor.textureToModifyVec= [listOfTexturesToCreate[1]]
  
  
  


   GLFW.PollEvents()






#    supertype(Int16)
#    supertype(UInt16)

#    supertype(Float32)

#    supertype(Int)==supertype(Int16)
   
#    supertype(Flo)==supertype(Int16)


# widen(widen(widen(Int16)))





#    @enum C a b c
   
#    fieldnames(C)
   
#    @enum uniformForShader(colorsMask0, colorsMask1, colorsMask2,isVisibleTexture0, isVisibleMask0,isVisibleMask1,isVisibleMask2)

#    using Glutils
#    @uniforms  (colorsMask0, colorsMask1, colorsMask2,isVisibleTexture0, isVisibleMask0,isVisibleMask1,isVisibleMask2) = program

#    c = Cfloat[0.0, 1.0, 0.0, 1.0]

#  colorSet =   [c,c,c ,c,c,c ,c]


#    @uniforms! begin
#    colorsMask0:=c
#    colorsMask1:=c
#    colorsMask2:=c
#    #isVisibleMask0:= true
#     end

# struct zz 
# a::Int
# b::Float32
# b::Float32

# end







# #adapted from http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/    
# FramebufferName = Ref(GLuint(0));
# glGenFramebuffers(1, FramebufferName);
# glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);


# define renderedTexture

# #Set "renderedTexture" as our colour attachement #0
# glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

# # Set the list of draw buffers.
# GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
# glDrawBuffers(1, DrawBuffers); # "1" is the size of DrawBuffers

# #Always check that our framebuffer is ok
# if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
# return false;


# #Render to our framebuffer
# glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
# glViewport(0,0,imageWidth,imageHeight); # Render on the whole framebuffer, complete from the lower left corner to the upper right
