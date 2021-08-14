

     using DrWatson
     @quickactivate "Probabilistic medical segmentation"
     include(DrWatson.scriptsdir("display","GLFW","includeAll.jl"))
     include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))

     using Main.ForDisplayStructs,ColorTypes, Dictionaries,Main.DisplayWords, Setfield
     using  Main.ReactToScroll, Main.SegmentationDisplay,Main.Uniforms, Main.OpenGLDisplayUtils
     using Rocket ,GLFW , Main.DataStructs, Main.StructsManag,Main.TextureManag, Main.SegmentationDisplay
  listOfTexturesToCreate = [
     Main.ForDisplayStructs.TextureSpec(
         name = "mainLab",
         dataType= UInt8,
         strokeWidth = 5,
         color = RGB(1.0,0.0,0.0)
        ),
     Main.ForDisplayStructs.TextureSpec(
         name = "testLab1",
         numb= Int32(1),
         dataType= UInt8,
         color = RGB(0.0,1.0,0.0)
        ),
         Main.ForDisplayStructs.TextureSpec(
         name = "testLab2",
         numb= Int32(2),
         dataType= UInt8,
         color = RGB(0.0,0.0,1.0)
          ),
          Main.ForDisplayStructs.TextureSpec(
           name = "textText",
           isTextTexture = true,
           dataType= UInt8,
           color = RGB(0.0,0.0,1.0)
         ),
         Main.ForDisplayStructs.TextureSpec(
         name= "CTIm",
         numb= Int32(3),
         isMainImage = true,
         dataType= Int16)  
  ]
  #   

  Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,40,40, 300,300)
   
#  mainMaskDummy = UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
#  testLab1Dat =UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
#  testLab2Dat= UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))

 mainMaskDummy =  zeros(UInt8,10,40,40)
 testLab1Dat = zeros(UInt8,10,40,40)
 testLab2Dat=  zeros(UInt8,10,40,40)
    
 ctDummy =  Int16.(map(xx-> (xx >0 ? 1 : 0), rand(Int16,10,40,40)))# will give white background for testing 


   slicesDat=  [ThreeDimRawDat{UInt8}(UInt8,"mainLab",mainMaskDummy)
     ,ThreeDimRawDat{Int16}(Int16,"CTIm",ctDummy)
     ,ThreeDimRawDat{UInt8}(UInt8,"testLab2",testLab2Dat)
     ,ThreeDimRawDat{UInt8}(UInt8,"testLab1",testLab1Dat)  ]
     mainScrollDat = FullScrollableDat(dimensionToScroll=1,dataToScroll= slicesDat )


    Main.SegmentationDisplay.passDataForScrolling(mainScrollDat)

    
        singleSliceDat = [
      TwoDimRawDat{UInt8}(UInt8,"mainLab", UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,40,40))))
     ,TwoDimRawDat{Int16}(Int16,"CTIm",Int16.(map(xx-> (xx >0 ? 1 : 0), rand(Int16,40,40))))
     ,TwoDimRawDat{UInt8}(UInt8,"testLab2",UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,40,40))))
     ,TwoDimRawDat{UInt8}(UInt8,"testLab1",UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,40,40)))) 
    ]

    exampleSingleSliceDat = SingleSliceDat(listOfDataAndImageNames=singleSliceDat)
    Main.SegmentationDisplay.updateSingleImagesDisplayed(exampleSingleSliceDat)


    window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window

    GLFW.PollEvents()

 
         textLiverMain = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
         textureB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]
         textureC = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[3]
         textTexture = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[4]
         textureCt = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[5]
         
         Main.SegmentationDisplay.mainActor.actor.textureToModifyVec= [textureC]
        
         dispObj= Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects
         wordsDispObj= Main.SegmentationDisplay.mainActor.actor.textDispObj
         stopList = dispObj.stopListening[]






  
         dispObj.stopListening[]= true
    glClearColor(0.0, 0.0, 0.1 , 1.0)
    glBindTexture(GL_TEXTURE_2D, 0); 
   # bindAndActivateForText(wordsDispObj.shader_program_words ,    wordsDispObj.fragment_shader_words,wordsDispObj.vbo_words,dispObj.vertex_shader )
    
   
    
    wordsDispObj.textureSpec.ID
    activateForTextDisp( wordsDispObj.shader_program_words, wordsDispObj.vbo_words )
    updateTexture(UInt8, ones(UInt8,1000,10000),wordsDispObj.textureSpec)
    basicRender(window)

    dispObj.stopListening[]= false




    reactivateMainObj(dispObj.shader_program ,dispObj.vbo  )



Main.SegmentationDisplay.updateSingleImagesDisplayed(exampleSingleSliceDat)
basicRender(window)

GLFW.PollEvents()
#     glUseProgram(shader_program_words)
#     glBindBuffer(GL_ARRAY_BUFFER, vbo_words[])
#     glBufferData(GL_ARRAY_BUFFER, sizeof(Main.ShadersAndVerticiesForText.verticesB), Main.ShadersAndVerticiesForText.verticesB, GL_STATIC_DRAW)
#   	encodeDataFromDataBuffer()


#     glClearColor(0.0, 0.0, 0.1 , 1.0)

#          stopList= true
        #  bindAndActivateForText(wordsDispObj.shader_program_words ,
        #   wordsDispObj.fragment_shader_words,wordsDispObj.vbo_words,dispObj.vertex_shader )
#          basicRender(window)
#          texId =  createTexture(0,Int32(1000), Int32(10000),GL_R8UI)
#          textSpec = wordsDispObj.textureSpec
#          textSpec= setproperties(textSpec,(ID=texId) )

#          samplerRef= glGetUniformLocation(wordsDispObj.shader_program_words, "TextTexture1")
#          glUniform1i(samplerRef,length(listOfTexturesToCreate)+1)






#          textSpec= wordsDispObj.textureSpec


#          data = ones(UInt8,1000,10000)
#          glActiveTexture(textSpec.actTextrureNumb); # active proper texture unit before binding
#          glBindTexture(GL_TEXTURE_2D, textSpec.ID[]); 
#          glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 1000, 10000, GL_RED_INTEGER, textSpec.OpGlType, collect(data))

#          basicRender(window)

#         # texId=  createTexture(0,Int32(100), Int32(1000),GL_R8UI)
#         # indexOfActiveText = 8


#         # widthh=Int32(100)
#         # heightt =Int32(1000)
#         # textureSpec= setproperties(wordsDispObj.textureSpec, 
#         # (ID=texId ,actTextrureNumb =getProperGL_TEXTURE(indexOfActiveText)
#         # ,OpGlType =GL_UNSIGNED_BYTE
#         # ,widthh = widthh
#         # ,heightt=heightt ))
        
#         # textureSpec.OpGlType == GL_UNSIGNED_BYTE
#         # textureSpec.actTextrureNumb == GL_TEXTURE8



        
#         data= ones(UInt8,100,1000)

#         xoffset=0
#         yoffset=0
#         widthh=textSpec.widthh
#         heightt =textSpec.heightt

#         glBindTexture(GL_TEXTURE_2D, textSpec.ID[]); 
#         glActiveTexture(textSpec.actTextrureNumb); # active proper texture unit before binding
#         glTexSubImage2D(GL_TEXTURE_2D,0,xoffset,yoffset, widthh, heightt, GL_RED_INTEGER, textSpec.OpGlType, collect(data))
#         basicRender(window)


#         glBindTexture(GL_TEXTURE_2D, texId[]); 
#         samplerRef= glGetUniformLocation(wordsDispObj.shader_program_words, "TextTexture1")
#         glUniform1i(samplerRef,indexOfActiveText);
       
#         glBindTexture(GL_TEXTURE_2D, texId[]); 
#         glActiveTexture(GL_TEXTURE8);
#         data= zeros(UInt8,100,1000)
#         glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 100, 1000, GL_RED_INTEGER, GL_UNSIGNED_BYTE, collect(data))     
#         basicRender(window)

#         updateTexture(UInt8, ones(UInt8,100,1000),wordsDispObj.textureSpec)
#         basicRender(window)


#         updateTexture(UInt8, zeros(UInt8,100,1000),textureSpec)
#         basicRender(window)

#         updateTexture(UInt8, zeros(UInt8,100,1000),textureSpec)
#         basicRender(window)

#         glClearColor(0.0, 0.0, 0.1 , 1.0)
#         updateTexture(UInt8, zeros(UInt8,100,1000),textureSpec)
#         basicRender(window)

#         stopList= false

#         # zz= ThreeDimRawDat{UInt8}(UInt8,"mainLab",mainMaskDummy)   
#         # sa= zz.dat
#         # maximum(sa)
#         # tD= threeToTwoDimm(zz.type,2,2,zz  )
#         # slll= collect(modSlice!(tD, [CartesianIndex(1,1)], UInt8(3)))
#         # maximum(tD.dat)
#         # maximum(sa)



#         #  setTextureVisibility(true ,textLiverMain.uniforms)

#         #  setMaskColor(RGB(1.0,0.0,0.0) ,textLiverMain.uniforms)
       
#         #  setMaskColor(RGB(0.0,1.0,0.0) ,textTexture.uniforms)
#         #  setTextureVisibility(true ,textTexture.uniforms)
       
#         #    setMaskColor(RGB(1.0,0.0,0.5) ,textureC.uniforms)
#         #    setTextureVisibility(true ,textureC.uniforms)
       
       
#         #  setMaskColor(RGB(0.5,0.5,0.0) ,textureB.uniforms)
#         #  setTextureVisibility(true ,textureB.uniforms)
       
       
       
#         # setCTWindow(Int32(0), Int32(0),Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[5].uniforms)
       






#          GLFW.PollEvents()


# z=zeros(3,3,3)

# selectdim(z,1,1)
# TwoDimRawDat()

        #  setVisOnKey(Option(listOfTexturesToCreate[1]))

#     zz = ThreeDimRawDat{UInt8}(UInt8,"mainLab",ones(UInt8,10,40,40))

#    twoModded =  threeToTwoDimm(UInt8, 2, 1, zz)

#   sl =  modSlice!(twoModded, [CartesianIndex(1,1), CartesianIndex(1,2)],UInt8(2) )

#  coords= [CartesianIndex(1,1), CartesianIndex(1,2)]

#  ddd = modifySliceFull!(mainScrollDat, 1,coords,"mainLab",UInt8(5))

#  getSlicesNumber(mainScrollDat)

#      using DrWatson
#      @quickactivate "Probabilistic medical segmentation"
     
#      include(DrWatson.scriptsdir("display","GLFW","includeAll.jl"))




    #  using ModernGL, GeometryTypes, GLFW
    #  using Main.PrepareWindowHelpers
    #  include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))
    #  using  Main.OpenGLDisplayUtils
    #  using Main.ShadersAndVerticies, Main.ForDisplayStructs,Main.ShadersAndVerticiesForText
    #  using ColorTypes,Main.TextureManag, Main.PrepareWindow, Setfield
#  using  Main.Uniforms


#      listOfTexturesToCreate = [
#       Main.ForDisplayStructs.TextureSpec(  name = "mainLab",          dataType= UInt8,          strokeWidth = 5,          color = RGB(1.0,0.0,0.0)         ),      Main.ForDisplayStructs.TextureSpec(          name = "testLab1",          numb= Int32(1),          dataType= UInt8,          color = RGB(0.0,1.0,0.0)         ),
#           Main.ForDisplayStructs.TextureSpec(          name = "testLab2",          numb= Int32(2),          dataType= UInt8,          color = RGB(0.0,0.0,1.0)           ),           Main.ForDisplayStructs.TextureSpec(            name = "textText",            isTextTexture = true,            dataType= UInt8,            color = RGB(0.0,0.0,1.0)          ),          Main.ForDisplayStructs.TextureSpec(          name= "CTIm",          numb= Int32(3),          isMainImage = true,          dataType= Int16)              ]
      



#      window,vertex_shader,fragment_shader ,shader_program,stopListening,vbo,ebo,fragment_shader_words,vbo_words = Main.PrepareWindow.displayAll(200,200,listOfTexturesToCreate)

#      # than we set those uniforms, open gl types and using data from arguments  to fill texture specifications
#      mainImageUnifs,listOfTextSpecsMapped= SegmentationDisplay.assignUniformsAndTypesToMasks(listOfTexturesToCreate,shader_program) 
#       listOfTextSpecsMapped=map((spec)-> setproperties(spec, (widthh= 40, heightt= 40 )) 
#                                               ,listOfTextSpecsMapped)
#       listOfTextSpecsMapped=   initializeTextures(listOfTextSpecsMapped)




#                                              stopListening[]=true
#   GLFW.PollEvents()

#      textureLiver = listOfTextSpecsMapped[1].ID
#      textureSecond =  listOfTextSpecsMapped[2].ID
    
#      listOfTextSpecsMapped[2].actTextrureNumb
    
#      aa=zeros(UInt8,40,40)
#      bb=zeros(UInt8,40,40)

#      aa[2,1]=1
#      aa[2,2]=1
#      aa[3,3]=1

#      bb[1,1]=1
#      bb[2,1]=1
#      bb[2,5]=1

#      livSamplerRef=  listOfTextSpecsMapped[1].uniforms.samplerRef
#      secSamplerRef= listOfTextSpecsMapped[2].uniforms.samplerRef

#       glUniform1i(livSamplerRef, 0) # read from active texture 0
#       glUniform1i(secSamplerRef, 1) # read from active texture 1
     
#      # load texture

#      setTextureVisibility(true , listOfTextSpecsMapped[1].uniforms)
#      setMaskColor(RGB(1.0,0.0,0.0) , listOfTextSpecsMapped[1].uniforms)
   
#      setTextureVisibility(true ,listOfTextSpecsMapped[2].uniforms)
#      setMaskColor(RGB(0.0,1.0,0.0) ,listOfTextSpecsMapped[2].uniforms)
   



#     glClearColor(0.0, 0.0, 0.1 , 1.0)

#      glActiveTexture(GL_TEXTURE0)
#      glBindTexture(GL_TEXTURE_2D, textureLiver[])
#      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,aa)
   

#      glActiveTexture(GL_TEXTURE1)
#      glBindTexture(GL_TEXTURE_2D, textureSecond[])
#      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,bb)
#      basicRender(window)
#      stopListening[]=false



    #  glActiveTexture(GL_TEXTURE0)
    #  glBindTexture(GL_TEXTURE_2D, textureLiver[])
    #  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,aa)
    #  basicRender(window)

    #  glBindTexture(GL_TEXTURE_2D, textureSecond[])
    #  glActiveTexture(GL_TEXTURE1)
    #  glBindTexture(GL_TEXTURE_2D, textureSecond[])
    #  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,bb)
    #  basicRender(window)







    #  glActiveTexture(GL_TEXTURE0); # active proper texture unit before binding
    #  glBindTexture(GL_TEXTURE_2D, textureLiver[])
    # # glUniform1i(samplerRefNumb,index);# we first look for uniform sampler in shader  
    #  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,aa)
    #  basicRender(window)


    #  glActiveTexture(GL_TEXTURE1); # active proper texture unit before binding
    #  glBindTexture(GL_TEXTURE_2D, textureSecond[])
    # # glUniform1i(samplerRefNumb,index);# we first look for uniform sampler in shader  
    #  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,bb)
    #  basicRender(window)





























#      using DrWatson
#      @quickactivate "Probabilistic medical segmentation"
     
#      include(DrWatson.scriptsdir("display","GLFW","includeAll.jl"))




#      using ModernGL, GeometryTypes, GLFW
#      using Main.PrepareWindowHelpers
#      include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))
#      using  Main.OpenGLDisplayUtils
#      using Main.ShadersAndVerticies, Main.ForDisplayStructs,Main.ShadersAndVerticiesForText
#      using ColorTypes,Main.TextureManag
 
#   window = initializeWindow(200,200)

#   listOfTexturesToCreate = [
#     Main.ForDisplayStructs.TextureSpec(
#         name = "mainLab",
#         dataType= UInt8,
#         strokeWidth = 5,
#         color = RGB(1.0,0.0,0.0)
#        ),
#     Main.ForDisplayStructs.TextureSpec(
#         name = "testLab1",
#         numb= Int32(1),
#         dataType= UInt8,
#         color = RGB(0.0,1.0,0.0)
#        ),
#         Main.ForDisplayStructs.TextureSpec(
#         name = "testLab2",
#         numb= Int32(2),
#         dataType= UInt8,
#         color = RGB(0.0,0.0,1.0)
#          ),
#          Main.ForDisplayStructs.TextureSpec(
#           name = "textText",
#           isTextTexture = true,
#           dataType= UInt8,
#           color = RGB(0.0,0.0,1.0)
#         ),
#         Main.ForDisplayStructs.TextureSpec(
#         name= "CTIm",
#         numb= Int32(3),
#         isMainImage = true,
#         dataType= Int16)  
#           ]
    

#    	# The shaders 
# 	println(createcontextinfo())
# 	gslsStr = get_glsl_version_string()

# 	vertex_shader = createVertexShader(gslsStr)
# 	fragment_shader_main = createFragmentShader(gslsStr,listOfTexturesToCreate)
	
# 		# Connect the shaders by combining them into a program
# 	shader_program = glCreateProgram()

# 	glAttachShader(shader_program, vertex_shader)
# 	glAttachShader(shader_program, fragment_shader_main)
	
# 	glLinkProgram(shader_program)
# 	glUseProgram(shader_program)
	
# 	###########buffers
# 	#create vertex buffer
# 	createVertexBuffer()
# 	# Create the Vertex Buffer Objects (VBO)
# 	vbo = createDAtaBuffer(Main.ShadersAndVerticies.vertices)

# 	# Create the Element Buffer Object (EBO)
# 	ebo = createElementBuffer(Main.ShadersAndVerticies.elements)
# 	############ how data should be read from data buffer
# 	encodeDataFromDataBuffer()

#   GLFW.PollEvents()

#      textureLiver = createTexture(0,Int32(40),Int32(40), GL_R8UI)
#      textureSecond = createTexture(0,Int32(40),Int32(40), GL_R8UI)
    
  
    
#      aa=zeros(UInt8,40,40)
#      bb=zeros(UInt8,40,40)

#      aa[2,1]=1
#      aa[2,2]=1
#      aa[2,3]=1

#      bb[1,1]=1
#      bb[2,1]=1
#      bb[2,5]=1

#      livSamplerRef=  glGetUniformLocation(shader_program, "mainLab")
#      secSamplerRef= glGetUniformLocation(shader_program, "testLab1")

#      glUniform1i(livSamplerRef, 0) # read from active texture 0
#      glUniform1i(secSamplerRef, 1) # read from active texture 1
     
#      # load texture


     
#      glActiveTexture(GL_TEXTURE0)
#      glBindTexture(GL_TEXTURE_2D, textureLiver[])
#      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,aa)
   

#      glActiveTexture(GL_TEXTURE1)
#      glBindTexture(GL_TEXTURE_2D, textureSecond[])
#      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,bb)
#      basicRender(window)



#     #  glActiveTexture(GL_TEXTURE0)
#     #  glBindTexture(GL_TEXTURE_2D, textureLiver[])
#     #  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,aa)
#     #  basicRender(window)

#     #  glBindTexture(GL_TEXTURE_2D, textureSecond[])
#     #  glActiveTexture(GL_TEXTURE1)
#     #  glBindTexture(GL_TEXTURE_2D, textureSecond[])
#     #  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,bb)
#     #  basicRender(window)




# using Glutils
#      @uniforms! begin

#     glGetUniformLocation(shader_program, "mainLabColorMask"):= Cfloat[1.0, 0.3, 0.0, 0.8]
#      glGetUniformLocation(shader_program, "testLab1ColorMask"):= Cfloat[0.3, 1.0, 0.0, 0.8]
   
#      glGetUniformLocation(shader_program, "mainLabisVisible"):= 1
#      glGetUniformLocation(shader_program, "testLab1isVisible"):= 1
#     end


#      glActiveTexture(GL_TEXTURE0); # active proper texture unit before binding
#      glBindTexture(GL_TEXTURE_2D, textureLiver[])
#     # glUniform1i(samplerRefNumb,index);# we first look for uniform sampler in shader  
#      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,aa)
#      basicRender(window)


#      glActiveTexture(GL_TEXTURE1); # active proper texture unit before binding
#      glBindTexture(GL_TEXTURE_2D, textureSecond[])
#     # glUniform1i(samplerRefNumb,index);# we first look for uniform sampler in shader  
#      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,bb)
#      basicRender(window))



































# using Base: Int16

#      using DrWatson
#      @quickactivate "Probabilistic medical segmentation"
     
#      include(DrWatson.scriptsdir("display","GLFW","includeAll.jl"))

#      #  include(DrWatson.scriptsdir("loadData","manageH5File.jl"))
#     #  using Main.h5manag


     
     
#      using Revise 
#      using Main.SegmentationDisplay
#      #data about textures we want to create
#      using  Main.ForDisplayStructs
#      using Parameters, ColorTypes
     
#      # list of texture specifications, important is that main texture - main image should be specified first
#      #Order is important !
#      listOfTexturesToCreate = [
#      Main.ForDisplayStructs.TextureSpec(
#          name = "mainLab",
#          dataType= UInt8,
#          strokeWidth = 5,
#          color = RGB(1.0,0.0,0.0)
#         ),
#      Main.ForDisplayStructs.TextureSpec(
#          name = "testLab1",
#          numb= Int32(1),
#          dataType= UInt8,
#          color = RGB(0.0,1.0,0.0)
#         ),
#          Main.ForDisplayStructs.TextureSpec(
#          name = "testLab2",
#          numb= Int32(2),
#          dataType= UInt8,
#          color = RGB(0.0,0.0,1.0)
#           ),
#           Main.ForDisplayStructs.TextureSpec(
#            name = "textText",
#            isTextTexture = true,
#            dataType= UInt8,
#            color = RGB(0.0,0.0,1.0)
#          ),
#          Main.ForDisplayStructs.TextureSpec(
#          name= "CTIm",
#          numb= Int32(3),
#          isMainImage = true,
#          dataType= Int16)  
#            ]
     
     
     
           
#          using Main.ReactingToInput
     
       
         
     
#          using Main.ForDisplayStructs
#          using Main.ReactToScroll
#          using Main.SegmentationDisplay
#          using Rocket
         
         
#      #data source
#      # exampleDat = Int16.(Main.h5manag.getExample())
#      # exampleLabels = UInt8.(Main.h5manag.getExampleLabels())
#      # dims = size(exampleDat)
#      # widthh=dims[2]
#      # heightt=dims[3]
#      # slicesNumb= dims[1]
     
#      #  slice = 200
#      #   listOfDataAndImageNamesSlice = [("mainLab",exampleLabels[slice,:,:]),("CTIm",exampleDat[slice,:,:] )]
     
     
#      #     listOfDataAndImageNames = [("mainLab",exampleLabels),("CTIm",exampleDat)]
       
         
#      #     imagedims=dims
#      #     imageWidth = dims[2]
#      #     imageHeight = dims[3]
#      #configuring
#         # Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,512,512, 1000,800)
     
      
#          # Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)
     
#          # Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,200 )
     
     
#      #  window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
#          # stopListening = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.stopListening
         
#        #  textSpec = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
#          # textSpecB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]
     
     
     
#          Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,40,40, 300,300)
     
     
#         ### playing with uniforms
#         program = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.shader_program
#         shader_program =program
         
     
     
     
#      ###### main data ...
     
     
     
     
     
#     #  testLab1Dat =UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
#     #  testLab2Dat= UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
     
#     #  mainMaskDummy = UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
# #    ctDummy =  Int16.(map(xx-> (xx >0 ? 1 : 0), rand(Int16,10,40,40)))# will give white background for testing 

#     testLab1Dat =ones(UInt8,10,40,40)
#      testLab2Dat= ones(UInt8,10,40,40)
     
#      mainMaskDummy = ones(UInt8,10,40,40)
     
     
#      ctDummy = ones(Int16,10,40,40)
#              listOfDataAndImageNames = [("mainLab",mainMaskDummy)
#          ,("CTIm",ctDummy )  
#          ,("testLab2",testLab2Dat) 
#          ,("testLab1",testLab1Dat) ]
#          #,("testLab2",zeros(Int8,10,512,512))
     
#          Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)
     
#          slicee = 3
#          listOfDataAndImageNamesSlice = [ ("mainLab",mainMaskDummy[slicee,:,:]) ,  ("testLab2",testLab2Dat[slicee,:,:]) 
#            ,("testLab1",testLab1Dat[slicee,:,:]) 
#          ,("CTIm",ctDummy[slicee,:,:] )]
     
     


     
#             Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,3 )
     
     
     
#          window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
         
     
#          textLiverMain = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
#          textureB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]
#          textureC = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[3]
#          textTexture = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[4]
#          textureCt = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[5]
         
#          Main.SegmentationDisplay.mainActor.actor.textureToModifyVec= [textureB]
#          dattt = Main.SegmentationDisplay.mainActor.actor.onScrollData[4][2]
#      maximum(dattt)
     
     

     
#      using Main.Uniforms, Main.OpenGLDisplayUtils, Main.OpenGLDisplayUtils
     

     
      #  setTextureVisibility(true ,textLiverMain.uniforms)
      #  setMaskColor(RGB(0.8,0.0,0.1) ,textLiverMain.uniforms)
     
      #  setMaskColor(RGB(0.0,1.0,0.0) ,textTexture.uniforms)
      #  setTextureVisibility(true ,textTexture.uniforms)
     
      #    setMaskColor(RGB(1.0,0.0,0.5) ,textureC.uniforms)
      #    setTextureVisibility(true ,textureC.uniforms)
     
     
      #  setMaskColor(RGB(0.5,0.5,0.0) ,textureB.uniforms)
      #  setTextureVisibility(true ,textureB.uniforms)
     
     
     
      # setCTWindow(Int32(0), Int32(0),Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[5].uniforms)
     
     

#      aa=zeros(UInt8,40,40)
#      bb=zeros(UInt8,40,40)

#     #  aa[1,1]=1
#     #  bb[2,2]=1


#     bb[1,1]=1
#     #  bb[2,1]=1
#     #  bb[2,5]=1


#      cc= convert(Vector{Tuple{String, Array{T, 2} where T}} ,[ ("mainLab",aa),("testLab2",bb) ])


#      Main.SegmentationDisplay.updateSingleImagesDisplayed(convert(Vector{Tuple{String, Array{T, 2} where T}}
#       ,[ ("mainLab",aa),("testLab1",bb) ])
#      ,3 )

#      #    listOfDataAndImageNamesSlice = [ ("mainLab",mainMaskDummy[slicee,:,:])
#   #     ,  ("testLab2",testLab2Dat[slicee,:,:]) 
#   #    ,("testLab1",testLab1Dat[slicee,:,:]) 
#   #  ,("CTIm",ctDummy[slicee,:,:] )]




# cc = zeros(3,3,3)
# xx = [CartesianIndex(1,1),CartesianIndex(1,2)]
# cc[3,xx].=1
# cc
# cc[3,:,:]

#   mouseCoords= [CartesianIndex(150,150), CartesianIndex(100,150)]

#   obj =  Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects
#   obj.stopListening[]=true #free GLFW context
#   textureList =  [textureB]

#   if (!isempty(textureList))
#       texture= textureList[1]
      
#      mappedCoords =  translateMouseToTexture(texture.strokeWidth
#                                               ,mouseCoords
#                                               ,obj.windowWidth
#                                               , obj.windowHeight
#                                               , obj.imageTextureWidth
#                                               , obj.imageTextureHeight
#                                               ,Main.SegmentationDisplay.mainActor.actor.currentDisplayedSlice)

# bb[mappedCoords].=1
# bb[mappedCoords]

#  data = Main.SegmentationDisplay.mainActor.actor.onScrollData[1][2]
#  maximum(aa)
#  updateTexture(bb, texture)

#                                               for datTupl in   Main.SegmentationDisplay.mainActor.actor.onScrollData
#           if(datTupl[1]==texture.name)
#               datTupl[2][mappedCoords].=1 # broadcasting new value to all points that we are intrested in     
#               updateTexture(datTupl[2][ Main.SegmentationDisplay.mainActor.actor.currentDisplayedSlice,:,:], texture)
#               break
#           end#if
#       end #for


       
#       basicRender(obj.window)
     
#   end #if 
#   obj.stopListening[]=false # reactivete event listening loop





     







#      using Main.CustomFragShad
#          strr= Main.CustomFragShad.createCustomFramgentShader(listOfTexturesToCreate)
#          for st in split(strr, "\n")
#          @info st
#          end
#          using Main.ShadersAndVerticies
     
    
     
