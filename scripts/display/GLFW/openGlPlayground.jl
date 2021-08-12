using Base: Int16

     using DrWatson
     @quickactivate "Probabilistic medical segmentation"
     
     include(DrWatson.scriptsdir("display","GLFW","includeAll.jl"))

    #  include(DrWatson.scriptsdir("loadData","manageH5File.jl"))
    #  using Main.h5manag


     
     
     using Revise 
     using Main.SegmentationDisplay
     #data about textures we want to create
     using  Main.ForDisplayStructs
     using Parameters
     
     # list of texture specifications, important is that main texture - main image should be specified first
     #Order is important !
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
     
     
     
           
         using Main.ReactingToInput
     
       
         
     
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
     #   listOfDataAndImageNamesSlice = [("mainLab",exampleLabels[slice,:,:]),("CTIm",exampleDat[slice,:,:] )]
     
     
     #     listOfDataAndImageNames = [("mainLab",exampleLabels),("CTIm",exampleDat)]
       
         
     #     imagedims=dims
     #     imageWidth = dims[2]
     #     imageHeight = dims[3]
     #configuring
        # Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,512,512, 1000,800)
     
      
         # Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)
     
         # Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,200 )
     
     
     #  window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
         # stopListening = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.stopListening
         
       #  textSpec = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
         # textSpecB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]
     
     
     
         Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate,40,40, 300,300)
     
     
        ### playing with uniforms
        program = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.shader_program
        shader_program =program
         
     
     
     
     ###### main data ...
     
     
     
     
     
     testLab1Dat =UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
     testLab2Dat= UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
     
     mainMaskDummy = UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
     # testLab1Dat =rand(UInt8,10,40,40)
     # testLab2Dat= rand(UInt8,10,40,40)
     
     # mainMaskDummy = rand(UInt8,10,40,40)
     
     
     ctDummy =  Int16.(map(xx-> (xx >0 ? 1 : 0), rand(Int16,10,40,40)))# will give white background for testing 
         listOfDataAndImageNames = [("mainLab",mainMaskDummy)
         ,("CTIm",ctDummy )  
         ,("testLab2",testLab2Dat) 
         ,("testLab1",testLab1Dat) ]
         #,("testLab2",zeros(Int8,10,512,512))
     
         Main.SegmentationDisplay.passDataForScrolling(listOfDataAndImageNames)
     
         slicee = 3
         listOfDataAndImageNamesSlice = [ ("mainLab",mainMaskDummy[slicee,:,:]) ,  ("testLab2",testLab2Dat[slicee,:,:]) 
           ,("testLab1",testLab1Dat[slicee,:,:]) 
         ,("CTIm",ctDummy[slicee,:,:] )]
     
     


     
            Main.SegmentationDisplay.updateSingleImagesDisplayed(listOfDataAndImageNamesSlice,3 )
     
     
     
         window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window
         
     
         textLiverMain = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1]
         textureB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2]
         textureC = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[3]
         textTexture = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[4]
         textureCt = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[5]
         
         Main.SegmentationDisplay.mainActor.actor.textureToModifyVec= [textureC]
         dattt = Main.SegmentationDisplay.mainActor.actor.onScrollData[4][2]
     maximum(dattt)
     
     

     
     using Main.Uniforms, Main.OpenGLDisplayUtils, Main.OpenGLDisplayUtils
     

     
       setTextureVisibility(true ,textLiverMain.uniforms)
       setMaskColor(RGB(1.0,0.0,0.0) ,textLiverMain.uniforms)
     
       setMaskColor(RGB(0.0,1.0,0.0) ,textTexture.uniforms)
       setTextureVisibility(true ,textTexture.uniforms)
     
         setMaskColor(RGB(1.0,0.0,0.5) ,textureC.uniforms)
         setTextureVisibility(true ,textureC.uniforms)
     
     
       setMaskColor(RGB(0.5,0.5,0.0) ,textureB.uniforms)
       setTextureVisibility(true ,textureB.uniforms)
     
     
     
      setCTWindow(Int32(0), Int32(0),Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[5].uniforms)
     
     

      textLiverMain.ID
      textureB.ID
     
      aa=zeros(UInt8,40,40)
      bb=zeros(UInt8,40,40)
      cc=zeros(UInt8,40,40)
      aa[3,1]=1
      bb[3,1]=1
      cc[1,1]=1
        
      #liverMain
      glActiveTexture(GL_TEXTURE0+1 ); # active proper texture unit before binding
      glBindTexture(GL_TEXTURE_2D, textureB.ID[])
      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,textureB.OpGlType,cc)
      basicRender(window)

      # glActiveTexture(GL_TEXTURE0 +3); # active proper texture unit before binding
      # glBindTexture(GL_TEXTURE_2D, textureB.ID[])
      # glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,textureB.OpGlType,aa)
      # basicRender(window)



     
     using Main.CustomFragShad
         strr= Main.CustomFragShad.createCustomFramgentShader(listOfTexturesToCreate)
         for st in split(strr, "\n")
         @info st
         end
         using Main.ShadersAndVerticies
     
    
     
        GLFW.PollEvents()
     ############
     using DrWatson
     @quickactivate "Probabilistic medical segmentation"
     
     include(DrWatson.scriptsdir("display","GLFW","includeAll.jl"))




     using ModernGL, GeometryTypes, GLFW
     using Main.PrepareWindowHelpers
     include(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))
     using  Main.OpenGLDisplayUtils
     using Main.ShadersAndVerticies, Main.ForDisplayStructs,Main.ShadersAndVerticiesForText
     using ColorTypes,Main.TextureManag
 
  window = initializeWindow(200,200)

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
    

   	# The shaders 
	println(createcontextinfo())
	gslsStr = get_glsl_version_string()

	vertex_shader = createVertexShader(gslsStr)
	fragment_shader_main = createFragmentShader(gslsStr,listOfTexturesToCreate)
	
		# Connect the shaders by combining them into a program
	shader_program = glCreateProgram()

	glAttachShader(shader_program, vertex_shader)
	glAttachShader(shader_program, fragment_shader_main)
	
	glLinkProgram(shader_program)
	glUseProgram(shader_program)
	
	###########buffers
	#create vertex buffer
	createVertexBuffer()
	# Create the Vertex Buffer Objects (VBO)
	vbo = createDAtaBuffer(Main.ShadersAndVerticies.vertices)

	# Create the Element Buffer Object (EBO)
	ebo = createElementBuffer(Main.ShadersAndVerticies.elements)
	############ how data should be read from data buffer
	encodeDataFromDataBuffer()

  GLFW.PollEvents()

     textureLiver = createTexture(0,Int32(40),Int32(40), GL_R8UI)
     textureSecond = createTexture(0,Int32(40),Int32(40), GL_R8UI)
    
  
    
     aa=zeros(UInt8,40,40)
     bb=zeros(UInt8,40,40)

     aa[2,1]=1

     bb[1,1]=1

     livSamplerRef=  glGetUniformLocation(shader_program, "mainLab")
     secSamplerRef= glGetUniformLocation(shader_program, "testLab1")

     glUniform1i(livSamplerRef, 0) # read from active texture 0
     glUniform1i(secSamplerRef, 1) # read from active texture 1
     
     # load texture


     
     glActiveTexture(GL_TEXTURE0)
     glBindTexture(GL_TEXTURE_2D, textureLiver[])
     glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,aa)
     basicRender(window)

     glBindTexture(GL_TEXTURE_2D, textureSecond[])
     glActiveTexture(GL_TEXTURE1)
     glBindTexture(GL_TEXTURE_2D, textureSecond[])
     glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,bb)
     basicRender(window)



    #  glActiveTexture(GL_TEXTURE0)
    #  glBindTexture(GL_TEXTURE_2D, textureLiver[])
    #  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,aa)
    #  basicRender(window)

    #  glBindTexture(GL_TEXTURE_2D, textureSecond[])
    #  glActiveTexture(GL_TEXTURE1)
    #  glBindTexture(GL_TEXTURE_2D, textureSecond[])
    #  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,bb)
    #  basicRender(window)




using Glutils
     @uniforms! begin

    glGetUniformLocation(shader_program, "mainLabColorMask"):= Cfloat[1.0, 0.0, 0.0, 0.8]
     glGetUniformLocation(shader_program, "testLab1ColorMask"):= Cfloat[0.0, 1.0, 0.0, 0.8]
   
     glGetUniformLocation(shader_program, "mainLabisVisible"):= 1
     glGetUniformLocation(shader_program, "testLab1isVisible"):= 1
    end


     glActiveTexture(GL_TEXTURE0); # active proper texture unit before binding
     glBindTexture(GL_TEXTURE_2D, textureLiver[])
    # glUniform1i(samplerRefNumb,index);# we first look for uniform sampler in shader  
     glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,aa)
     basicRender(window)


     glActiveTexture(GL_TEXTURE1); # active proper texture unit before binding
     glBindTexture(GL_TEXTURE_2D, textureSecond[])
    # glUniform1i(samplerRefNumb,index);# we first look for uniform sampler in shader  
     glTexSubImage2D(GL_TEXTURE_2D,0,0,0,40,40, GL_RED_INTEGER,GL_UNSIGNED_BYTE,bb)
     basicRender(window)