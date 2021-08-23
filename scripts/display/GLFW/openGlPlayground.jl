

     using DrWatson
     @quickactivate "Julia Med 3d"
     include(DrWatson.scriptsdir("display","GLFW","includeAll.jl"))

     using Main.ModernGlUtil, Match, Parameters,DataTypesBasic,Main.ShadersAndVerticies,ModernGL, GeometryTypes, GLFW, Main.ForDisplayStructs,ColorTypes, Dictionaries,Main.DisplayWords, Setfield
     using Main.CustomFragShad, Main.PrepareWindowHelpers, Main.ReactToScroll, Main.SegmentationDisplay,Main.Uniforms, Main.OpenGLDisplayUtils
     using FreeTypeAbstraction,Rocket ,GLFW , Main.DataStructs, Main.StructsManag,Main.TextureManag, Main.SegmentationDisplay
 using Main.PrepareWindow,Glutils, Main.ForDisplayStructs, Dictionaries, Parameters, ColorTypes

     listOfTexturesToCreate = [
     TextureSpec{UInt8}(
         name = "mainLab",
         strokeWidth = 5,
         color = RGB(1.0,0.0,0.0)
         ,minAndMaxValue= UInt8.([0,1])
      ,isEditable = true
        ),
     TextureSpec{UInt8}(
         name = "testLab1",
         numb= Int32(1),
         color = RGB(0.0,1.0,0.0)
         ,minAndMaxValue= UInt8.([0,1])
         ,isEditable = true
        ),
        TextureSpec{UInt8}(
         name = "testLab2",
         numb= Int32(2),
         color = RGB(0.0,0.0,1.0)
         ,minAndMaxValue= UInt8.([0,1])
         ,isEditable = true
          ),
         TextureSpec{Float32}(
           name = "nuclearMaskking",
           isNuclearMask= true,
           isContinuusMask= true,
           colorSet = [RGB(0.0,0.0,1.0),RGB(1.0,0.0,0.0)]
           ,minAndMaxValue= Float32.([0.0,2.0])
           ,isEditable = true
         ),
        TextureSpec{Int16}(
         name= "CTIm",
         numb= Int32(3),
         isMainImage = true,
         minAndMaxValue= Int16.([0,100]))  
  ];
  #   
  fractionOfMainIm= Float32(0.8);
  heightToWithRatio=Float32(0.5);

  texureDepth =40;
  textureHeight = 40;
  textureWidth = 40;

  datToScrollDims= DataToScrollDims(imageSize= (texureDepth,textureWidth,textureHeight),voxelSize= (1.0,1.0,0.3), dimensionToScroll = 3 )

  Main.SegmentationDisplay.coordinateDisplay(listOfTexturesToCreate ,fractionOfMainIm ,datToScrollDims ,1000)
   




#  mainMaskDummy = UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
#  testLab1Dat =UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))
#  testLab2Dat= UInt8.(map(xx-> (xx >0 ? 1 : 0), rand(Int8,10,40,40)))

 mainMaskDummy =  zeros(UInt8,texureDepth,textureWidth,textureHeight);
 testLab1Dat = zeros(UInt8,texureDepth,textureWidth,textureHeight);
 testLab2Dat=  zeros(UInt8,texureDepth,textureWidth,textureHeight);
 nuclearMaskDat =abs.(rand(Float32,texureDepth,textureWidth,textureHeight));    
 nuclearMaskDat = nuclearMaskDat./maximum(nuclearMaskDat)   ;
 ctDummy =  Int16.(map(xx-> (xx >0 ? 1 : 0), rand(Int16,texureDepth,textureWidth,textureHeight)));# will give white background for testing 


 typeof(size(mainMaskDummy))



   slicesDat=  [ThreeDimRawDat{UInt8}(UInt8,"mainLab",mainMaskDummy)
     ,ThreeDimRawDat{Int16}(Int16,"CTIm",ctDummy)
     ,ThreeDimRawDat{UInt8}(UInt8,"testLab2",testLab2Dat)
     ,ThreeDimRawDat{UInt8}(UInt8,"testLab1",testLab1Dat)  
     ,ThreeDimRawDat{Float32}(Float32,"nuclearMaskking",nuclearMaskDat)  
     
     ];

     mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
     supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:texureDepth );



     singleSliceDat = [
      TwoDimRawDat{UInt8}(UInt8,"mainLab", UInt8.(map(xx-> (xx >0 ? 1 : 0), zeros(Int8,textureWidth,textureHeight))))
     ,TwoDimRawDat{Int16}(Int16,"CTIm",Int16.(map(xx-> (xx >0 ? 1 : 0), zeros(Int16,textureWidth,textureHeight))))
     ,TwoDimRawDat{UInt8}(UInt8,"testLab2",UInt8.(map(xx-> (xx >0 ? 1 : 0), zeros(Int8,textureWidth,textureHeight))))
     ,TwoDimRawDat{UInt8}(UInt8,"testLab1",UInt8.(map(xx-> (xx >0 ? 1 : 0), zeros(Int8,textureWidth,textureHeight)))) 
     ,TwoDimRawDat{Float32}(Float32,"nuclearMaskking", nuclearMaskDat[1,:,:]) 
    ];
    sislines= textLinesFromStrings(["asd Line1", "as Line 2", "main Line 2", "main Line 2", "main Line 2",  "uuuuuuuuuuuuuuuuuu"]);


    exampleSingleSliceDat = SingleSliceDat(listOfDataAndImageNames=singleSliceDat
                                            ,textToDisp=sislines);

    Main.SegmentationDisplay.updateSingleImagesDisplayed(exampleSingleSliceDat);





     mainScrollDat = FullScrollableDat(dataToScrollDims =DataToScrollDims(imageSize= (texureDepth,textureWidth,textureHeight),voxelSize= (1.0,1.0,0.3), dimensionToScroll = 3 )
                                      ,dimensionToScroll=1
                                      ,dataToScroll= slicesDat
                                      ,mainTextToDisp= mainLines
                                      ,sliceTextToDisp=supplLines );


    Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);

 

    GLFW.PollEvents()

 
         textLiverMain = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[1];

         textureB = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[2];
         textureC = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[3];
         nuclearTexture = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[4];
         textureCt = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications[5];
         
         Main.SegmentationDisplay.mainActor.actor.textureToModifyVec= [textLiverMain];
         
         window = Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.window;

         dispObj= Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects;
         wordsDispObj= Main.SegmentationDisplay.mainActor.actor.textDispObj;
         stopList = dispObj.stopListening[];
         calcDim = Main.SegmentationDisplay.mainActor.actor.calcDimsStruct;

         actor =  Main.SegmentationDisplay.mainActor.actor;
         length(  actor.forUndoVector)


         C:\Users\1\Documents\GitHub\Probabilistic-medical-segmentation\.vscode

         using DocumenterTools
         DocumenterTools.generate()

         popfirst!(actor.forUndoVector)


         arr = Vector{Function}()
         push!(arr,()->print("asd "))
pop!(actor.forUndoVector)()


         function aa()
           print("asd ")
        end 

        import  FunctionWrappers
 
        f1 = @inferred F64AnyFunc(identity)

        f1 = FunctionWrappers.F64F64Func(sin)
        f2 = @inferred F64F64Func(f1)

         addToforUndoVector(actor,()-> print("asd "))
         
 
         actor.lastRecordedMousePosition

        setTextureVisibility(false,textLiverMain.uniforms )
        setTextureVisibility(false,textureB.uniforms )
        setTextureVisibility(false,textureC.uniforms )
        setTextureVisibility(false,textureCt.uniforms )
        
        setTextureVisibility(false,nuclearTexture.uniforms )
        
        basicRender(window)


        setTextureVisibility(true,nuclearTexture.uniforms )

        basicRender(window)



        setTextureVisibility(true,textLiverMain.uniforms )
        setTextureVisibility(true,textureB.uniforms )
        setTextureVisibility(true,textureC.uniforms )
        
        setTextureVisibility(true,nuclearTexture.uniforms )

        basicRender(window)


     dat =ones(10,11,12)

     
     dataToScrollDims= DataToScrollDims(imageSize= (10,11,12),voxelSize= (1.0,1.0,0.3), dimensionToScroll = 2 )
    
     cartTwoToThree(dataToScrollDims,Int32(5),CartesianIndex(1,2))

```@doc
Based on DataToScrollDims it will enrich passed CalcDimsStruct texture width, height and  heightToWithRatio
based on data passed from DataToScrollDims
```

ress= getHeightToWidthRatio(calcDim,dataToScrollDims)

arr= [0x00000001, 0x00000002, 0x00000003, 0x00000004, 0x00000005]
glDeleteTextures(length(arr), arr)






1+1


add BenchmarkTools ,ColorTypes , Conda  , DataTypesBasic  , Dictionaries  , Distributed  , Documenter , DocumenterTools , DrWatson , FreeType  , FreeTypeAbstraction  , GLFW  ,  HDF5  , Match , ModernGL , Observables  , Parameters , PyCall , Revise , Rocket  , Setfield  


















#getHeightToWidthRatio
  #    listOfTextSpecs= map(x->setproperties(x[2],(whichCreated=x[1])),enumerate(listOfTexturesToCreate))

  #    println(createcontextinfo())
  #     gslsStr = get_glsl_version_string()

  #       fsh = """
  #       $(gslsStr)
  #       $(createCustomFramgentShader(listOfTextSpecs,textLiverMain,textureB))  
  #       """
  #      fragShader= createShader(fsh, GL_FRAGMENT_SHADER)
      

  # GLFW.PollEvents()



  #        for st in split(fsh, "\n")
  #        @info st
  #        end
     
function aaa()
  @info "aa"
end

function bbb()
  @info "bbb"
end

dd = [ aaa,bbb]

for func in dd
  func()
end  

typeof(dd)

         dispObj.stopListening[]= true
         maskA = textureB
         maskB = textureC
  vertex_shader = dispObj.vertex_shader
     
         println(createcontextinfo())
         gslsStr = get_glsl_version_string()
         listOfTextSpecsc=  Main.SegmentationDisplay.mainActor.actor.mainForDisplayObjects.listOfTextSpecifications

         fragment_shade,shader_prog= createAndInitShaderProgram(dispObj.vertex_shader,  listOfTextSpecsc,maskA,maskB,gslsStr)
         activateTextures(listOfTextSpecsc )


newForDisp = setproperties(dispObj,(shader_program=shader_prog,fragment_shader=fragment_shade ) )

actor.mainForDisplayObjects=newForDisp


        #  glBindBuffer(GL_ARRAY_BUFFER, dispObj.vbo[])
        #  glBufferData(GL_ARRAY_BUFFER, calcDim.mainQuadVertSize  ,calcDim.mainImageQuadVert , GL_STATIC_DRAW)
        #       encodeDataFromDataBuffer()
reactivateMainObj(shader_prog, newForDisp.vbo,actor.calcDimsStruct  )
activateTextures(listOfTextSpecsc )

# basicRender(window)


        # glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
        # # Swap front and back buffers
        # GLFW.SwapBuffers(window)
      

         @uniforms! begin
         dispObj.mainImageUniforms.isMaskDiffrenceVis:=1
                end
      #  basicRender(window)


      #  setTextureVisibility(true,textureB.uniforms )
      #  setTextureVisibility(true,textLiverMain.uniforms )
      #  basicRender(window)
       setTextureVisibility(true,textLiverMain.uniforms )
       setTextureVisibility(true,textureB.uniforms )
       
       setTextureVisibility(true,nuclearTexture.uniforms )
       basicRender(window)


       dispObj.stopListening[]= false




  # continuusColorTextSpecs= listOfTexturesToCreate
  # tuples= map(x->[ (x.name,[ [a.r,a.g,a.b] for a in x.colorSet  ],"r" ,1),(x.name,[[a.r,a.g,a.b] for a in x.colorSet  ],"g",2),(x.name,[[a.r,a.g,a.b] for a in x.colorSet  ],"b",3)],continuusColorTextSpecs)|> list-> reduce(vcat,list)
  # tuples[1]
  # getNuclearMaskFunctions(continuusColorTextSpecs)

 
                # handler = KeyboardCallbackSubscribable(false,false,false,false,["a1"], Subject(KeyboardStruct, scheduler = AsyncScheduler()))


# scancode =GLFW.KEY_RIGHT_CONTROL
# act=1

# scCode = @match scancode begin
#   GLFW.KEY_RIGHT_CONTROL=> (handler.isCtrlPressed= (act==1); "ctrl" )
#   GLFW.KEY_LEFT_CONTROL => (handler.isCtrlPressed= (act==1); "ctrl")
#   GLFW.KEY_LEFT_SHIFT =>( handler.isShiftPressed= (act==1); "shift")
#   GLFW.KEY_RIGHT_SHIFT =>( handler.isShiftPressed=( act==1); "shift")
#   GLFW.KEY_RIGHT_ALT =>( handler.isAltPressed= (act==1); "alt")
#   GLFW.KEY_LEFT_ALT => (handler.isAltPressed= (act==1); "alt")
#   GLFW.KEY_ENTER =>( handler.isEnterPressed= (act==1); "enter")
#   _ => "notImp" # not Important
# end
#   res = KeyboardStruct(isCtrlPressed=handler.isCtrlPressed || scCode=="ctrl" 
#           , isShiftPressed= handler.isShiftPressed ||scCode=="shift" 
#           ,isAltPressed= handler.isAltPressed ||scCode=="alt"
#           ,isEnterPressed= handler.isEnterPressed 
#           ,lastKeysPressed= handler.lastKeysPressed 
#           ,mostRecentScanCode = scancode
#           ,mostRecentKeyName = "" # just marking it as empty
#           ) 
  
# pp = strToNumber(res.lastKeysPressed)
# processKeysInfo(pp,actor,KeyboardStruct(lastKeysPressed= ["a", "1"]))


#  join(res.lastKeysPressed)



















#   for tpair in enumerate(listOfTexturesToCreate)
#     tpair[2].whichCreated = tpair[1]
#   end#for




#         #  dispObj.stopListening[]= true
#         #  glClearColor(0.0, 0.0, 0.1 , 1.0)
#         dispObj.stopListening[]= true
  activateForTextDisp( wordsDispObj.shader_program_words , wordsDispObj.vbo_words,calcDim)

  face = wordsDispObj.fontFace
 


    lineTextureWidth = 2000
    lineTextureHeight = 2000
    
    dispObj.stopListening[]= true
        #data = ones(UInt8,calcDim.imageTextureHeight,calcDim.imageTextureWidth)
    data = zeros(UInt8,lineTextureWidth,lineTextureHeight)
      
    # render a string into an existing matrix
    a = renderstring!(zeros(UInt8,lineTextureWidth,lineTextureHeight), "Line 1 score", face,  110, 110, 110,valign = :vtop, halign = :hleft)
    b = renderstring!(zeros(UInt8,lineTextureWidth,lineTextureHeight), "Line 2 score", face,  110, 110, 110,valign = :vtop, halign = :hleft)
    b = vcat(a[1:200,:] ,b[1:200,:] ) 
    b =collect(transpose(reverse(b; dims=(1))))
        
    #updateTexture(UInt8,b,wordsDispObj.textureSpec,0,0,Int32(lineTextureWidth),Int32(lineTextureHeight )) #  ,Int32(10000),Int32(1000)
    updateTexture(UInt8,b,wordsDispObj.textureSpec,0,7600,Int32(2000),Int32(400)) #  ,Int32(10000),Int32(1000)

    basicRender(window)
    dispObj.stopListening[]= false


#     strToNumber("11sfdsdf")
  


#     lineTextureWidth = 2000
#     lineTextureHeight = 2000
    

#     struct1 = SimpleLineTextStruct(text= "testing line 1",fontSize= 110,extraLineSpace=1  )
#     struct2 = SimpleLineTextStruct(text= "testing line 2",fontSize= 110,extraLineSpace=1  )
#     struct3 = SimpleLineTextStruct(text= "testing line 3",fontSize= 110,extraLineSpace=1  )
#     struct4 = SimpleLineTextStruct(text= "testing line 1",fontSize= 110,extraLineSpace=1  )
#     struct4 = SimpleLineTextStruct(text= "testing line 2",fontSize= 110,extraLineSpace=1  )
#     struct5 = SimpleLineTextStruct(text= "testing line 3",fontSize= 110,extraLineSpace=1  )
#     struct6 = SimpleLineTextStruct(text= "testing line 1",fontSize= 110,extraLineSpace=1  )
#     struct7 = SimpleLineTextStruct(text= "testing line 2",fontSize= 110,extraLineSpace=1  )
#     struct8 = SimpleLineTextStruct(text= "testing line 3",fontSize= 110,extraLineSpace=1  )
#     struct9 = SimpleLineTextStruct(text= "testing line 1",fontSize= 110,extraLineSpace=1  )
#     struct10 = SimpleLineTextStruct(text= "testing line 2",fontSize= 110,extraLineSpace=1  )
#     struct11 = SimpleLineTextStruct(text= "testing line 3",fontSize= 110,extraLineSpace=1  )

#     strList = [struct1,struct2,struct3,struct4,struct5,struct6,struct7,struct6,struct7,struct7,struct4,struct5,struct6,struct7,struct6,struct7]
#     #,struct5,struct6,struct7,struct8,struct9,struct10,struct11
    



# #    res =  hcat(renderSingleLineOfText(struct1,lineTextureWidth, face )
# #   ,renderSingleLineOfText(struct2,lineTextureWidth, face )
# #   ,renderSingleLineOfText(struct3,lineTextureWidth, face )
# #     )
# #     maximum(res)
# # sz = size(res)

# # # updateTexture(UInt8,b,wordsDispObj.textureSpec,0,7600,Int32(size(res)[1]),Int32(size(res)[2])) #  ,Int32(10000),Int32(1000)
# #     updateTexture(UInt8,res,wordsDispObj.textureSpec,0,2000,Int32(sz[1]),Int32(sz[2])) #  ,Int32(10000),Int32(1000)
# dispObj.stopListening[]= true
# strListB= textLinesFromStrings(["asasfkajshalkjdhs", "3w7gqaw76dgabs89y3p8w", "ahsd78oy3o821hbf", "9823bv67asfasdlasjpdaus"])
#   d=   addTextToTextureB(wordsDispObj,strListB, calcDim )

#     basicRender(window)
#     dispObj.stopListening[]= false









#     dispObj.stopListening[]= true

#     updateTexture(UInt8, zeros(UInt8,2000,8000),wordsDispObj.textureSpec) #  ,Int32(10000),Int32(1000)
#    basicRender(window)
#     dispObj.stopListening[]= false


#     GLFW.PollEvents()

#  reactivateMainObj(dispObj.shader_program ,dispObj.vbo,calcDim  )



# Main.SegmentationDisplay.updateSingleImagesDisplayed(exampleSingleSliceDat)
# basicRender(window)



#     juliaDataType= UInt8
#     toSend = ones(UInt8,2000,8000)
#     toSendFlat = reduce(vcat,toSend)
#     pboIds= [Ref(GLuint(0))] 
#     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[1][]);
# #    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(toSend), 0, GL_STREAM_DRAW);
#     glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(toSendFlat), Ptr{juliaDataType}(), GL_STREAM_DRAW);

#     # bind the texture and PBO
#     textureId =wordsDispObj.textureSpec.ID
#     glActiveTexture(wordsDispObj.textureSpec.actTextrureNumb); # active proper texture unit before binding
#     glBindTexture(GL_TEXTURE_2D, textureId[]);
#     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[1][]);
   
#     # update data directly on the mapped buffer - this is internal function implemented below
#     ptrB = Ptr{juliaDataType}(glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY))
#     for i=1:length(toSend)
#         unsafe_store!(ptrB, toSend[i], i)
#     end

#     basicRender(window)
 


#     // copy pixels from PBO to texture object
#     // Use offset instead of ponter.
#     glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, PIXEL_FORMAT, GL_UNSIGNED_BYTE, 0);


# ######### pixel buffer objects test



# DATA_SIZE = 8 * sizeof(juliaDataTyp) *width * height  # number of bytes our image will have so in 2D it will be width times height times number of bytes needed for used datatype we need to multiply by 8 becouse sizeof() return bytes instead of bits
# pbo = Ref(GLuint(pboNumber))  
# glGenBuffers(1, pbo)


# # glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[i]);
# # glBufferData(GL_PIXEL_PACK_BUFFER, nbytes, NULL, GL_STREAM_READ);

# # glReadPixels(0, 0, width, height, fmt, GL_UNSIGNED_BYTE, 0);   # When a GL_PIXEL_PACK_BUFFER is bound, the last 0 is used as offset into the buffer to read into. */




# glBindTexture(GL_TEXTURE_2D,textureId[]); 
# # copy pixels from PBO to texture object
# # Use offset instead of pointer.
# # glTexSubImage2D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, GLsizei(width), GLsizei(height),  GL_RED_INTEGER, GL_SHORT, Ptr{juliaDataTyp}());

# glTexSubImage2D(GL_TEXTURE_2D,0,0,0, width, height, GL_RED_INTEGER, subImageDataType, Ptr{juliaDataType}());


# # bind the PBO
# glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID[]);


# # Note that glMapBuffer() causes sync issue.
# # If GPU is working with this buffer, glMapBuffer() will wait(stall)
# # until GPU to finish its job. To avoid waiting (idle), you can call
# # first glBufferData() with NULL pointer before glMapBuffer().
# # If you do that, the previous data in PBO will be discarded and
# # glMapBuffer() returns a new allocated pointer immediately
# # even if GPU is still working with the previous data.
# glBufferData(GL_PIXEL_UNPACK_BUFFER, DATA_SIZE, Ptr{juliaDataType}(), GL_STREAM_DRAW);

# # map the buffer object into client's memory
# glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY)

 
# ptr = Ptr{juliaDataType}(glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY))
# # update data directly on the mapped buffer - this is internal function implemented below

# updatePixels(ptr,data,length(data));

# glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); # release the mapped buffer

# # it is good idea to release PBOs with ID 0 after use.
# # Once bound with 0, all pixel operations are back to normal ways.
# glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);











#     matrlist = map(x-> renderSingleLineOfText(x,Int32(2000),face) ,strList) 
#     matr=  reduce( hcat  ,matrlist)




# ##

  # reactivateMainObj(dispObj.shader_program ,dispObj.vbo,calcDim  )



# Main.SegmentationDisplay.updateSingleImagesDisplayed(exampleSingleSliceDat)
# basicRender(window)

# GLFW.PollEvents()




# fractionOfMainIm= Float32(0.5)
# heightToWithRatio=Float32(2)
# width = 1000
# height = 500















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
     
    
     
    