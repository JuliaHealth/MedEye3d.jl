
"""
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
module coordinating response to the  keyboard input - mainly shortcuts that  helps controlling  active/visible texture2D
"""
#@doc ReactOnKeyboardSTR
module ReactOnKeyboard
using ModernGL,Main.DisplayWords,Main.StructsManag, Setfield,Main.PrepareWindow,  Main.DataStructs ,Glutils, Rocket, GLFW,Dictionaries, Main.ForDisplayStructs,Main.TextureManag, Main.OpenGLDisplayUtils, Main.Uniforms, Match, Parameters,DataTypesBasic
export reactToKeyboard , registerKeyboardFunctions,processKeysInfo

KeyboardCallbackSubscribableStr= """
Object that enables managing input from keyboard - it stores the information also about
needed keys wheather they are kept pressed  
examples of keyboard input 
    action RELEASE GLFW.Action
    key s StringPRESS
    key s String
    action PRESS GLFW.Action
    key s StringRELEASE
    key s String
    action RELEASE GLFW.Action

"""
mutable struct KeyboardCallbackSubscribable <: Subscribable{KeyboardStruct}
# true when pressed and kept true until released
# true if corresponding keys are kept pressed and become flase when relesed
    isCtrlPressed::Bool # left - scancode 37 right 105 - Int32
    isShiftPressed::Bool  # left - scancode 50 right 62- Int32
    isAltPressed::Bool# left - scancode 64 right 108- Int32
    isEnterPressed::Bool# scancode 36
    isTAbPressed::Bool# scancode 36
    isSpacePressed::Bool# scancode 36
    isF1Pressed::Bool
    isF2Pressed::Bool
    isF3Pressed::Bool
    lastKeysPressed::Vector{String} # last pressed keys - it listenes to keys only if ctrl/shift or alt is pressed- it clears when we release those case or when we press enter
    subject :: Subject{KeyboardStruct} 
end 

"""
will "tell" what functions should be invoked in order to process keyboard input 
"""
function Rocket.on_subscribe!(handler::KeyboardCallbackSubscribable, actor::SyncActor{Any, ActorWithOpenGlObjects})
    return subscribe!(handler.subject, actor)
end


"""
given pressed keys lik 1-9 and all letters resulting key is encoded as string and will be passed here
handler object responsible for capturing action 
str - name of key lik 1,5 f,.j ... but not ctrl shift etc
action - for example key press or release
scancode - if key do not have short name like ctrl ... it has scancode
"""
function (handler::KeyboardCallbackSubscribable)(str::String, action::GLFW.Action)

    if( (action==instances(GLFW.Action)[2])  ) 
        push!(handler.lastKeysPressed ,str)
   end#if
end #handler

GLFW.PRESS

function (handler::KeyboardCallbackSubscribable)(scancode ::GLFW.Key, action::GLFW.Action)
    #1 pressed , 2 released -1 sth else
    act =  @match action begin
        instances(GLFW.Action)[2] => 1
        instances(GLFW.Action)[1] => 2
        _ => -1
    end

   if(act>0)# so we have press or relese
       
         scCode = @match scancode begin
            GLFW.KEY_RIGHT_CONTROL=> (handler.isCtrlPressed= (act==1); "ctrl" )
            GLFW.KEY_LEFT_CONTROL => (handler.isCtrlPressed= (act==1); "ctrl")
            GLFW.KEY_LEFT_SHIFT =>( handler.isShiftPressed= (act==1); "shift")
            GLFW.KEY_RIGHT_SHIFT =>( handler.isShiftPressed=( act==1); "shift")
            GLFW.KEY_RIGHT_ALT =>( handler.isAltPressed= (act==1); "alt")
            GLFW.KEY_LEFT_ALT => (handler.isAltPressed= (act==1); "alt")
            GLFW.KEY_SPACE => (handler.isSpacePressed= (act==1); "space")
            GLFW.KEY_TAB => (handler.isTAbPressed= (act==1); "tab")
            GLFW.KEY_ENTER =>( handler.isEnterPressed= (act==1); "enter")
            GLFW.KEY_F1 =>( handler.isEnterPressed= (act==1); "f1")
            GLFW.KEY_F2 =>( handler.isEnterPressed= (act==1); "f2")
            GLFW.KEY_F3 =>( handler.isEnterPressed= (act==1); "f3")
            _ => "notImp" # not Important

    
            

         end
            res = KeyboardStruct(isCtrlPressed=handler.isCtrlPressed || scCode=="ctrl" 
                    , isShiftPressed= handler.isShiftPressed ||scCode=="shift" 
                    ,isAltPressed= handler.isAltPressed ||scCode=="alt"
                    ,isSpacePressed= handler.isSpacePressed ||scCode=="space"
                    ,isTAbPressed= handler.isTAbPressed ||scCode=="tab"
                    ,isF1Pressed= handler.isF1Pressed ||scCode=="f1"
                    ,isF2Pressed= handler.isF2Pressed ||scCode=="f2"
                    ,isF3Pressed= handler.isF3Pressed ||scCode=="f3"
                    ,isEnterPressed= handler.isEnterPressed 
                    ,lastKeysPressed= handler.lastKeysPressed 
                    ,mostRecentScanCode = scancode
                    ,mostRecentKeyName = "" # just marking it as empty
                    ,mostRecentAction = action) 
            

            if(shouldBeExecuted(res,act))
                next!(handler.subject, res ) 
                handler.lastKeysPressed=[] 

            end#if 

    end#if    
  

end #second handler



"""
registering functions to the GLFW
window - GLFW window with Visualization
stopListening - atomic boolean enabling unlocking GLFW context
"""
function registerKeyboardFunctions(window::GLFW.Window,stopListening::Base.Threads.Atomic{Bool}    )

    stopListening[]=true # stoping event listening loop to free the GLFW context
                           
    keyboardSubs = KeyboardCallbackSubscribable(false,false,false,false,false,false,false,false,false,[], Subject(KeyboardStruct, scheduler = AsyncScheduler()))
                                  
        GLFW.SetKeyCallback(window, (_, key, scancode, action, mods) -> begin
        name = GLFW.GetKeyName(key, scancode)
        if name == nothing
            keyboardSubs(key,action)                                                        
        else
            keyboardSubs(name,action)
        end
        end)

   stopListening[]=false # reactivate event listening loop

return keyboardSubs

end #registerKeyboardFunctions




"""
processing information from keys - the instance of this function will be chosen on
the basis mainly of multiple dispatch
"""
function processKeysInfo(textSpecObs::Identity{TextureSpec{T}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct )where T
    textSpec =  textSpecObs.value
    if(keyInfo.isCtrlPressed)    
        setVisAndRender(false,actor.actor,textSpec.uniforms )
        @info " set visibility of $(textSpec.name) to false" 
        #to enabling undoing it 
        addToforUndoVector(actor, ()-> setVisAndRender(true,actor.actor,textSpec.uniforms )    )
    elseif(keyInfo.isShiftPressed)  
        setVisAndRender(true,actor.actor,textSpec.uniforms )
        @info " set visibility of $(textSpec.name) to true" 
       #to enabling undoing it 
       addToforUndoVector(actor, ()->setVisAndRender(false,actor.actor,textSpec.uniforms )   )

    elseif(keyInfo.isAltPressed)  
        oldTex = actor.actor.textureToModifyVec
        actor.actor.textureToModifyVec= [textSpec]
        @info " set texture for manual modifications to  $(textSpec.name)"
       if(!isempty(oldTex))
       addToforUndoVector(actor, ()->begin  @info actor.actor.textureToModifyVec=[oldTex[1]] end)
       end
    end #if
end #processKeysInfo

"""
sets  visibility and render the result to the screen
"""
function setVisAndRender(isVis::Bool,actor::ActorWithOpenGlObjects,unifs::TextureUniforms )
    setTextureVisibility(isVis,unifs )
    basicRender(actor.mainForDisplayObjects.window)

end#setVisAndRender

"""
for case when we want to subtract two masks
"""
function processKeysInfo(maskNumbs::Identity{Tuple{Identity{TextureSpec{T}}, Identity{TextureSpec{T}}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T
    textSpecs =  maskNumbs.value
    maskA = textSpecs[1].value
    maskB = textSpecs[2].value
    
    if(keyInfo.isCtrlPressed)    # when we want to stop displaying diffrence
        undoDiffrence(actor,maskA,maskB )
        addToforUndoVector(actor, ()->displayMaskDiffrence(maskA,maskB,actor ) )

    elseif(keyInfo.isShiftPressed)  # when we want to display diffrence
        displayMaskDiffrence(maskA,maskB,actor )
        addToforUndoVector(actor, ()-> undoDiffrence(actor,maskA,maskB ))

    end #if
   

end#processKeysInfo
"""
for case  we want to undo subtracting two masks
"""
function undoDiffrence(actor::SyncActor{Any, ActorWithOpenGlObjects},maskA,maskB )
    @uniforms! begin
    actor.actor.mainForDisplayObjects.mainImageUniforms.isMaskDiffrenceVis:=0
           end
setTextureVisibility(true,maskA.uniforms )
setTextureVisibility(true,maskB.uniforms )
basicRender(actor.actor.mainForDisplayObjects.window)

end#undoDiffrence




"""
in case we want to  get new number set for manual modifications
    toBeSavedForBack - just marks weather we wat to save the info how to undo latest action
    - false if we invoke it from undoing 
"""
function processKeysInfo(numbb::Identity{Int64}
                        ,actor::SyncActor{Any, ActorWithOpenGlObjects}
                        ,keyInfo::KeyboardStruct 
                        ,toBeSavedForBack::Bool = true) where T

    valueForMasToSett = valueForMasToSetStruct(value = numbb.value)
    old = actor.actor.valueForMasToSet.value
    actor.actor.valueForMasToSet =valueForMasToSett

    updateImagesDisplayed(actor.actor.currentlyDispDat
    , actor.actor.mainForDisplayObjects
    , actor.actor.textDispObj
    , actor.actor.calcDimsStruct ,valueForMasToSett )
# for undoing action
if(toBeSavedForBack)
    addToforUndoVector(actor, ()-> processKeysInfo( Option(old),actor, keyInfo,false ))
end

end#processKeysInfo



"""
In order to enable undoing last action we just invoke last function from list 
"""
function processKeysInfo(numbb::Identity{Bool},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T
    if(!isempty(actor.actor.forUndoVector))
     pop!(actor.actor.forUndoVector)()
    end
end#processKeysInfo



"""
when shift plus will be pressed it will increase stroke width
when shift minus will be pressed it will increase stroke width
"""
function processKeysInfo(annot::Identity{AnnotationStruct}
    ,actor::SyncActor{Any, ActorWithOpenGlObjects}
    ,keyInfo::KeyboardStruct 
    ,toBeSavedForBack::Bool = true) where T

    textureList = actor.actor.textureToModifyVec
    if (!isempty(textureList))
        texture= textureList[1]
        oldsWidth = texture.strokeWidth
        texture.strokeWidth= oldsWidth+=annot.value.strokeWidthChange
        # for undoing action
        if(toBeSavedForBack)
        addToforUndoVector(actor, ()-> processKeysInfo( Option(AnnotationStruct(oldsWidth)),actor, keyInfo,false ))
        end#if
    end#if

end#processKeysInfo


"""
KEY_F1 - will display wide window for bone Int32(1000),Int32(-1000)
KEY_F1 - will display window for soft tissues Int32(400),Int32(-200)
KEY_F1 - will display wide window for lung viewing  Int32(0),Int32(-1000)
"""
function processKeysInfo(wind::Identity{WindowControlStruct}
    ,actor::SyncActor{Any, ActorWithOpenGlObjects}
    ,keyInfo::KeyboardStruct 
    ,toBeSavedForBack::Bool = true) where T
    #we have some predefined windows


    windowStruct =  @match wind.value.letterCode begin
    "F1" => WindowControlStruct("F1",Int32(1000), Int32(-1000) )
    "F2" => WindowControlStruct("F2",Int32(400), Int32(-200) )
    "F3" => WindowControlStruct("F3",Int32(0), Int32(-1000) )
    _ => wind.value
        end

    #updating current windowing object and getting reference to old
    old = actor.actor.mainForDisplayObjects.windowControlStruct 
    actor.actor.mainForDisplayObjects= setproperties(actor.actor.mainForDisplayObjects,(windowControlStruct =windowStruct))

    # setting window and showig it 
    setCTWindow(windowStruct.min_shown_white,windowStruct.max_shown_black, actor.actor.mainForDisplayObjects.mainImageUniforms)
    basicRender(actor.actor.mainForDisplayObjects.window)
   
       # for undoing action
    if(toBeSavedForBack)
         addToforUndoVector(actor, ()-> processKeysInfo( Option(old),actor, keyInfo,false ))
    end

end#processKeysInfo







"""
In case we want to change the dimansion of scrolling so for example from transverse 
    toBeSavedForBack - just marks weather we wat to save the info how to undo latest action
    - false if we invoke it from undoing 
"""

function processKeysInfo(toScrollDatPrim::Identity{DataToScrollDims}
                    ,actor::SyncActor{Any, ActorWithOpenGlObjects}
                    ,keyInfo::KeyboardStruct
                    ,toBeSavedForBack::Bool = true ) where T
    toScrollDat= toScrollDatPrim.value

    old = actor.actor.onScrollData.dimensionToScroll

    newCalcDim= getHeightToWidthRatio(actor.actor.calcDimsStruct,toScrollDat )|>
                    getMainVerticies
     actor.actor.calcDimsStruct = newCalcDim
#In order to make the  background black  before we will render quad of possibly diffrent dimensions we will set all to invisible - and obtain black background
textSpecs = actor.actor.mainForDisplayObjects.listOfTextSpecifications

for textSpec in textSpecs
    setTextureVisibility(false,textSpec.uniforms )
end#for    
basicRender(actor.actor.mainForDisplayObjects.window)


    #we need to change textures only if dimensions do not match
  #  if(actor.actor.calcDimsStruct.imageTextureWidth!=newCalcDim.imageTextureWidth  || actor.actor.calcDimsStruct.imageTextureHeight!=newCalcDim.imageTextureHeight )
        # first we need to update information about dimensions etc 


        #next we need to delete all textures and create new ones 

        arr = map(it->it.ID[],textSpecs)

        glDeleteTextures(length(arr), arr)# deleting

        #getting new 
        initializeTextures(textSpecs,newCalcDim)

   # end#if


actor.actor.onScrollData.dimensionToScroll = toScrollDat.dimensionToScroll

actor.actor.onScrollData.slicesNumber = getSlicesNumber(actor.actor.onScrollData)
#getting  the slice of intrest based on last recorded mouse position 

current=actor.actor.lastRecordedMousePosition[toScrollDat.dimensionToScroll]

#displaying all


singleSlDat= actor.actor.onScrollData.dataToScroll|>
(scrDat)-> map(threeDimDat->threeToTwoDimm(threeDimDat.type,Int64(current),toScrollDat.dimensionToScroll,threeDimDat ),scrDat) |>
(twoDimList)-> SingleSliceDat(listOfDataAndImageNames=twoDimList
                            ,sliceNumber=current
                            ,textToDisp = getTextForCurrentSlice(actor.actor.onScrollData, Int32(current))  )

                            # glFinish()
                            # glFlush()
                            # glClearColor(0.0, 0.0, 0.0 , 1.0)
                            # GLFW.SwapBuffers(actor.actor.mainForDisplayObjects.window)

dispObj = actor.actor.mainForDisplayObjects
#for displaying new quad - to accomodate new proportions
reactivateMainObj(dispObj.shader_program, dispObj.vbo,newCalcDim  )

glClear(GL_COLOR_BUFFER_BIT)

updateImagesDisplayed(singleSlDat
                    ,actor.actor.mainForDisplayObjects
                    ,actor.actor.textDispObj
                    ,newCalcDim 
                    ,actor.actor.valueForMasToSet      )





 #@info "singleSlDat" singleSlDat
     #saving information about current slice for future reference
actor.actor.currentDisplayedSlice = current
# to enbling getting back
if(toBeSavedForBack)
    addToforUndoVector(actor, ()-> processKeysInfo( Option(old),actor, keyInfo,false ))
end#if

end#processKeysInfo



processKeysInfo(a::Const{Nothing},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing}, Identity{TextureSpec{T}}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing}, Const{Nothing}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{ Identity{TextureSpec{T}}, Const{Nothing}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T = "" # just doing nothing in case of empty option



"""
SUBTRACTING MASKS
used in order to enable subtracting one mask from the other - hence displaying 
pixels where value of mask a is present but mask b not (order is important)
automatically both masks will be set to be invisible and only the diffrence displayed

In order to achieve this  we need to have all of the samplers references stored in a list 
1) we need to set both masks to invisible - it will be done from outside the shader
2) we set also from outside uniform marking visibility of diffrence to true
3) also from outside we need to set which texture to subtract from which we will achieve this by setting maskAtoSubtr and maskBtoSubtr int uniforms
    those integers will mark which samplers function will use
4) in shader function will be treated as any other mask and will give contribution to output color multiplied by its visibility(0 or 1)    
5) inside the function color will be defined as multiplication of two colors of mask A and mask B - colors will be acessed similarly to samplers
6) color will be returned only if value associated with  maskA is greater than mask B and proportional to this difffrence

In order to provide maximum performance and avoid branching inside shader multiple shader programs will be attached and one choosed  that will use diffrence needed
maskToSubtrastFrom,maskWeAreSubtracting - specifications o textures we are operating on 
"""
function displayMaskDiffrence(maskA::TextureSpec, maskB::TextureSpec,actor::SyncActor{Any, ActorWithOpenGlObjects})
 #defining variables

 dispObj = actor.actor.mainForDisplayObjects
 vertex_shader = dispObj.vertex_shader
 listOfTextSpecsc=  actor.actor.mainForDisplayObjects.listOfTextSpecifications
 fragment_shade,shader_prog= createAndInitShaderProgram(dispObj.vertex_shader,  listOfTextSpecsc,maskA,maskB,dispObj.gslsStr)
 # saving new variables to the actor
 newForDisp = setproperties(dispObj,(shader_program=shader_prog,fragment_shader=fragment_shade ) )
 actor.actor.mainForDisplayObjects=newForDisp
 #making all ready to display
 reactivateMainObj(shader_prog, newForDisp.vbo,actor.actor.calcDimsStruct  )
 activateTextures(listOfTextSpecsc )
 #making diffrence visible
 @uniforms! begin
 dispObj.mainImageUniforms.isMaskDiffrenceVis:=1
        end
 setTextureVisibility(false,maskA.uniforms )
 setTextureVisibility(false,maskB.uniforms )
 basicRender(actor.actor.mainForDisplayObjects.window)

end#displayMaskDiffrence





"""
Given keyInfo struct wit information about pressed keys it can process them to make some actions  - generally activating keyboard shortcuts
shift + number - make mask associated with given number visible
ctrl + number -  make mask associated with given number invisible 
alt + number -  make mask associated with given number active for mouse interaction 
tab + number - sets the number that will be  used as an input to masks modified by mouse
shift + numberA + "-"(minus sign) +numberB  - display diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B invisible
ctrl + numberA + "-"(minus sign) +numberB  - stops displaying diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B visible
space + 1 or 2 or 3 - change the plane of view (transverse, coronal, sagittal)
ctrl + z - undo last action
tab +/- increase or decrease stroke width
F1, F2 ... - swith between defined window display characteristics - like min shown white and mx shown black ...
"""
function reactToKeyboard(keyInfo::KeyboardStruct
                        , actor::SyncActor{Any, ActorWithOpenGlObjects})
                      
    #we got this only when ctrl/shift/als is released or enter is pressed
    obj = actor.actor.mainForDisplayObjects
    obj.stopListening[]=true #free GLFW context

    # processing here on is based on multiple dispatch mainly 
    processKeysInfo(parseString(keyInfo.lastKeysPressed,actor,keyInfo),actor,keyInfo)
    

    obj.stopListening[]=false # reactivete event listening loop

end#reactToKeyboard






"""
return true in case the combination of keys should invoke some action
"""
function shouldBeExecuted(keyInfo::KeyboardStruct, act::Int64)::Bool
    if(act>0)# so we have press or relese 
        res =  @match keyInfo.mostRecentScanCode begin
      GLFW.KEY_RIGHT_CONTROL => return act==2 # returning true if we relese key
      GLFW.KEY_LEFT_CONTROL => return act==2
      GLFW.KEY_LEFT_SHIFT => return act==2
      GLFW.KEY_RIGHT_SHIFT=> return act==2
      GLFW.KEY_RIGHT_ALT => return act==2
      GLFW.KEY_LEFT_ALT => return act==2
      GLFW.KEY_SPACE => return act==2
      GLFW.KEY_TAB => return act==2
      GLFW.KEY_ENTER  => return act==1 # returning true if pressed pressed
      GLFW.KEY_F1  => return act==1 
      GLFW.KEY_F2  => return act==1 
      GLFW.KEY_F3  => return act==1
            _ => false # not Important
         end#match

         return res
 
        end#if     
   # if we got here we did not found anything intresting      
return false

end#shouldBeExecuted




"""
given number from keyboard input it return array With texture that holds the texture specification we are looking for 
listOfTextSpecifications - list with all registered Texture specifications
numb - string that may represent number - if it does not function will return empty option
return Option - either Texture specification or empty Option 
"""
function findTextureBasedOnNumb(listOfTextSpecifications::Vector{TextureSpec} 
                                ,numb::Int32
                                ,dict::Dictionary{Int32, Int64})::Option
    if(haskey(dict, numb))
        return Option(listOfTextSpecifications[dict[numb]])
    end#if
    #if we are here it mean no such texture was found    
     @info "no texture associated with this number" numb
    return Option()

end #findTextureBasedOnNumb


"""
Given string it parser it to given object on the basis of with and multiple dispatch futher actions will be done
it checks each character weather is numeric - gets substring of all numeric characters and parses it into integer
listOfTextSpecifications - list with all registered Texture specifications
return option of diffrent type depending on input
"""
function parseString(str::Vector{String},actor::SyncActor{Any, ActorWithOpenGlObjects} ,keyInfo::KeyboardStruct)::Option{}
    joined = join(str)
	filtered =  filter(x->isnumeric(x) , joined )
    listOfTextSpecs = actor.actor.mainForDisplayObjects.listOfTextSpecifications
    searchDict = actor.actor.mainForDisplayObjects.numIndexes
    # for controlling window
    if(keyInfo.isF1Pressed)
        return Option(WindowControlStruct(letterCode="F1"))
    elseif(keyInfo.isF2Pressed)
        return Option(WindowControlStruct(letterCode="F2"))
    elseif(keyInfo.isF3Pressed)
        return Option(WindowControlStruct(letterCode="F3"))

    # for undoing actions            
    elseif(occursin("z" , joined) )
        return Option(true)
    # for control of stroke width    
    elseif(keyInfo.isTAbPressed && !isempty(joined) && occursin("+" , joined) )
        return  Option(AnnotationStruct(1))
    elseif(keyInfo.isTAbPressed && !isempty(joined) && occursin("-" , joined) )
        return  Option(AnnotationStruct(-1))
    elseif(isempty(filtered))#nothing to be done   
        return Option()
    # when we want to set new value for manual mask change     
    elseif(keyInfo.isTAbPressed && !isempty(filtered))
        @info "Sending number" parse(Int64,filtered)
        return Option(parse(Int64,filtered))
    #in case we want to change the dimension of plane for slicing data     
    elseif(keyInfo.isSpacePressed && !isempty(filtered)  &&  parse(Int64,filtered)<4)
        @info "changing plane of slicing to " parse(Int64,filtered)
        return Option(setproperties(actor.actor.onScrollData.dataToScrollDims ,  (dimensionToScroll= parse(Int64,filtered)) )    )            
     # in case we want to display diffrence of two masks   
    elseif(occursin("-" , joined))

     mapped = map(splitted-> filter(x->isnumeric(x) , splitted) ,split(joined,"-")) |>
      (filtered)-> filter(it-> it!="", filtered) |>
      (filtered)->map(it->parse(Int32,it)  ,filtered)
        if(length(mapped)==2)
            textSpectOptions = map(it->findTextureBasedOnNumb(listOfTextSpecs,it, searchDict )  ,mapped )
            return Option( (textSpectOptions[1],textSpectOptions[2])  )
         
        end#if    
        return Option()
    # in case we want to undo last action
    end#if
        #in case we have single number
	return   findTextureBasedOnNumb(listOfTextSpecs,parse(Int32,filtered), searchDict ) 
end#strToNumber


end #ReactOnMouseClickAndDrag
