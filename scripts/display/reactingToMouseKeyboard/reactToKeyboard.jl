using DrWatson
@quickactivate "Probabilistic medical segmentation"



ReactOnKeyboardSTR="""
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
module coordinating response to the  keyboard input - mainly shortcuts that  helps controlling  active/visible texture2D
"""
#@doc ReactOnKeyboardSTR
module ReactOnKeyboard
using Main.DisplayWords, Setfield,Main.PrepareWindow,  Main.DataStructs ,Glutils, Rocket, GLFW,Dictionaries, Main.ForDisplayStructs,Main.TextureManag, Main.OpenGLDisplayUtils, Main.Uniforms, Match, Parameters,DataTypesBasic
export reactToKeyboard , registerKeyboardFunctions

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
@doc KeyboardCallbackSubscribableStr
mutable struct KeyboardCallbackSubscribable <: Subscribable{KeyboardStruct}
# true when pressed and kept true until released
# true if corresponding keys are kept pressed and become flase when relesed
    isCtrlPressed::Bool # left - scancode 37 right 105 - Int32
    isShiftPressed::Bool  # left - scancode 50 right 62- Int32
    isAltPressed::Bool# left - scancode 64 right 108- Int32
    isEnterPressed::Bool# scancode 36
    isTAbPressed::Bool# scancode 36
    isSpacePressed::Bool# scancode 36
    lastKeysPressed::Vector{String} # last pressed keys - it listenes to keys only if ctrl/shift or alt is pressed- it clears when we release those case or when we press enter
    subject :: Subject{KeyboardStruct} 
end 

```@doc
will "tell" what functions should be invoked in order to process keyboard input 
```
function Rocket.on_subscribe!(handler::KeyboardCallbackSubscribable, actor::SyncActor{Any, ActorWithOpenGlObjects})
    return subscribe!(handler.subject, actor)
end


handlerStr="""
given pressed keys lik 1-9 and all letters resulting key is encoded as string and will be passed here
handler object responsible for capturing action 
str - name of key lik 1,5 f,.j ... but not ctrl shift etc
action - for example key press or release
scancode - if key do not have short name like ctrl ... it has scancode
"""
@doc handlerStr
function (handler::KeyboardCallbackSubscribable)(str::String, action::GLFW.Action)

    if( (action==instances(GLFW.Action)[2])  ) 
        push!(handler.lastKeysPressed ,str)
   end#if
end #handler

GLFW.PRESS

@doc handlerStr
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
            _ => "notImp" # not Important
         end
            res = KeyboardStruct(isCtrlPressed=handler.isCtrlPressed || scCode=="ctrl" 
                    , isShiftPressed= handler.isShiftPressed ||scCode=="shift" 
                    ,isAltPressed= handler.isAltPressed ||scCode=="alt"
                    ,isSpacePressed= handler.isSpacePressed ||scCode=="space"
                    ,isTAbPressed= handler.isTAbPressed ||scCode=="tab"
                    ,isEnterPressed= handler.isEnterPressed 
                    ,lastKeysPressed= handler.lastKeysPressed 
                    ,mostRecentScanCode = scancode
                    ,mostRecentKeyName = "" # just marking it as empty
                    ,mostRecentAction = action
                    ) 
            

            if(shouldBeExecuted(res,act))
                next!(handler.subject, res ) 
                handler.lastKeysPressed=[] 

            end#if 

    end#if    
  

end #second handler



registerKeyboardFunctionsStr="""
registering functions to the GLFW
window - GLFW window with Visualization
stopListening - atomic boolean enabling unlocking GLFW context
"""
@doc registerKeyboardFunctionsStr
function registerKeyboardFunctions(window::GLFW.Window,stopListening::Base.Threads.Atomic{Bool}    )

    stopListening[]=true # stoping event listening loop to free the GLFW context
                           
    keyboardSubs = KeyboardCallbackSubscribable(false,false,false,false,false,false,[], Subject(KeyboardStruct, scheduler = AsyncScheduler()))
                                  
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




```@doc
processing information from keys - the instance of this function will be chosen on
the basis mainly of multiple dispatch
```
function processKeysInfo(textSpecObs::Identity{TextureSpec{T}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct )where T
    textSpec =  textSpecObs.value
    if(keyInfo.isCtrlPressed)    
        setTextureVisibility(false,textSpec.uniforms )
        @info " set visibility of $(textSpec.name) to false" 
    elseif(keyInfo.isShiftPressed)  
        setTextureVisibility(true,textSpec.uniforms )
        @info " set visibility of $(textSpec.name) to true" 
    elseif(keyInfo.isAltPressed)  
        actor.actor.textureToModifyVec= [textSpec]
        @info " set texture for manual modifications to  $(textSpec.name)"       
    end #if
basicRender(actor.actor.mainForDisplayObjects.window)
end #processKeysInfo


```@doc
for case when we want to subtract two masks
```
function processKeysInfo(maskNumbs::Identity{Tuple{Identity{TextureSpec{T}}, Identity{TextureSpec{T}}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T
    textSpecs =  maskNumbs.value
    maskA = textSpecs[1].value
    maskB = textSpecs[2].value
    
    if(keyInfo.isCtrlPressed)    # when we want to stop displaying diffrence
        @uniforms! begin
        actor.actor.mainForDisplayObjects.mainImageUniforms.isMaskDiffrenceVis:=0
               end
        setTextureVisibility(true,maskA.uniforms )
        setTextureVisibility(true,maskB.uniforms )

    elseif(keyInfo.isShiftPressed)  # when we want to display diffrence
        displayMaskDiffrence(maskA,maskB,actor )
    end #if
basicRender(actor.actor.mainForDisplayObjects.window)
   

end#processKeysInfo


```@doc
in case we want to  get new number set for manual modifications

```
function processKeysInfo(numbb::Identity{Int64},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T

    @info "in  processKeysInfo for tab " numbb.value
    valueForMasToSett = valueForMasToSetStruct(value = numbb.value)
    @info "2"
    actor.actor.valueForMasToSet =valueForMasToSett
    actor.actor.currentlyDispDat.textToDisp= [valueForMasToSett.text,actor.actor.currentlyDispDat.textToDisp...]
    @info "3"

    updateImagesDisplayed(actor.actor.currentlyDispDat
    , actor.actor.mainForDisplayObjects
    , actor.actor.textDispObj
    , actor.actor.calcDimsStruct ,valueForMasToSett )
    @info "4"

  

end#processKeysInfo



processKeysInfo(a::Const{Nothing},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing}, Identity{TextureSpec{T}}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing}, Const{Nothing}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{ Identity{TextureSpec{T}}, Const{Nothing}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T = "" # just doing nothing in case of empty option



displayMaskDiffrenceStr= """
SUBTRACTIN MASKS
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
@doc displayMaskDiffrenceStr
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


end#displayMaskDiffrence



```@doc
invoked when we want to undo last performed action 
```
function processKeysInfo(numb::Identity{Tuple{Bool}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct )


end#processKeysInfo



reactToKeyboardStr = """
Given keyInfo struct wit information about pressed keys it can process them to make some actions  - generally activating keyboard shortcuts
shift + number - make mask associated with given number visible
ctrl + number -  make mask associated with given number invisible 
alt + number -  make mask associated with given number active for mouse interaction 
tab + number - sets the number that will be  used as an input to masks modified by mouse
shift + numberA + "-"(minus sign) +numberB  - display diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B invisible
ctrl + numberA + "-"(minus sign) +numberB  - stops displaying diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B visible
space + 1 or 2 or 3 - change the plane of view (transverse, coronal, sagittal)
ctrl + z - undo last action
"""
@doc reactToKeyboardStr
function reactToKeyboard(keyInfo::KeyboardStruct
                        , actor::SyncActor{Any, ActorWithOpenGlObjects})
                      
    #we got this only when ctrl/shift/als is released or enter is pressed
    obj = actor.actor.mainForDisplayObjects
    obj.stopListening[]=true #free GLFW context

    # processing here on is based on multiple dispatch mainly 
    processKeysInfo(parseString(keyInfo.lastKeysPressed,actor,keyInfo),actor,keyInfo)
    

    obj.stopListening[]=false # reactivete event listening loop

end#reactToKeyboard




```@doc
return true in case the combination of keys should invoke some action
```
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
      GLFW.KEY_ENTER  => return act==1 # returning true if enter is pressed
            _ => false # not Important
         end#match
         @info res
         return res
 
        end#if     
   # if we got here we did not found anything intresting      
return false

end#shouldBeExecuted




```@doc
given number from keyboard input it return array With texture that holds the texture specification we are looking for 
listOfTextSpecifications - list with all registered Texture specifications
numb - string that may represent number - if it does not function will return empty option
return Option - either Texture specification or empty Option 
```
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


```@doc
Given string it parser it to given object on the basis of with and multiple dispatch futher actions will be done
it checks each character weather is numeric - gets substring of all numeric characters and parses it into integer
listOfTextSpecifications - list with all registered Texture specifications
return option of diffrent type depending on input
```
function parseString(str::Vector{String},actor::SyncActor{Any, ActorWithOpenGlObjects} ,keyInfo::KeyboardStruct)::Option{}
    joined = join(str)
	filtered =  filter(x->isnumeric(x) , joined )
    listOfTextSpecs = actor.actor.mainForDisplayObjects.listOfTextSpecifications
    searchDict = actor.actor.mainForDisplayObjects.numIndexes
    if(occursin("z" , joined) )
        return Option(true)

    elseif(isempty(filtered))#nothing to be done   
        return Option()
    elseif(keyInfo.isTAbPressed && !isempty(filtered))# when we want to set new value for manual mask change 
        @info "Sending number" parse(Int64,filtered)
        return Option(parse(Int64,filtered))    
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
