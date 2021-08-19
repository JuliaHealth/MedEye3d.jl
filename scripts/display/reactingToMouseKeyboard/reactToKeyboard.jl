using DrWatson
@quickactivate "Probabilistic medical segmentation"



ReactOnKeyboardSTR="""
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
module coordinating response to the  keyboard input - mainly shortcuts that  helps controlling  active/visible texture2D
"""
#@doc ReactOnKeyboardSTR
module ReactOnKeyboard
using Rocket, GLFW,Dictionaries, Main.ForDisplayStructs,Main.TextureManag, Main.OpenGLDisplayUtils, Main.Uniforms, Match, Parameters,DataTypesBasic
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
            GLFW.KEY_ENTER =>( handler.isEnterPressed= (act==1); "enter")
            _ => "notImp" # not Important
         end
            res = KeyboardStruct(isCtrlPressed=handler.isCtrlPressed || scCode=="ctrl" 
                    , isShiftPressed= handler.isShiftPressed ||scCode=="shift" 
                    ,isAltPressed= handler.isAltPressed ||scCode=="alt"
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
                           
    keyboardSubs = KeyboardCallbackSubscribable(false,false,false,false,[], Subject(KeyboardStruct, scheduler = AsyncScheduler()))
                                  
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
Registering function how should behave to deal with result of search for texture related to keyboard input 
```
function setVisOnKey(textSpecObs::Identity{TextureSpec},keyInfo::KeyboardStruct  , actor::SyncActor{Any, ActorWithOpenGlObjects}) 

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

end #setVisOnKey

setVisOnKey(a::Const{Nothing},keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option


```@doc
processing information from keys - the instance of this function will be chosen on
the basis mainly of multiple dispatch
```
function processKeysInfo(numb::Identity{Int32},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct )
    opt = findTextureBasedOnNumb(actor.actor.mainForDisplayObjects.listOfTextSpecifications
    , numb.value
    ,actor.actor.mainForDisplayObjects.numIndexes ) 
setVisOnKey( opt,keyInfo , actor)
basicRender(actor.actor.mainForDisplayObjects.window)
end #processKeysInfo

processKeysInfo(a::Const{Nothing},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option


reactToKeyboardStr = """
Given keyInfo struct wit information about pressed keys it can process them to make some actions  - generally activating keyboard shortcuts
shift + number - make mask associated with given number visible
ctrl + number -  make mask associated with given number invisible 
"""
@doc reactToKeyboardStr
function reactToKeyboard(keyInfo::KeyboardStruct
                        , actor::SyncActor{Any, ActorWithOpenGlObjects})
                      
    #we got this only when ctrl/shift/als is released or enter is pressed
    obj = actor.actor.mainForDisplayObjects
    obj.stopListening[]=true #free GLFW context

    # processing here on is based on multiple dispatch mainly 
    processKeysInfo(strToNumber(keyInfo.lastKeysPressed),actor,keyInfo)
    

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
    return Option()

end #findTextureBasedOnNumb


```@doc
Given string it checks each character weather is numeric - gets substring of all numeric characters and parses it into integer
```
function strToNumber(str::Vector{String})::Option{Int32}
	filtered =  filter(x->isnumeric(x) , join(str) )
    if(isempty(filtered))  return Option()  end
	return   Option(parse(Int32, filtered))
end#strToNumber


end #ReactOnMouseClickAndDrag
