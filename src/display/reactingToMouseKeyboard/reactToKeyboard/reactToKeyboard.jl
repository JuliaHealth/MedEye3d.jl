
"""
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
module coordinating response to the  keyboard input - mainly shortcuts that  helps controlling  active/visible texture2D
"""
#@doc ReactOnKeyboardSTR
module ReactOnKeyboard
using ModernGL, ..DisplayWords, ..StructsManag, Setfield, ..PrepareWindow,   ..DataStructs , Rocket, GLFW,Dictionaries,  ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils,  ..Uniforms, Match, Parameters,DataTypesBasic
using ..KeyboardMouseHelper,..MaskDiffrence, ..KeyboardVisibility, ..OtherKeyboardActions, ..WindowControll, ..ChangePlane
export reactToKeyboard , registerKeyboardFunctions,processKeysInfo



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
            GLFW.KEY_F4 =>( handler.isF4Pressed= (act==1); "f4")
            GLFW.KEY_F5 =>( handler.isF5Pressed= (act==1); "f5")
            GLFW.KEY_F6 =>( handler.isF6Pressed= (act==1); "f6")
            GLFW.KEY_Z =>( handler.isZPressed= (act==1); "z")

            GLFW.KEY_F =>( handler.isFPressed= (act==1); "f")
            GLFW.KEY_S =>( handler.isSPressed= (act==1); "s")
            
            GLFW.KEY_KP_ADD =>( handler.isPlusPressed= (act==1); "+")
            GLFW.KEY_EQUAL =>( handler.isPlusPressed= (act==1); "+")

            GLFW.KEY_KP_SUBTRACT =>( handler.isMinusPressed= (act==1); "-")
            GLFW.KEY_MINUS =>( handler.isMinusPressed= (act==1); "-")
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
                    ,isF4Pressed= handler.isF4Pressed ||scCode=="f4"
                    ,isF5Pressed= handler.isF5Pressed ||scCode=="f5"
                    ,isF6Pressed= handler.isF6Pressed ||scCode=="f6"

                    ,isZPressed= handler.isZPressed ||scCode=="z"
                    ,isFPressed= handler.isFPressed ||scCode=="f"
                    ,isSPressed= handler.isSPressed ||scCode=="s"

                    ,isPlusPressed= handler.isPlusPressed ||scCode=="+"
                    ,isMinusPressed= handler.isMinusPressed ||scCode=="-"

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
                           
    keyboardSubs = KeyboardCallbackSubscribable()


    GLFW.SetKeyCallback(window, (_, key, scancode, action, mods) -> begin
        name = GLFW.GetKeyName(key, scancode)
        if name === nothing || name =="+" || name =="-" || name =="z"
            keyboardSubs(key,action)                                                        
        else
            keyboardSubs(name,action)
        end
        end)

   stopListening[]=false # reactivate event listening loop

return keyboardSubs

end #registerKeyboardFunctions


#multiple dispatch controls what will be invoked
processKeysInfo(a::Const{Nothing},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing}, Identity{TextureSpec{T}}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing}, Const{Nothing}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{ Identity{TextureSpec{T}}, Const{Nothing}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T = "" # just doing nothing in case of empty option
#just passing definitions from submodules
processKeysInfo(textSpecObs::Identity{TextureSpec{T}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T = KeyboardVisibility.processKeysInfo(textSpecObs,actor,keyInfo)
processKeysInfo(toScrollDatPrim::Identity{DataToScrollDims},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct,toBeSavedForBack::Bool = true ) where T = ChangePlane.processKeysInfo(toScrollDatPrim,actor, keyInfo, toBeSavedForBack )
processKeysInfo(maskNumbs::Identity{Tuple{Identity{TextureSpec{T}}, Identity{TextureSpec{G}}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where {T,G} = MaskDiffrence.processKeysInfo(maskNumbs,actor,keyInfo)
processKeysInfo(numbb::Identity{Int64}     ,actor::SyncActor{Any, ActorWithOpenGlObjects}  ,keyInfo::KeyboardStruct    ,toBeSavedForBack::Bool = true) where T = OtherKeyboardActions.processKeysInfo(numbb,actor,keyInfo,toBeSavedForBack   )
processKeysInfo(numbb::Identity{Bool},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T = OtherKeyboardActions.processKeysInfoUndo( numbb, actor,keyInfo  )
processKeysInfo(annot::Identity{AnnotationStruct}  ,actor::SyncActor{Any, ActorWithOpenGlObjects} ,keyInfo::KeyboardStruct ,toBeSavedForBack::Bool = true) where T = OtherKeyboardActions.processKeysInfo(annot,actor,keyInfo,toBeSavedForBack  )

processKeysInfo(isTobeFast::Identity{Tuple{Bool,Bool}}  ,actor::SyncActor{Any, ActorWithOpenGlObjects} ,keyInfo::KeyboardStruct ,toBeSavedForBack::Bool = true) where T = KeyboardMouseHelper.processKeysInfo(isTobeFast,actor,keyInfo,toBeSavedForBack  )

processKeysInfo(wind::Identity{WindowControlStruct} ,actor::SyncActor{Any, ActorWithOpenGlObjects}  ,keyInfo::KeyboardStruct  ,toBeSavedForBack::Bool = true) where T = WindowControll.processKeysInfo(wind,actor,keyInfo,toBeSavedForBack)





"""
Given keyInfo struct wit information about pressed keys it can process them to make some actions  - generally activating keyboard shortcuts
shift + number - make mask associated with given number visible
ctrl + number -  make mask associated with given number invisible 
alt + number -  make mask associated with given number active for mouse interaction 
tab + number - sets the number that will be  used as an input to masks modified by mouse
shift + numberA + "m" +numberB  - display diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B invisible
ctrl + numberA + "m" +numberB  - stops displaying diffrence between masks associated with numberA and numberB - also it makes automaticall mask A and B visible
space + 1 or 2 or 3 - change the plane of view (transverse, coronal, sagittal)
ctrl + z - undo last action
tab +/- increase or decrease stroke width
F1, F2 ... - switch between defined window display characteristics - like min shown white and mx shown black ...
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
      GLFW.KEY_F4  => return act==2
      GLFW.KEY_F5  => return act==2
      GLFW.KEY_F6  => return act==2


      GLFW.KEY_KP_ADD =>return act==1 
      GLFW.KEY_EQUAL =>return act==1 

      GLFW.KEY_KP_SUBTRACT =>return act==1 
      GLFW.KEY_MINUS =>return act==1 
      GLFW.KEY_Z =>return act==1 
      GLFW.KEY_F =>return act==1 
      GLFW.KEY_S =>return act==1 


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
    elseif(keyInfo.isF4Pressed)
        return Option(WindowControlStruct(letterCode="F4"))
    elseif(keyInfo.isF5Pressed)
        return Option(WindowControlStruct(letterCode="F5"))
    elseif(keyInfo.isF6Pressed)
        return Option(WindowControlStruct(letterCode="F6"))
    # for undoing actions            
    elseif(keyInfo.isZPressed )
        return Option(true)
    elseif(keyInfo.isFPressed )
        return Option((true,false))
    elseif(keyInfo.isSPressed )
        return Option((false,true))
    # for control of stroke width    
    elseif(keyInfo.isTAbPressed &&  keyInfo.isPlusPressed)
        return  Option(AnnotationStruct(1))
    elseif(keyInfo.isTAbPressed && keyInfo.isMinusPressed )
        return  Option(AnnotationStruct(-1))
    elseif(isempty(filtered))#nothing to be done   
        return Option()
    # when we want to set new value for manual mask change     
    elseif(keyInfo.isTAbPressed && !isempty(filtered))
        return Option(parse(Int64,filtered))
    #in case we want to change the dimension of plane for slicing data     
    elseif(keyInfo.isSpacePressed && !isempty(filtered)  &&  parse(Int64,filtered)<4)
        return Option(setproperties(actor.actor.onScrollData.dataToScrollDims ,  (dimensionToScroll= parse(Int64,filtered)) )    )            
     # in case we want to display diffrence of two masks   
    elseif(occursin("m" , joined))

     mapped = map(splitted-> filter(x->isnumeric(x) , splitted) ,split(joined,"m")) |>
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


end #..ReactOnMouseClickAndDrag
