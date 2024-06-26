
"""
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
module coordinating response to the  keyboard input - mainly shortcuts that  helps controlling  active/visible texture2D
"""
#@doc ReactOnKeyboardSTR
module ReactOnKeyboard
using ModernGL, ..DisplayWords, ..StructsManag, Setfield, ..PrepareWindow,   ..DataStructs , GLFW,Dictionaries,  ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils,  ..Uniforms, Match, Parameters,DataTypesBasic
using ..KeyboardMouseHelper,..MaskDiffrence, ..KeyboardVisibility, ..OtherKeyboardActions, ..WindowControll, ..ChangePlane
export reactToKeyboard , registerKeyboardFunctions,processKeysInfo





"""
registering functions to the GLFW
window - GLFW window with Visualization
"""
function registerKeyboardFunctions(window::GLFW.Window, mainChannel::Base.Channel{Any})

    GLFW.SetKeyCallback(window, (_, key, scancode, action, mods) -> begin
        name = GLFW.GetKeyName(key, scancode)

        act = nothing
        scCode = nothing
        second_action = collect(instances(GLFW.Action))[2]
        first_action =  collect(instances(GLFW.Action))[1]
        if action == second_action
            act = 1
        elseif action == first_action
            act = 2
        else
            act = -1
        end

       if(act>0)# so we have press or relese

            if scancode == GLFW.KEY_RIGHT_CONTROL || scancode == GLFW.KEY_LEFT_CONTROL
                scCode = "ctrl"
            elseif scancode == GLFW.KEY_LEFT_SHIFT || scancode == GLFW.KEY_RIGHT_SHIFT
                scCode = "shift"
            elseif scancode == GLFW.KEY_RIGHT_ALT || scancode == GLFW.KEY_LEFT_ALT
                scCode = "alt"
            elseif scancode == GLFW.KEY_SPACE
                scCode = "space"
            elseif scancode == GLFW.KEY_TAB
                scCode = "tab"
            elseif scancode == GLFW.KEY_ENTER
                scCode = "enter"
            elseif scancode == GLFW.KEY_F1 || scancode == GLFW.KEY_F2 || scancode == GLFW.KEY_F3
                scCode = "f1"
            elseif scancode == GLFW.KEY_F4
                scCode = "f4"
            elseif scancode == GLFW.KEY_F5
                scCode = "f5"
            elseif scancode == GLFW.KEY_F6
                scCode = "f6"
            elseif scancode == GLFW.KEY_Z
                scCode = "z"
            elseif scancode == GLFW.KEY_F
                scCode = "f"
            elseif scancode == GLFW.KEY_S
                scCode = "s"
            elseif scancode == GLFW.KEY_KP_ADD || scancode == GLFW.KEY_EQUAL
                scCode = "+"
            elseif scancode == GLFW.KEY_KP_SUBTRACT || scancode == GLFW.KEY_MINUS
                scCode = "-"
            else
                scCode = "notImp"
            end
            res = KeyboardStruct(scCode=="ctrl"
                        ,scCode=="shift"
                        ,scCode=="alt"
                        ,scCode=="space"
                        ,scCode=="tab"
                        ,scCode=="f1"
                        ,scCode=="f2"
                        ,scCode=="f3"
                        ,scCode=="f4"
                        ,scCode=="f5"
                        ,scCode=="f6"

                        ,scCode=="z"
                        ,scCode=="f"
                        ,scCode=="s"

                        ,scCode=="+"
                        ,scCode=="-"

                        ,isEnterPressed= ""#handler.isEnterPressed ISSUE
                        ,lastKeysPressed= ""#handler.lastKeysPressed ISSUE
                        ,mostRecentScanCode = scancode
                        ,mostRecentKeyName = "" # just marking it as empty
                        ,mostRecentAction = action)




            if name === nothing || name =="+" || name =="-" || name =="z"  || name =="f"  || name =="s"
                put!(mainChannel, res)
            else
                put!(mainChannel, res)
            end
        end
    end)
end #registerKeyboardFunctions


#multiple dispatch controls what will be invoked
processKeysInfo(a::Const{Nothing},stateObject::StateDataFields,keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing}, Identity{TextureSpec{T}}}},stateObject::StateDataFields,keyInfo::KeyboardStruct ) where T = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing}, Const{Nothing}}},stateObject::StateDataFields,keyInfo::KeyboardStruct ) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{ Identity{TextureSpec{T}}, Const{Nothing}}},stateObject::StateDataFields,keyInfo::KeyboardStruct ) where T = "" # just doing nothing in case of empty option
#just passing definitions from submodules
processKeysInfo(textSpecObs::Identity{TextureSpec{T}},stateObject::StateDataFields,keyInfo::KeyboardStruct ) where T = KeyboardVisibility.processKeysInfo(textSpecObs,stateObject,keyInfo)
processKeysInfo(toScrollDatPrim::Identity{DataToScrollDims},stateObject::StateDataFields,keyInfo::KeyboardStruct,toBeSavedForBack::Bool = true ) where T = ChangePlane.processKeysInfo(toScrollDatPrim,stateObject, keyInfo, toBeSavedForBack )
processKeysInfo(maskNumbs::Identity{Tuple{Identity{TextureSpec{T}}, Identity{TextureSpec{G}}}},stateObject::StateDataFields,keyInfo::KeyboardStruct ) where {T,G} = MaskDiffrence.processKeysInfo(maskNumbs,stateObject,keyInfo)
processKeysInfo(numbb::Identity{Int64}     ,stateObject::StateDataFields  ,keyInfo::KeyboardStruct    ,toBeSavedForBack::Bool = true) where T = OtherKeyboardActions.processKeysInfo(numbb,stateObject,keyInfo,toBeSavedForBack   )
processKeysInfo(numbb::Identity{Bool},stateObject::StateDataFields,keyInfo::KeyboardStruct ) where T = OtherKeyboardActions.processKeysInfoUndo( numbb, stateObject,keyInfo  )
processKeysInfo(annot::Identity{AnnotationStruct}  ,stateObject::StateDataFields ,keyInfo::KeyboardStruct ,toBeSavedForBack::Bool = true) where T = OtherKeyboardActions.processKeysInfo(annot,stateObject,keyInfo,toBeSavedForBack  )

# processKeysInfo(isTobeFast::Identity{Tuple{Bool,Bool}}  ,actor::SyncActor{Any, ActorWithOpenGlObjects} ,keyInfo::KeyboardStruct ,toBeSavedForBack::Bool = true) where T = KeyboardMouseHelper.processKeysInfo(isTobeFast,actor,keyInfo,toBeSavedForBack  )

# processKeysInfo(wind::Identity{WindowControlStruct} ,actor::SyncActor{Any, ActorWithOpenGlObjects}  ,keyInfo::KeyboardStruct  ,toBeSavedForBack::Bool = true) where T = WindowControll.processKeysInfo(wind,actor,keyInfo,toBeSavedForBack)





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
                        , mainState::StateDataFields)

    #we got this only when ctrl/shift/als is released or enter is pressed
    obj = mainState.mainForDisplayObjects
    # processing here on is based on multiple dispatch mainly
    processKeysInfo(parseString(keyInfo.lastKeysPressed,mainState,keyInfo),mainState,keyInfo)
end#reactToKeyboard






"""
return true in case the combination of keys should invoke some action
"""
function shouldBeExecuted(keyInfo::KeyboardStruct, act::Int64)::Bool
    if(act>0)# so we have press or relese

        if keyInfo.mostRecentScanCode in [GLFW.KEY_RIGHT_CONTROL, GLFW.KEY_LEFT_CONTROL, GLFW.KEY_LEFT_SHIFT, GLFW.KEY_RIGHT_SHIFT, GLFW.KEY_RIGHT_ALT, GLFW.KEY_LEFT_ALT, GLFW.KEY_SPACE, GLFW.KEY_TAB, GLFW.KEY_F4, GLFW.KEY_F5, GLFW.KEY_F6]
            return act == 2
        elseif keyInfo.mostRecentScanCode in [GLFW.KEY_ENTER, GLFW.KEY_F1, GLFW.KEY_F2, GLFW.KEY_F3, GLFW.KEY_KP_ADD, GLFW.KEY_EQUAL, GLFW.KEY_KP_SUBTRACT, GLFW.KEY_MINUS, GLFW.KEY_Z, GLFW.KEY_F, GLFW.KEY_S]
            return act == 1
        else
            return false
        end
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
function parseString(str::Vector{String},stateObject::StateDataFields ,keyInfo::KeyboardStruct)::Option{}
    joined = join(str)
	filtered =  filter(x->isnumeric(x) , joined )
    listOfTextSpecs = stateObject.mainForDisplayObjects.listOfTextSpecifications
    searchDict = stateObject.mainForDisplayObjects.numIndexes
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
        return Option(setproperties(stateObject.onScrollData.dataToScrollDims ,  (dimensionToScroll= parse(Int64,filtered)) )    )
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
