
"""
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
module coordinating response to the  keyboard input - mainly shortcuts that  helps controlling  active/visible texture2D
"""
#@doc ReactOnKeyboardSTR
module ReactOnKeyboard
using ModernGL, ..DisplayWords, ..StructsManag, Setfield, ..PrepareWindow,   ..DataStructs , GLFW,Dictionaries,  ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils,  ..Uniforms, Match, Parameters,DataTypesBasic
using ..KeyboardMouseHelper,..MaskDiffrence, ..KeyboardVisibility, ..OtherKeyboardActions, ..WindowControll, ..ChangePlane
export reactToKeyInput, reactToKeyboard , registerKeyboardFunctions,processKeysInfo





"""
registering functions to the GLFW
window - GLFW window with Visualization
"""
function registerKeyboardFunctions(window::GLFW.Window, mainChannel::Base.Channel{Any})

    GLFW.SetKeyCallback(window, (_, key, scancode, action, mods) -> begin
    println(scancode, action)
    keyInputInstance = KeyInputFields(scancode, action)
    println(keyInputInstance)
    put!(mainChannel, keyInputInstance)
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
Function reactToKeyInput, handled keyboardStruct modification with the keyInput data
passed through the channel
"""
function reactToKeyInput(keyInputInfo::KeyInputFields, mainState::StateDataFields)
    println("working or not ?")
    @info collect(instances(GLFW.Action))[2]
    @info collect(instances(GLFW.Action))[1]
    second_action = collect(instances(GLFW.Action))[2]
    first_action =  collect(instances(GLFW.Action))[1]
    act = nothing
    if keyInputInfo.action == second_action
        act = 1
    elseif keyInputInfo.action == first_action
        act = 2
    else
        act = -1
    end
    println("look here $act" )
   if(act>0)# so we have press or relese

        isCtrlPressed = false
        isShiftPressed = false
        isAltPressed = false
        isSpacePressed = false
        isTabPressed = false
        isF1Pressed = false
        isF2Pressed = false
        isF3Pressed = false
        isF4Pressed = false
        isF5Pressed = false
        isF6Pressed = false
        isZPressed = false
        isFPressed = false
        isSPressed = false
        isPlusPressed = false
        isMinusPressed = false
        isEnterPressed = false
        scCode = ""
        if keyInputInfo.scancode == GLFW.KEY_RIGHT_CONTROL || keyInputInfo.scancode == GLFW.KEY_LEFT_CONTROL
            isCtrlPressed = (act == 1)
            scCode = "ctrl"


        elseif keyInputInfo.scancode == GLFW.KEY_LEFT_SHIFT || keyInputInfo.scancode == GLFW.KEY_RIGHT_SHIFT
            isShiftPressed = (act == 1)
            scCode = "shift"

        elseif keyInputInfo.scancode == GLFW.KEY_RIGHT_ALT || keyInputInfo.scancode == GLFW.KEY_LEFT_ALT
            isAltPressed = (act == 1)
            scCode = "alt"


        elseif keyInputInfo.scancode == GLFW.KEY_SPACE
            isSpacePressed = (act == 1)
            scCode = "space"
        elseif keyInputInfo.scancode == GLFW.KEY_TAB
            isTabPressed = (act == 1)
            scCode = "tab"
        elseif keyInputInfo.scancode == GLFW.KEY_ENTER
            isEnterPressed = (act == 1)
            scCode = "enter"
        elseif keyInputInfo.scancode == GLFW.KEY_F1 || keyInputInfo.scancode == GLFW.KEY_F2 || keyInputInfo.scancode == GLFW.KEY_F3
            isEnterPressed = (act == 1)
            scCode = "f1"
        elseif keyInputInfo.scancode == GLFW.KEY_F4
            isF4Pressed = (act == 1)
            scCode = "f4"
        elseif keyInputInfo.scancode == GLFW.KEY_F5
            isF5Pressed = (act == 1)
            scCode = "f5"
        elseif keyInputInfo.scancode == GLFW.KEY_F6
            isF6Pressed = (act == 1)
            scCode = "f6"
        elseif keyInputInfo.scancode == GLFW.KEY_Z
            isZPressed = (act == 1)
            scCode = "z"
        elseif keyInputInfo.scancode == GLFW.KEY_F
            isFPressed = (act == 1)
            scCode = "f"
        elseif keyInputInfo.scancode == GLFW.KEY_S
            isSPressed = (act == 1)
            scCode = "s"
        elseif keyInputInfo.scancode == GLFW.KEY_KP_ADD || keyInputInfo.scancode == GLFW.KEY_EQUAL
            isPlusPressed = (act == 1)
            scCode = "+"
        elseif keyInputInfo.scancode == GLFW.KEY_KP_SUBTRACT || keyInputInfo.scancode == GLFW.KEY_MINUS
            isMinusPressed = (act == 1)
            scCode = "-"
        else
            scCode = "notImp"
        end
        mainState.fieldKeyboardStruct.isCtrlPressed = isCtrlPressed || scCode=="ctrl"
        mainState.fieldKeyboardStruct.isShiftPressed = isShiftPressed || scCode=="shift"
        mainState.fieldKeyboardStruct.isAltPressed = isAltPressed || scCode=="alt"
        mainState.fieldKeyboardStruct.isSpacePressed = isSpacePressed || scCode=="space"
        mainState.fieldKeyboardStruct.isTabPressed = isTabPressed || scCode=="tab"
        mainState.fieldKeyboardStruct.isF1Pressed = isF1Pressed || scCode=="f1"
        mainState.fieldKeyboardStruct.isF2Pressed = isF2Pressed || scCode=="f2"
        mainState.fieldKeyboardStruct.isF3Pressed = isF3Pressed || scCode=="f3"
        mainState.fieldKeyboardStruct.isF4Pressed = isF4Pressed || scCode=="f4"
        mainState.fieldKeyboardStruct.isF5Pressed = isF5Pressed || scCode=="f5"
        mainState.fieldKeyboardStruct.isF6Pressed = isF6Pressed || scCode=="f6"
        mainState.fieldKeyboardStruct.isZPressed = isZPressed || scCode=="z"
        mainState.fieldKeyboardStruct.isFPressed = isFPressed || scCode=="f"
        mainState.fieldKeyboardStruct.isSPressed = isSPressed || scCode=="s"
        mainState.fieldKeyboardStruct.isPlusPressed = isPlusPressed || scCode=="+"
        mainState.fieldKeyboardStruct.isMinusPressed = isMinusPressed || scCode=="-"
        mainState.fieldKeyboardStruct.isEnterPressed = isEnterPressed
        mainState.fieldKeyboardStruct.lastKeysPressed = keyInputInfo.lastKeysPressed
        mainState.fieldKeyboardStruct.mostRecentScanCode = keyInputInfo.scancode
        mainState.fieldKeyboardStruct.mostRecentKeyName = "" # just marking it as empty
        mainState.fieldKeyboardStruct.mostRecentAction = act

    end
    reactToKeyboard(mainState.fieldKeyboardStruct, mainState)
end



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
