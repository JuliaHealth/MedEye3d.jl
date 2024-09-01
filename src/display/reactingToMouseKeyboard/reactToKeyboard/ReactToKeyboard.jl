
"""
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
module coordinating response to the  keyboard input - mainly shortcuts that  helps controlling  active/visible texture2D
"""
#@doc ReactOnKeyboardSTR
module ReactOnKeyboard
using ModernGL, Setfield, GLFW, Dictionaries, Parameters, DataTypesBasic
using ..DisplayWords, ..StructsManag, ..PrepareWindow, ..DataStructs, ..ForDisplayStructs, ..TextureManag, ..OpenGLDisplayUtils, ..Uniforms
using ..KeyboardMouseHelper, ..KeyboardVisibility, ..OtherKeyboardActions, ..WindowControll, ..ChangePlane
# using ..MaskDiffrence
export reactToKeyInput, reactToKeyboard, registerKeyboardFunctions, processKeysInfo





"""
registering functions to the GLFW
window - GLFW window with Visualization
"""
function registerKeyboardFunctions(window::GLFW.Window, mainChannel::Base.Channel{Any})
    GLFW.SetKeyCallback(window, (_, key, scancode, action, mods) -> begin
        # @info "information from the registerKeyboardFunction : scancode : $key, action : $action"
        keyInputInstance = KeyInputFields(scancode=Int32(key), action=action)
        # println(keyInputInstance)
        put!(mainChannel, keyInputInstance)
    end)
end #registerKeyboardFunctions


#multiple dispatch controls what will be invoked
processKeysInfo(a::Const{Nothing}, stateObject::StateDataFields, keyInfo::KeyboardStruct) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing},Identity{TextureSpec{T}}}}, stateObject::StateDataFields, keyInfo::KeyboardStruct) where {T} = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Const{Nothing},Const{Nothing}}}, stateObject::StateDataFields, keyInfo::KeyboardStruct) = "" # just doing nothing in case of empty option
processKeysInfo(a::Identity{Tuple{Identity{TextureSpec{T}},Const{Nothing}}}, stateObject::StateDataFields, keyInfo::KeyboardStruct) where {T} = "" # just doing nothing in case of empty option
#just passing definitions from submodules
processKeysInfo(textSpecObs::Identity{TextureSpec{T}}, stateObject::StateDataFields, keyInfo::KeyboardStruct) where {T} = KeyboardVisibility.processKeysInfo(textSpecObs, stateObject, keyInfo)
processKeysInfo(toScrollDatPrim::Identity{DataToScrollDims}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true) = ChangePlane.processKeysInfo(toScrollDatPrim, stateObject, keyInfo, toBeSavedForBack)
# processKeysInfo(maskNumbs::Identity{Tuple{Identity{TextureSpec{T}},Identity{TextureSpec{G}}}}, stateObject::StateDataFields, keyInfo::KeyboardStruct) where {T,G} = MaskDiffrence.processKeysInfo(maskNumbs, stateObject, keyInfo)
processKeysInfo(numbb::Identity{Int64}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true) = OtherKeyboardActions.processKeysInfo(numbb, stateObject, keyInfo, toBeSavedForBack)
processKeysInfo(numbb::Identity{Bool}, stateObject::StateDataFields, keyInfo::KeyboardStruct) = OtherKeyboardActions.processKeysInfoUndo(numbb, stateObject, keyInfo)
processKeysInfo(annot::Identity{AnnotationStruct}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true) = OtherKeyboardActions.processKeysInfo(annot, stateObject, keyInfo, toBeSavedForBack)
processKeysInfo(isTobeFast::Identity{Tuple{Bool,Bool}}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true) = KeyboardMouseHelper.processKeysInfo(isTobeFast, stateObject, keyInfo, toBeSavedForBack)

processKeysInfo(wind::Identity{WindowControlStruct}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true) = WindowControll.processKeysInfo(wind, stateObject, keyInfo, toBeSavedForBack)





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
function reactToKeyboard(keyInfo::KeyboardStruct, mainState::StateDataFields)
    #we got this only when ctrl/shift/als is released or enter is pressed
    obj = mainState.mainForDisplayObjects
    # processing here on is based on multiple dispatch mainly
    processKeysInfo(parseString(keyInfo.lastKeysPressed, mainState, keyInfo), mainState, keyInfo)
end#reactToKeyboard


"""
Function reactToKeyInput, handled keyboardStruct modification with the keyInput data
passed through the channel
"""
function reactToKeyInput(keyInputInfo::KeyInputFields, mainState::StateDataFields)
    keyReleaseAction = collect(instances(GLFW.Action))[1]
    keyPressAction = collect(instances(GLFW.Action))[2]
    act = nothing
    if keyInputInfo.action == keyReleaseAction
        act = 1
    elseif keyInputInfo.action == keyPressAction
        act = 2
    else
        act = -1
    end

    if (act > 0)# so we have press or relese
        scCode = ""
        if keyInputInfo.scancode == Int32(GLFW.KEY_RIGHT_CONTROL) || keyInputInfo.scancode == Int32(GLFW.KEY_LEFT_CONTROL)
            scCode = "ctrl"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_LEFT_SHIFT) || keyInputInfo.scancode == Int32(GLFW.KEY_RIGHT_SHIFT)
            scCode = "shift"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_RIGHT_ALT) || keyInputInfo.scancode == Int32(GLFW.KEY_LEFT_ALT)
            scCode = "alt"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_SPACE)
            scCode = "space"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_TAB)
            scCode = "tab"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_ENTER)
            scCode = "enter"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F1)
            scCode = "f1"
        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F2)
            scCode = "f2"
        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F3)
            scCode = "f3"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F4)
            scCode = "f4"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F5)
            scCode = "f5"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F6)
            scCode = "f6"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F7)
            scCode = "f7"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F8)
            scCode = "f8"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F9)
            scCode = "f9"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_Z)
            scCode = "z"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_F)
            scCode = "f"


        elseif keyInputInfo.scancode == Int32(GLFW.KEY_S)
            scCode = "s"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_KP_ADD) || keyInputInfo.scancode == Int32(GLFW.KEY_EQUAL)
            scCode = "+"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_KP_SUBTRACT) || keyInputInfo.scancode == Int32(GLFW.KEY_MINUS)
            scCode = "-"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_1)
            scCode = "1"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_2)
            scCode = "2"

        elseif keyInputInfo.scancode == Int32(GLFW.KEY_3)
            scCode = "3"
        else
            scCode = "notImp"
        end
        mainState.fieldKeyboardStruct.isCtrlPressed = (act == 1) && scCode == "ctrl"
        mainState.fieldKeyboardStruct.isShiftPressed = (act == 1) && scCode == "shift"
        mainState.fieldKeyboardStruct.isAltPressed = (act == 1) && scCode == "alt"
        mainState.fieldKeyboardStruct.isSpacePressed = (act == 1) && scCode == "space"
        mainState.fieldKeyboardStruct.isTAbPressed = (act == 1) && scCode == "tab"
        mainState.fieldKeyboardStruct.isEnterPressed = (act == 1) && scCode == "enter"
        mainState.fieldKeyboardStruct.isF1Pressed = (act == 1) && scCode == "f1"
        mainState.fieldKeyboardStruct.isF2Pressed = (act == 1) && scCode == "f2"
        mainState.fieldKeyboardStruct.isF3Pressed = (act == 1) && scCode == "f3"
        mainState.fieldKeyboardStruct.isF4Pressed = (act == 1) && scCode == "f4"
        mainState.fieldKeyboardStruct.isF5Pressed = (act == 1) && scCode == "f5"
        mainState.fieldKeyboardStruct.isF6Pressed = (act == 1) && scCode == "f6"
        mainState.fieldKeyboardStruct.isF7Pressed = (act == 1) && scCode == "f7"
        mainState.fieldKeyboardStruct.isF8Pressed = (act == 1) && scCode == "f8"
        mainState.fieldKeyboardStruct.isF9Pressed = (act == 1) && scCode == "f9"
        mainState.fieldKeyboardStruct.isZPressed = (act == 1) && scCode == "z"
        mainState.fieldKeyboardStruct.isFPressed = (act == 1) && scCode == "f"
        mainState.fieldKeyboardStruct.isSPressed = (act == 1) && scCode == "s"
        mainState.fieldKeyboardStruct.isPlusPressed = scCode == (act == 1) && scCode == "+"
        mainState.fieldKeyboardStruct.isMinusPressed = (act == 1) && scCode == "-"

        push!(mainState.fieldKeyboardStruct.lastKeysPressed, scCode)
    end
    reactToKeyboard(mainState.fieldKeyboardStruct, mainState)
end



"""
return true in case the combination of keys should invoke some action
"""
function shouldBeExecuted(keyInfo::KeyboardStruct, act::Int64)::Bool
    if (act > 0)# so we have press or relese

        if keyInfo.mostRecentScanCode in [Int32(GLFW.KEY_RIGHT_CONTROL), Int32(GLFW.KEY_LEFT_CONTROL), Int32(GLFW.KEY_LEFT_SHIFT), Int32(GLFW.KEY_RIGHT_SHIFT), Int32(GLFW.KEY_RIGHT_ALT), Int32(GLFW.KEY_LEFT_ALT), Int32(GLFW.KEY_SPACE),
            Int32(GLFW.KEY_TAB), Int32(GLFW.KEY_F4), Int32(GLFW.KEY_F5), Int32(GLFW.KEY_F6)]
            return act == 2
        elseif keyInfo.mostRecentScanCode in [Int32(GLFW.KEY_ENTER), Int32(GLFW.KEY_F1), Int32(GLFW.KEY_F2), Int32(GLFW.KEY_F3), Int32(GLFW.KEY_KP_ADD), Int32(GLFW.KEY_EQUAL),
            Int32(GLFW.KEY_KP_SUBTRACT), Int32(GLFW.KEY_MINUS), Int32(GLFW.KEY_Z), Int32(GLFW.KEY_F), Int32(GLFW.KEY_S)]
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
function findTextureBasedOnNumb(listOfTextSpecifications::Vector{TextureSpec}, numb::Int32, dict::Dictionary{Int32,Int64})::Option
    if (haskey(dict, numb))
        return Option(listOfTextSpecifications[dict[numb]])
    end#if
    #if we are here it mean no such texture was found
    # @info "no texture associated with this number" numb
    return Option()

end #findTextureBasedOnNumb


"""
Given string it parser it to given object on the basis of with and multiple dispatch futher actions will be done
it checks each character weather is numeric - gets substring of all numeric characters and parses it into integer
listOfTextSpecifications - list with all registered Texture specifications
return option of diffrent type depending on input
"""
function parseString(str::Vector{String}, stateObject::StateDataFields, keyInfo::KeyboardStruct)::Option{}
    joined = join(str)
    filtered = filter(x -> isnumeric(x), joined)
    # println("here you go filtered with numeric ", filtered)
    listOfTextSpecs = stateObject.mainForDisplayObjects.listOfTextSpecifications
    searchDict = stateObject.mainForDisplayObjects.numIndexes
    # for controlling window
    if (keyInfo.isF1Pressed)
        return Option(WindowControlStruct(letterCode="F1"))
    elseif (keyInfo.isF2Pressed)
        return Option(WindowControlStruct(letterCode="F2"))
    elseif (keyInfo.isF3Pressed)
        return Option(WindowControlStruct(letterCode="F3"))
    elseif (keyInfo.isF4Pressed)
        return Option(WindowControlStruct(letterCode="F4"))
    elseif (keyInfo.isF5Pressed)
        return Option(WindowControlStruct(letterCode="F5"))
    elseif (keyInfo.isF6Pressed)
        return Option(WindowControlStruct(letterCode="F6"))
    elseif (keyInfo.isF7Pressed)
        return Option(WindowControlStruct(letterCode="F7"))
    elseif (keyInfo.isF8Pressed)
        return Option(WindowControlStruct(letterCode="F8"))
    elseif (keyInfo.isF9Pressed)
        return Option(WindowControlStruct(letterCode="F9"))
        # for undoing actions
    elseif (keyInfo.isZPressed)
        return Option(true)
    elseif (keyInfo.isFPressed)
        return Option((true, false))
    elseif (keyInfo.isSPressed)
        return Option((false, true))
        # for control of stroke width
    elseif (keyInfo.isTAbPressed && keyInfo.isPlusPressed)
        return Option(AnnotationStruct(1))
    elseif (keyInfo.isTAbPressed && keyInfo.isMinusPressed)
        return Option(AnnotationStruct(-1))
    elseif (isempty(filtered))#nothing to be done
        return Option()
        # when we want to set new value for manual mask change
    elseif (keyInfo.isTAbPressed && !isempty(filtered))
        return Option(parse(Int64, filtered))
        #in case we want to change the dimension of plane for slicing data
    elseif (keyInfo.isSpacePressed && !isempty(filtered) && parse(Int64, filtered[length(filtered)]) < 4)
        return Option(setproperties(stateObject.onScrollData.dataToScrollDims, (dimensionToScroll = parse(Int64, filtered[length(filtered)]))))
        # in case we want to display diffrence of two masks
    elseif (occursin("m", joined))

        mapped = map(splitted -> filter(x -> isnumeric(x), splitted), split(joined, "m")) |>
                 (filtered) -> filter(it -> it != "", filtered) |>
                               (filtered) -> map(it -> parse(Int32, it), filtered)
        if (length(mapped) == 2)
            textSpectOptions = map(it -> findTextureBasedOnNumb(listOfTextSpecs, it, searchDict), mapped)
            return Option((textSpectOptions[1], textSpectOptions[2]))

        end#if
        return Option()
        # in case we want to undo last action
    end#if
    #in case we have single number
    return findTextureBasedOnNumb(listOfTextSpecs, parse(Int32, filtered[length(filtered)]), searchDict)
end#strToNumber


end #..ReactOnMouseClickAndDrag
