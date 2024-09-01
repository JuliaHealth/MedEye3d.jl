"""
functions that enable modyfing of reactions to mouse using keyboard
for example by pressing f and s we can controll wheather we have fast or slow scroll
"""
module KeyboardMouseHelper

using ..StructsManag, ..PrepareWindow, ..DataStructs, ..ForDisplayStructs, ..TextureManag, ..OpenGLDisplayUtils, ..Uniforms
using Parameters, DataTypesBasic, Logging, Setfield, GLFW


"""
For controlling the  fast scroll  on pressing normal f key and stopping it on n key
"""
function processKeysInfo(isTobeFast::Identity{Tuple{Bool,Bool}}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true)

    isTobeFastVal = isTobeFast.value[1]
    # isTobeFastVal = isTobeFastVal && !isTobeFast.value[2]
    stateObject.mainForDisplayObjects = setproperties(stateObject.mainForDisplayObjects, (isFastScroll = isTobeFastVal))
    # for undoing action
    # if(toBeSavedForBack)
    #     addToforUndoVector(stateObject, ()-> processKeysInfo( Option((!isTobeFastVal,false)),stateObject, keyInfo,false ))
    # end

end#processKeysInfo


end#KeyboardMouseHelper
