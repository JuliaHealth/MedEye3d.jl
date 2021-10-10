"""
functions that enable modyfing of reactions to mouse using keyboard
for example by pressing f and s we can controll wheather we have fast or slow scroll
"""
module KeyboardMouseHelper
using  ..StructsManag, Logging, Setfield, ..PrepareWindow,   ..DataStructs , Rocket, GLFW, ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils,  ..Uniforms, Match, Parameters,DataTypesBasic   
   

"""
For controlling the  fast scroll  on pressing normal f key and stopping it on n key 
"""
function processKeysInfo(isTobeFast::Identity{Tuple{Bool,Bool}}
                        ,actor::SyncActor{Any, ActorWithOpenGlObjects}
                        ,keyInfo::KeyboardStruct 
                        ,toBeSavedForBack::Bool = true) where T

                        isTobeFastVal = isTobeFast.value[1]
    #passing information to actor that we should do now fast scrolling
    # actor.actor.mainForDisplayObjects= setproperties .isFastScroll = isTobeFastVal
       actor.actor.mainForDisplayObjects=setproperties(actor.actor.mainForDisplayObjects, (isFastScroll=isTobeFastVal))
# for undoing action
if(toBeSavedForBack)
    addToforUndoVector(actor, ()-> processKeysInfo( Option((!isTobeFastVal,false)),actor, keyInfo,false ))
end

end#processKeysInfo


end#KeyboardMouseHelper