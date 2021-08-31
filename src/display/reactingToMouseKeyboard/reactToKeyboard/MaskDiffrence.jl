
"""
for case when we want to subtract two masks
"""
module MaskDiffrence
using ModernGL, ..DisplayWords, ..StructsManag, Setfield, ..PrepareWindow,   ..DataStructs , Rocket, GLFW,Dictionaries,  ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils,  ..Uniforms, Match, Parameters,DataTypesBasic
export undoDiffrence,displayMaskDiffrence
"""
for case when we want to subtract two masks
"""
function processKeysInfo(maskNumbs::Identity{Tuple{Identity{TextureSpec{T}}, Identity{TextureSpec{G}}}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where {T,G}
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
SUBTRACTING MASKS
used in order to enable subtracting one mask from the other - hence displaying 
pixels where value of mask a is present but mask b not (order is important)
automatically both masks will be set to be invisible and only the diffrence displayed

In order to achieve this  we need to have all of the samplers references stored in a list 
1) we need to set both masks to invisible - it will be done from outside the shader
2) we set also from outside uniform marking visibility of diffrence to true
3) also from outside we need to set which texture to subtract from which we will achieve this by setting maskAtoSubtr and maskBtoSubtr int ..Uniforms
    those integers will mark which samplers function will use
4) in shader function will be treated as any other mask and will give contribution to output color multiplied by its visibility(0 or 1)    
5) inside the function color will be defined as multiplication of two colors of mask A and mask B - colors will be acessed similarly to samplers
6) color will be returned only if value associated with  maskA is greater than mask B and proportional to this difffrence

In order to provide maximum performance and avoid branching inside shader multiple shader programs will be attached and one choosed  that will use diffrence needed
maskToSubtrastFrom,maskWeAreSubtracting - specifications o textures we are operating on 
"""
function displayMaskDiffrence(maskA::TextureSpec, maskB::TextureSpec,actor::SyncActor{Any, ActorWithOpenGlObjects})
    if(!maskA.isMainImage && !maskB.isMainImage)
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
    end#if
end#displayMaskDiffrence


end#MaskDiffrence