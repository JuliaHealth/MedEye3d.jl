"""
functions that controll window - so basically treshords for mask display
"""
module WindowControll
using ModernGL, ..DisplayWords, ..StructsManag, Setfield, ..PrepareWindow,   ..DataStructs , Rocket, GLFW,Dictionaries,  ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils,  ..Uniforms, Match, Parameters,DataTypesBasic
export  setTextureWindow,getNewTresholdValue

"""
KEY_F1 - will display wide window for bone Int32(1000),Int32(-1000)
KEY_F2 - will display window for soft tissues Int32(400),Int32(-200)
KEY_F3 - will display wide window for lung viewing  Int32(0),Int32(-1000)
KEY_F4,  KEY_F5 -
    sets minimum (F4) and maximum (KEY_F5) value for display (with combination of + and minus signs - to increase or decrease given treshold) - 
        in case of continuus colors it will clamp values - so all above max will be equaled to max ; and min if smallert than min
        in case of main CT mask - it will controll min shown white and max shown black
        in case of maks with single color associated we will step data so if data is outside the rande it will return 0 - so will not affect display
KEY_F6 - controlls contribution  of given mask to the overall image - maximum value is 1 minimum 0 if we have 3 masks and all control contribution is set to 1 and all are visible their corresponding influence to pixel color is 33%
      if plus is pressed it will increse contribution by 0.1  
      if minus is pressed it will decrease contribution by 0.1  
"""
function processKeysInfo(wind::Identity{WindowControlStruct}
    ,actor::SyncActor{Any, ActorWithOpenGlObjects}
    ,keyInfo::KeyboardStruct 
    ,toBeSavedForBack::Bool = true) where T
    #we have some predefined windows
    joined = join(keyInfo.lastKeysPressed)

    old = actor.actor.mainForDisplayObjects.windowControlStruct 
    windowStruct =  primaryModificationsOfWindContr(wind.value,keyInfo)

    dispatchToFunctions(windowStruct,actor,keyInfo)
  
    #to display change

   basicRender(actor.actor.mainForDisplayObjects.window)

       # for undoing action
    if(toBeSavedForBack)
         addToforUndoVector(actor, ()-> processKeysInfo( Option(old),actor, keyInfo,false ))
    end

end#processKeysInfo

"""
On the basis of the input WindowControlStruct and keyInfo it makes necessary primary modifications to WindowControlStruct 
"""
function primaryModificationsOfWindContr(windowStruct::WindowControlStruct,keyInfo::KeyboardStruct )::WindowControlStruct
    return @match windowStruct.letterCode begin
    "F1" => WindowControlStruct(letterCode="F1",min_shown_white=Int32(1000), max_shown_black=Int32(-1000) )
    "F2" => WindowControlStruct(letterCode="F2",min_shown_white=Int32(400), max_shown_black=Int32(-200) )
    "F3" => WindowControlStruct(letterCode="F3",min_shown_white=Int32(0),max_shown_black= Int32(-1000) )
    "F4" => WindowControlStruct(letterCode="F4", toIncrease= keyInfo.isPlusPressed, toDecrease= keyInfo.isMinusPressed,lower=true )
    "F5" => WindowControlStruct(letterCode="F5",  toIncrease= keyInfo.isPlusPressed,toDecrease=keyInfo.isMinusPressed,upper=true)
    "F6" => WindowControlStruct(letterCode="F6",  toIncrease= keyInfo.isPlusPressed,toDecrease=keyInfo.isMinusPressed,maskContributionToChange=true)
    _ => windowStruct
        end

end   #primaryModificationsOfWindContr 

"""
Based on window struct and key info it will controll  which function should be invoked

"""
function dispatchToFunctions(windowStruct::WindowControlStruct
    ,actor::SyncActor{Any, ActorWithOpenGlObjects}
    ,keyInfo::KeyboardStruct )

   mainWindows=  @match (windowStruct.letterCode) begin
        "F1" =>  (setmainWindow(actor,windowStruct), "Fsth")
        "F2" =>  (setmainWindow(actor,windowStruct), "Fsth")
        "F3" =>  (setmainWindow(actor,windowStruct), "Fsth")
        bad  => "nothing"
end#match

 textureList= actor.actor.textureToModifyVec
currentWindowInActor = actor.actor.mainForDisplayObjects.windowControlStruct
if(mainWindows== "nothing" && !isempty(textureList)) 
    textur = textureList[1]   
     matched = @match (windowStruct.letterCode, windowStruct.toIncrease ,  windowStruct.toDecrease ) begin 
        ("F4" ,true,false) =>    lowTreshUp(windowStruct,actor,textur,currentWindowInActor)
        ("F4" ,false,true) =>    lowTreshDown(windowStruct,actor,textur,currentWindowInActor)
        ("F5" ,false,true) =>    highTreshDown(windowStruct,actor,textur,currentWindowInActor)
        ("F5" ,true,false) =>    highTreshUp(windowStruct,actor,textur,currentWindowInActor)
        ("F6" ,true,false) =>    maskContrUp(windowStruct,actor,textur,currentWindowInActor)
        ("F6" ,false,true) =>    maskContrDown(windowStruct,actor,textur,currentWindowInActor)
        bad                 => "nothing"     
    end#match
else
        #updating current windowing object and getting reference to old
        actor.actor.mainForDisplayObjects= setproperties(actor.actor.mainForDisplayObjects,(windowControlStruct =windowStruct))

end #if

end#dispatchToFunctions

"""
sets lower treshold and Increase it
"""
function lowTreshUp(windowStruct::WindowControlStruct,actor::SyncActor{Any, ActorWithOpenGlObjects},textur::TextureSpec,currentWindowInActor::WindowControlStruct) 
    if(textur.isMainImage)
        setmainWindow(actor,setproperties(currentWindowInActor, (max_shown_black= currentWindowInActor.max_shown_black+15)))        
    else
        textur.minAndMaxValue[1]+=getNewTresholdChangeValue(textur)
        setTextureWindow( textur, actor )
    end#if    
end#lowTreshUp
"""
sets lower treshold and decrese it
"""
function lowTreshDown(windowStruct::WindowControlStruct,actor::SyncActor{Any, ActorWithOpenGlObjects},textur::TextureSpec,currentWindowInActor::WindowControlStruct) 
    if(textur.isMainImage)
        setmainWindow(actor,setproperties(currentWindowInActor, (max_shown_black= currentWindowInActor.max_shown_black-15)))        
    else
        texturParam = parameter_type(textur) 
        if(texturParam==UInt8 ||texturParam==UInt16 ||texturParam==UInt32 ||texturParam==UInt64 )
            textur.minAndMaxValue[1]= max(0,textur.minAndMaxValue[1]-getNewTresholdChangeValue(textur))
        else
            textur.minAndMaxValue[1]-=getNewTresholdChangeValue(textur)    
        end#if    
        setTextureWindow( textur, actor )
    end#if   
end#lowTreshDown

"""
sets upper treshold and Increase it
"""
function highTreshUp(windowStruct::WindowControlStruct,actor::SyncActor{Any, ActorWithOpenGlObjects},textur::TextureSpec,currentWindowInActor::WindowControlStruct) 
    if(textur.isMainImage)
        setmainWindow(actor,setproperties(currentWindowInActor, (min_shown_white= currentWindowInActor.min_shown_white+15)))        
    else
       
            textur.minAndMaxValue[2]+=max(getNewTresholdChangeValue(textur),1)
      
        setTextureWindow( textur, actor )
    end#if   
end#highTreshUp

"""
sets upper treshold and decrese it
"""
function highTreshDown(windowStruct::WindowControlStruct,actor::SyncActor{Any, ActorWithOpenGlObjects},textur::TextureSpec,currentWindowInActor::WindowControlStruct) 
    if(textur.isMainImage)
        setmainWindow(actor,setproperties(currentWindowInActor, (min_shown_white= currentWindowInActor.min_shown_white-15)))                
    else
        texturParam = parameter_type(textur) 
        if(texturParam==UInt8 ||texturParam==UInt16 ||texturParam==UInt32 ||texturParam==UInt64 )
            textur.minAndMaxValue[2]= maximum([0,textur.minAndMaxValue[2]-getNewTresholdChangeValue(textur), textur.minAndMaxValue[1]] )
        else
            textur.minAndMaxValue[2]=max(textur.minAndMaxValue[2]-getNewTresholdChangeValue(textur) ,textur.minAndMaxValue[1] ) 
        end#if   
        setTextureWindow( textur, actor )
    end#if 
end#highTreshDown


"""
sets mask contribution and  decrese it
"""
function maskContrDown(windowStruct::WindowControlStruct,actor::SyncActor{Any, ActorWithOpenGlObjects},textur::TextureSpec,currentWindowInActor::WindowControlStruct) 
    if(textur.isMainImage)
        changeMainTextureContribution(textur,Float32(-0.1), actor)
    else
        changeTextureContribution(textur,Float32(-0.1) )
    end#if
end#maskContrDown

"""
sets mask contribution and increase it
"""
function maskContrUp(windowStruct::WindowControlStruct,actor::SyncActor{Any, ActorWithOpenGlObjects},textur::TextureSpec,currentWindowInActor::WindowControlStruct) 
    if(textur.isMainImage)
        changeMainTextureContribution(textur, Float32(0.1), actor)
    else
        changeTextureContribution(textur, Float32(0.1) )
    end#if
end#maskContrUp

"""
set main window - min shown white and max shown black on the basis of textur data and windowStruct
"""
function setmainWindow(actor::SyncActor{Any, ActorWithOpenGlObjects},windowStruct::WindowControlStruct)
        #updating current windowing object and getting reference to old
        setCTWindow(windowStruct.min_shown_white,windowStruct.max_shown_black, actor.actor.mainForDisplayObjects.mainImageUniforms)
        actor.actor.mainForDisplayObjects= setproperties(actor.actor.mainForDisplayObjects,(windowControlStruct =windowStruct))
end#setmainWindow



"""
sets minimum and maximum value for display - 
    in case of continuus colors it will clamp values - so all above max will be equaled to max ; and min if smallert than min
    in case of main CT mask - it will controll min shown white and max shown black
    in case of maks with single color associated we will step data so if data is outside the rande it will return 0 - so will not affect display
"""
function setTextureWindow(textur::TextureSpec,actor::SyncActor{Any, ActorWithOpenGlObjects})
    # activeTextureList= actor.actor.textureToModifyVec
    # allTexturesList = actor.actor.mainForDisplayObjects.listOfTextSpecifications
    # notModifiedTextures= filter(it->it.name!=textur.name, allTexturesList)

    coontrolMinMaxUniformVals(textur)
end#    setTextureWindow



"""
helper function for setTextureWindow on the basis of given texture spec will give value proportional to the range
"""
function getNewTresholdChangeValue(textur::TextureSpec)::Int64
    return  ceil(abs(textur.minAndMaxValue[2]-textur.minAndMaxValue[1])/20 )  
end #getNewTresholdValue   


end#WindowControll



    # #in case we pressed F4 or F5 we want to manually change the window
    #     if( !isempty(textureList) && (windowStruct.toIncrease || windowStruct.toDecrease    ) )
    #         textur= textureList[1]
    #         if(textur.isMainImage)
    #             if(windStr.lower)
    #                 windowStruct= setproperties(windowStruct,  (max_shown_black= getNewTresholdValue(old.max_shown_black, windowStruct),min_shown_white=old.min_shown_white    ) )
    #             elseif(windStr.upper)   
    #                 windowStruct= setproperties(windowStruct,  (min_shown_white= getNewTresholdValue(old.min_shown_white, windowStruct) , max_shown_black=old.max_shown_black  ) )
    #             end#ifs
    #             setCTWindow(windowStruct.min_shown_white,windowStruct.max_shown_black, actor.actor.mainForDisplayObjects.mainImageUniforms)
    #         else# if texture is not main image    
    #             setTextureWindow(textur, windowStruct)  
    #         end#if
    #     else        
    #     # setting window and showig it 
    #     setCTWindow(windowStruct.min_shown_white,windowStruct.max_shown_black, actor.actor.mainForDisplayObjects.mainImageUniforms)
    #     end#if   