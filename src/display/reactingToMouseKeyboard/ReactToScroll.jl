
"""
module that holds functions needed to  react to scrolling
Generally first we need to pass the GLFW callback to the Rocket obeservable 
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
"""
module ReactToScroll
using ModernGL, ..DisplayWords, Rocket, GLFW, ..ForDisplayStructs, ..TextureManag, Logging, ..DataStructs, ..StructsManag

export reactToScroll
export registerMouseScrollFunctions




"""
configuting Rocket on Subscribe so we get custom handler of input as we see we still need to define actor
"""
function Rocket.on_subscribe!(handler::ScrollCallbackSubscribable, actor::SyncActor{Any,ActorWithOpenGlObjects})
    return subscribe!(handler.subject, actor)
end



"""
we define how handler should act on the subject - observable so it will pass event onto subject
If we will scroll fast number will change much  and we will skip some slices
"""
function (handler::ScrollCallbackSubscribable)(_, xoff, yoff)

    if (!handler.isBusy[]) # if program is ready to repsond   
        handler.numberToSend = 0
        next!(handler.subject, Int64(handler.numberToSend + yoff))#true if we scroll up
    else
        handler.numberToSend += yoff
    end
end




"""
uploading data to given texture; of given types associated
returns subscription in order to enable unsubscribing in the end 
window - GLFW window 
stopListening - atomic boolean able to stop the event listening cycle
return scrollback - that holds boolean subject (observable) to which we can react by subscribing appropriate actor
"""
function registerMouseScrollFunctions(window::GLFW.Window, stopListening::Base.Threads.Atomic{Bool}, isBusy::Base.Threads.Atomic{Bool})

    stopListening[] = true # stoping event listening loop to free the GLFW context

    scrollback = ScrollCallbackSubscribable( isBusy,0 ,Subject(Int64, scheduler = AsyncScheduler()))
    # scrollback = ScrollCallbackSubscribable(isBusy, 0, Subject(Int64, scheduler=Rocket.ThreadsScheduler()))
    GLFW.SetScrollCallback(window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))

    stopListening[] = false # reactivate event listening loop

    return scrollback

end #registerMouseScrollFunctions

"""
captures information send from handler that scroll was executed by the 
"""




"""
in case of the scroll p true will be send in case of down - false
in response to it it sets new screen int variable and changes displayed screen
toBeSavedForBack - just marks weather we wat to save the info how to undo latest action
 - false if we invoke it from undoing 
"""
function reactToScroll(scrollNumb::Int64, actor::SyncActor{Any,ActorWithOpenGlObjects}, toBeSavedForBack::Bool=true)
    actor.actor.mainForDisplayObjects.stopListening[] = true
    current = actor.actor.currentDisplayedSlice
    old = current
    #when shift is pressed scrolling is 10 times faster
    if (!actor.actor.mainForDisplayObjects.isFastScroll)
        current += scrollNumb
    else
        current += scrollNumb * 10
    end


    #isScrollUp ? current+=1 : current-=1

    # we do not want to move outside of possible range of slices
    lastSlice = actor.actor.onScrollData.slicesNumber
    if (lastSlice > 1)

        actor.actor.isSliceChanged = true
        actor.actor.isBusy[] = true
        if (current < 1)
            current = 1
        end
        if (lastSlice < 1)
            lastSlice = 1
        end
        if (current >= lastSlice)
            current = lastSlice
        end
        #logic to change displayed screen
        #we select slice that we are intrested in
        singleSlDat = actor.actor.onScrollData.dataToScroll |>
                      (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(current), actor.actor.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
                                  (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=current, textToDisp=getTextForCurrentSlice(actor.actor.onScrollData, Int32(current)))

        updateImagesDisplayed(singleSlDat, actor.actor.mainForDisplayObjects, actor.actor.textDispObj, actor.actor.calcDimsStruct, actor.actor.valueForMasToSet)



        actor.actor.currentlyDispDat = singleSlDat
        # updating the last mouse position so when we will change plane it will better show actual position       
        currentDim = Int64(actor.actor.onScrollData.dataToScrollDims.dimensionToScroll)
        lastMouse = actor.actor.lastRecordedMousePosition
        locArr = [lastMouse[1], lastMouse[2], lastMouse[3]]
        locArr[currentDim] = current
        actor.actor.lastRecordedMousePosition = CartesianIndex(locArr[1], locArr[2], locArr[3])
        #saving information about current slice for future reference
        actor.actor.currentDisplayedSlice = current
        #enable undoing the action
        if (toBeSavedForBack)
            func = () -> reactToScroll(old -= scrollNumb, actor, false)
            addToforUndoVector(actor, func)
        end

    end#if     
    actor.actor.isBusy[] = false
    actor.actor.mainForDisplayObjects.stopListening[] = false


end#reactToScroll





end #ReactToScroll