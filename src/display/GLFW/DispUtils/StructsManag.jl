
using ColorTypes

"""
utilities for dealing data structs like FullScrollableDat or SingleSliceDat
"""
module StructsManag
using  Setfield,   ..ForDisplayStructs,   ..DataStructs, Rocket
export getThreeDims,addToforUndoVector,cartTwoToThree,getHeightToWidthRatio,threeToTwoDimm,modSlice!, threeToTwoDimm,modifySliceFull!,getSlicesNumber,getMainVerticies,AsyncScheduler_spawned

```@doc
given two dim dat it sets points in given coordinates in given slice to given value
coords - coordinates in a plane of chosen slice to modify
value - value to set for given points
return reference to modified slice
```
function modSlice!(data::TwoDimRawDat{T}
                ,coords::Vector{CartesianIndex{2}}
                ,value::T ) where {T}
   data.dat[coords].=value
   data
end#modSlice



```@doc
gives access to the slice of intrest - way of slicing is defined at the begining
typ - type of data 
slice - slice we want to access
sliceDim - on the basis of what dimension we are slicing
return 2 dimensional array  wrapper -TwoDimRawDat  object representing slice of given 3 dimensional array
!! important returned TwoDimRawDat holds view to the original 3 dimensional data  
```
function threeToTwoDimm(typ::Type{T}
                ,sliceInner::Int
                ,sliceDim::Int
                ,threedimDat::ThreeDimRawDat{T})::TwoDimRawDat{T} where {T}
         
maxSlice =size(threedimDat.dat)[sliceDim]
slice= sliceInner 
if(sliceInner>maxSlice) slice=maxSlice   end
               return TwoDimRawDat{T}(typ,threedimDat.name,selectdim(threedimDat.dat, sliceDim, slice)   )
end#ThreeToTwoDimm


modifySliceFull!Str= """
modifies given slice in given coordinates of given data - queried by name
data - full data we work on and modify
coords - coordinates in a plane of chosen slice to modify (so list of x and y coords)
value - value to set for given points
return reference to modified slice
"""
@doc modifySliceFull!Str
function modifySliceFull!(data::FullScrollableDat
                        ,slice::Int
                        ,coords::Vector{CartesianIndex{2}}
                        ,name::String
                        ,value)
                      
     threeDimDat=data.nameIndexes[name] |>
     (ind)-> data.dataToScroll[ind]
     if(typeof(value)!=threeDimDat.type )  throw(DomainError(value, "supplied value should be of compatible type - $(threeDimDat.type )"))  end #if

     return threeToTwoDimm(threeDimDat.type,slice,data.dimensionToScroll,threeDimDat  ) |>
     (twoDimDat)-> modSlice!(twoDimDat,coords,value)
end#modifySliceFull!

```@doc
Return number of slices present in on slice data - takes into account slices dimensions
```
function getSlicesNumber(data::FullScrollableDat)::Int32
return Int32(size(data.dataToScroll[1].dat)[data.dimensionToScroll])
end#getSlicesNumber



```@doc
Based on DataToScrollDims it will enrich passed CalcDimsStruct texture width, height and  heightToWithRatio
based on data passed from DataToScrollDims
```
function getHeightToWidthRatio( calcDim::CalcDimsStruct ,dataToScrollDims::DataToScrollDims)::CalcDimsStruct
  toSelect= filter(it-> it!=dataToScrollDims.dimensionToScroll , [1,2,3] )# will be used to get texture width and height

  return setproperties(calcDim, (imageTextureWidth= dataToScrollDims.imageSize[toSelect[1]]
                              ,imageTextureHeight =dataToScrollDims.imageSize[toSelect[2]]
                              ,heightToWithRatio =  dataToScrollDims.voxelSize[toSelect[1]]/dataToScrollDims.voxelSize[toSelect[2]]
                              ,textTextureZeros= calcDim.textTextureZeros  
  ))
end#getHeightToWidthRatio



```@doc
Based on DataToScrollDims ,2 dim cartesian coordinate and  slice number it gives 3 dimensional coordinate of mouse position
```
function cartTwoToThree(dataToScrollDims::DataToScrollDims 
                           ,sliceNumber::Int
                           ,cartIn::CartesianIndex{2})::CartesianIndex{3}
dimToScroll = dataToScrollDims.dimensionToScroll
toSelect= filter(it-> it!=dimToScroll , [1,2,3] )# will be used to get texture width and height
resArr= [1,1,1]


resArr[toSelect[1]] = cartIn[1]
resArr[toSelect[2]] = cartIn[2]

resArr[dimToScroll]=Int64(sliceNumber)

  return CartesianIndex(resArr[1],resArr[2],resArr[3]  )
end#cartTwoToThree




```@doc
Given function and actor it passes the function to forUndoVector -
   in case the length of the vector is too big the last element woill be removed
```
function addToforUndoVector(actor::SyncActor{Any, ActorWithOpenGlObjects} 
                           ,fun)

   push!(actor.actor.forUndoVector,fun)

    if(length(actor.actor.forUndoVector) >actor.actor.maxLengthOfForUndoVector )
      popfirst!(actor.actor.forUndoVector)
    end  

end#addToforUndoVector


```@doc
utility function to create series of ThreeDimRawDat from list of tuples where
first entry is String and second entry is 3 dimensional array with data 
```
function getThreeDims(list )

  return map(tupl->ThreeDimRawDat{typeof(tupl[2][1])}(typeof(tupl[2][1]),tupl[1],tupl[2]) ,list)

end#getThreeDims


# parameter_type(x::TextureSpec) = parameter_type(typeof(x))



```@doc
calculates proper dimensions form main quad display on the basis of data stored in CalcDimsStruct 
some of the values calculated will be needed for futher derivations for example those that will  calculate mouse positions
reurn CalcDimsStruct enriched by new data 
    ```
function getMainVerticies(calcDimStruct::CalcDimsStruct)::CalcDimsStruct
    #corrections that will be added on both sides (in case of height correction top and bottom in case of width correction left and right)
    # to achieve required ratio
    widthCorr=0.0
    heightCorr=0.0

    correCtedWindowQuadHeight= calcDimStruct.avWindHeightForMain
    correCtedWindowQuadWidth = calcDimStruct.avWindWidtForMain
    isWidthToBeCorrected = false
    isHeightToBeCorrected = false

    
    #first one need to check weather current height to width ratio is as we want it  and if not  is it to high or to low
    if(calcDimStruct.avMainImRatio>calcDimStruct.heightToWithRatio ) 
        #if we have to big height to width ratio we need to reduce size of acual quad from top and bottom
        # we know that we would not need to change width  hence we will use the width to calculate the quad height
        correCtedWindowQuadHeight= calcDimStruct.heightToWithRatio * calcDimStruct.avWindWidtForMain
        isHeightToBeCorrected= true
    end# if to heigh
    
    if(calcDimStruct.avMainImRatio<calcDimStruct.heightToWithRatio )
        #if we have to low height to width ratio we need to reduce size of acual quad from left and right
        # we know that we would not need to change height  hence we will use height to calculate the quad height
        correCtedWindowQuadWidth = calcDimStruct.avWindHeightForMain/calcDimStruct.heightToWithRatio 
        isWidthToBeCorrected= true
    end# if to wide

    # now we still need ratio of the resulting quad window size after corrections relative to  total window size
    quadToTotalHeightRatio= correCtedWindowQuadHeight/calcDimStruct.windowHeight
    quadToTotalWidthRatio= correCtedWindowQuadWidth/calcDimStruct.windowWidth

    # original ratios of available space of main image to total window dimensions
    avQuadToTotalHeightRatio= calcDimStruct.avWindHeightForMain/calcDimStruct.windowHeight
    avQuadToTotalWidthRatio= calcDimStruct.avWindWidtForMain/calcDimStruct.windowWidth
    # if those would be equal to corrected ones we would just start from the  bottom left corner and create quad from there - yet now we need to calculate corrections based on the diffrence of quantities just above and corrected ones
    
    if(isHeightToBeCorrected)
        heightCorr=abs(quadToTotalHeightRatio-avQuadToTotalHeightRatio)
    end # if isHeightToBeCorrected

    if(isWidthToBeCorrected)
        widthCorr=abs(quadToTotalWidthRatio-avQuadToTotalWidthRatio)
    end # if isWidthToBeCorrected

    correctedWidthForTextAccounting = (-1+ calcDimStruct.fractionOfMainIm*2)
    #as in OpenGl we start from -1 and end at 1 those ratios needs to be doubled in order to translate them in the OPEN Gl coordinate system yet we will achieve this doubling by just adding the corrections from both sides
    #hence we do not need  to multiply by 2 becose we get from -1 to 1 so total is 2 
      res =  Float32.([
      # positions                  // colors           // texture coords
      correctedWidthForTextAccounting-widthCorr,  1.0-heightCorr, 0.0,   1.0, 0.0, 0.0,   1.0, 1.0,   # top right
      correctedWidthForTextAccounting-widthCorr, -1.0+heightCorr, 0.0,   0.0, 1.0, 0.0,   1.0, 0.0,   # bottom right
      -1.0+widthCorr, -1.0+heightCorr, 0.0,                0.0, 0.0, 1.0,   0.0, 0.0,   # bottom left
      -1.0+widthCorr,  1.0-heightCorr, 0.0,                1.0, 1.0, 0.0,   0.0, 1.0    # top left 
    ])

    windowWidthCorr=Int32(round( (widthCorr/2)*calcDimStruct.windowWidth))
    windowHeightCorr= Int32(round((heightCorr/2)*calcDimStruct.windowHeight))
   
   



    return setproperties(calcDimStruct, (correCtedWindowQuadHeight= Int32(round(correCtedWindowQuadHeight))
                                        ,correCtedWindowQuadWidth= Int32(round(correCtedWindowQuadWidth))
                                        ,quadToTotalHeightRatio=quadToTotalHeightRatio
                                        ,quadToTotalWidthRatio=quadToTotalWidthRatio
                                        ,widthCorr=widthCorr
                                        ,heightCorr=heightCorr
                                        ,mainImageQuadVert=res
                                        ,mainQuadVertSize = sizeof(res)
                                        ,windowWidthCorr=windowWidthCorr
                                        ,windowHeightCorr=windowHeightCorr
                                        ))
    
    

    end #getMainVerticies







####################### adapted from https://github.com/biaslab/Rocket.jl/blob/8aa557c90717bed9d24e36cc6b147dcc076d6b67/src/schedulers/async.jl






    import Base: show, similar

    """
        AsyncScheduler_spawned
    
    `AsyncScheduler_spawned` executes scheduled actions asynchronously and uses `Channel` object to order different actions on a single asynchronous task
    """
    struct AsyncScheduler_spawned{N} <: Rocket.AbstractScheduler end
    
    Base.show(io::IO, ::AsyncScheduler_spawned) = print(io, "AsyncScheduler_spawned()")
    
    function AsyncScheduler_spawned(size::Int = typemax(Int))
        return AsyncScheduler_spawned{size}()
    end
    
    Base.similar(::AsyncScheduler_spawned{N}) where N = AsyncScheduler_spawned{N}()
    
    makeinstance(::Type{D}, ::AsyncScheduler_spawned{N}) where { D, N } = AsyncScheduler_spawnedInstance{D}(N)
    
    instancetype(::Type{D}, ::Type{<:AsyncScheduler_spawned}) where D = AsyncScheduler_spawnedInstance{D}
    
    struct AsyncScheduler_spawnedDataMessage{D}
        data :: D
    end
    
    struct AsyncScheduler_spawnedErrorMessage
        err
    end
    
    struct AsyncScheduler_spawnedCompleteMessage end
    
    const AsyncScheduler_spawnedMessage{D} = Union{AsyncScheduler_spawnedDataMessage{D}, AsyncScheduler_spawnedErrorMessage, AsyncScheduler_spawnedCompleteMessage}
    
    mutable struct AsyncScheduler_spawnedInstance{D}
        channel        :: Channel{AsyncScheduler_spawnedMessage{D}}
        isunsubscribed :: Bool
        subscription   :: Teardown
    
        AsyncScheduler_spawnedInstance{D}(size::Int = typemax(Int)) where D = begin
            return new(Channel{AsyncScheduler_spawnedMessage{D}}(size, spawn=true), false, voidTeardown)
        end
    end
    
    isunsubscribed(instance::AsyncScheduler_spawnedInstance) = instance.isunsubscribed
    getchannel(instance::AsyncScheduler_spawnedInstance) = instance.channel
    
    function dispose(instance::AsyncScheduler_spawnedInstance)
        if !isunsubscribed(instance)
            instance.isunsubscribed = true
            close(instance.channel)
            @async begin
                unsubscribe!(instance.subscription)
            end
        end
    end
    
    function __process_channeled_message(instance::AsyncScheduler_spawnedInstance{D}, message::AsyncScheduler_spawnedDataMessage{D}, actor) where D
        on_next!(actor, message.data)
    end
    
    function __process_channeled_message(instance::AsyncScheduler_spawnedInstance, message::AsyncScheduler_spawnedErrorMessage, actor)
        on_error!(actor, message.err)
        dispose(instance)
    end
    
    function __process_channeled_message(instance::AsyncScheduler_spawnedInstance, message::AsyncScheduler_spawnedCompleteMessage, actor)
        on_complete!(actor)
        dispose(instance)
    end
    
    struct AsyncScheduler_spawnedSubscription{ H <: AsyncScheduler_spawnedInstance } <: Teardown
        instance :: H
    end
    
    Base.show(io::IO, ::AsyncScheduler_spawnedSubscription) = print(io, "AsyncScheduler_spawnedSubscription()")
    
    as_teardown(::Type{ <: AsyncScheduler_spawnedSubscription}) = UnsubscribableTeardownLogic()
    
    function on_unsubscribe!(subscription::AsyncScheduler_spawnedSubscription)
        dispose(subscription.instance)
        return nothing
    end
    
    function scheduled_subscription!(source, actor, instance::AsyncScheduler_spawnedInstance)
        subscription = AsyncScheduler_spawnedSubscription(instance)
    
        channeling_task = @async begin
            while !isunsubscribed(instance)
                message = take!(getchannel(instance))
                if !isunsubscribed(instance)
                    __process_channeled_message(instance, message, actor)
                end
            end
        end
    
        subscription_task = @async begin
            if !isunsubscribed(instance)
                tmp = on_subscribe!(source, actor, instance)
                if !isunsubscribed(instance)
                    subscription.instance.subscription = tmp
                else
                    unsubscribe!(tmp)
                end
            end
        end
    
        bind(getchannel(instance), channeling_task)
    
        return subscription
    end
    
    function scheduled_next!(actor, value::D, instance::AsyncScheduler_spawnedInstance{D}) where { D }
        put!(getchannel(instance), AsyncScheduler_spawnedDataMessage{D}(value))
    end
    
    function scheduled_error!(actor, err, instance::AsyncScheduler_spawnedInstance)
        put!(getchannel(instance), AsyncScheduler_spawnedErrorMessage(err))
    end
    
    function scheduled_complete!(actor, instance::AsyncScheduler_spawnedInstance)
        put!(getchannel(instance), AsyncScheduler_spawnedCompleteMessage())
    end




    import Base: show, similar

    ##
    
    struct SubjectListener{I}
        schedulerinstance :: I
        actor
    end
    
    Base.show(io::IO, ::SubjectListener) = print(io, "SubjectListener()")
    
    ##
    function convert(::Rocket.Subject, subj::MedEye3d.StructsManag.Subject)
        return subj
    end    

    """
        Subject(::Type{D}; scheduler::H = AsapScheduler())
    
    A Subject is a special type of Observable that allows values to be multicasted to many Observers. Subjects are like EventEmitters.
    Every Subject is an Observable and an Actor. You can subscribe to a Subject, and you can call `next!` to feed values as well as `error!` and `complete!`.
    
    Note: By convention, every actor subscribed to a Subject observable is not allowed to throw exceptions during `next!`, `error!` and `complete!` calls. 
    Doing so would lead to undefined behaviour. Use `safe()` operator to bypass this rule. 
    
    See also: [`SubjectFactory`](@ref), [`ReplaySubject`](@ref), [`BehaviorSubject`](@ref), [`safe`](@ref)
    """
    mutable struct Subject{D, H, I} <: Rocket.AbstractSubject{D}
        listeners   :: Rocket.List{SubjectListener{I}}
        scheduler   :: H
        isactive    :: Bool
        iscompleted :: Bool
        isfailed    :: Bool
        lasterror   :: Any
    
        Subject{D, H, I}(scheduler::H) where { D, H <: Rocket.AbstractScheduler, I } = new(Rocket.List(SubjectListener{I}), scheduler, true, false, false, nothing)
    end
    
    function Subject(::Type{D}; scheduler::H = AsapScheduler()) where { D, H <: Rocket.AbstractScheduler }
        return Subject{D, H, instancetype(D, H)}(scheduler)
    end
    
    Base.show(io::IO, ::Subject{D, H}) where { D, H } = print(io, "Subject($D, $H)")
    
    Base.similar(subject::Subject{D, H}) where { D, H } = Subject(D; scheduler = similar(subject.scheduler))
    
    ##
    
    isactive(subject::Subject)    = subject.isactive
    iscompleted(subject::Subject) = subject.iscompleted
    isfailed(subject::Subject)    = subject.isfailed
    lasterror(subject::Subject)   = subject.lasterror
    
    setinactive!(subject::Subject)       = subject.isactive    = false
    setcompleted!(subject::Subject)      = subject.iscompleted = true
    setfailed!(subject::Subject)         = subject.isfailed    = true
    setlasterror!(subject::Subject, err) = subject.lasterror   = err
    
    ##
    
    function on_next!(subject::Subject{D, H, I}, data::D) where { D, H, I }
        for listener in subject.listeners
            scheduled_next!(listener.actor, data, listener.schedulerinstance)
        end
    end
    
    function on_error!(subject::Subject, err)
        if isactive(subject)
            setinactive!(subject)
            setfailed!(subject)
            setlasterror!(subject, err)
            for listener in subject.listeners
                scheduled_error!(listener.actor, err, listener.schedulerinstance)
            end
            empty!(subject.listeners)
        end
    end
    
    function on_complete!(subject::Subject)
        if isactive(subject)
            setinactive!(subject)
            setcompleted!(subject)
            for listener in subject.listeners
                scheduled_complete!(listener.actor, listener.schedulerinstance)
            end
            empty!(subject.listeners)
        end
    end
    
    ##
    
    function on_subscribe!(subject::Subject{D}, actor) where { D }
        if isfailed(subject)
            error!(actor, lasterror(subject))
            return SubjectSubscription(nothing)
        elseif iscompleted(subject)
            complete!(actor)
            return SubjectSubscription(nothing)
        else
            instance = makeinstance(D, subject.scheduler)
            return scheduled_subscription!(subject, actor, instance)
        end
    end
    
    function on_subscribe!(subject::Subject, actor, instance)
        listener      = SubjectListener(instance, actor)
        listener_node = pushnode!(subject.listeners, listener)
        return SubjectSubscription(listener_node)
    end
    
    ##
    
    mutable struct SubjectSubscription <: Rocket.Teardown
        listener_node :: Union{Nothing, Rocket.ListNode}
    end
    
    as_teardown(::Type{ <: SubjectSubscription }) = UnsubscribableTeardownLogic()
    
    function on_unsubscribe!(subscription::SubjectSubscription)
        if subscription.listener_node !== nothing
            remove(subscription.listener_node)
            subscription.listener_node = nothing
        end
        return nothing
    end
    
    Base.show(io::IO, ::SubjectSubscription) = print(io, "SubjectSubscription()")
    
    ##
    
    """
        SubjectFactory(scheduler::H) where { H <: Rocket.AbstractScheduler }
    
    A base subject factory that creates an instance of Subject with specified scheduler.
    
    See also: [`AbstractSubjectFactory`](@ref), [`Subject`](@ref)
    """
    struct SubjectFactory{ H <: Rocket.Rocket.AbstractScheduler } <: AbstractSubjectFactory
        scheduler :: H
    end
    
    create_subject(::Type{L}, factory::SubjectFactory) where L = Subject(L, scheduler = similar(factory.scheduler))
    
    Base.show(io::IO, ::SubjectFactory{H}) where H = print(io, "SubjectFactory($H)")
    





















end#StructsManag


