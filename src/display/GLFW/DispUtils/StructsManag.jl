
"""
utilities for dealing data structs like FullScrollableDat or SingleSliceDat
"""
module StructsManag
using Setfield, ColorTypes
using ..ForDisplayStructs, ..DataStructs
export getThreeDims, addToforUndoVector, cartTwoToThree, getHeightToWidthRatio, threeToTwoDimm, modSlice!, threeToTwoDimm, modifySliceFull!, getSlicesNumber, getMainVerticies, getTextureCoordinatesFromScreen, applyZoomPan

"""
Calculates texture coordinates from screen coordinates, accounting for viewport zoom/pan/padding.
"""
function getTextureCoordinatesFromScreen(x::Real, y::Real, calcDimsStruct::CalcDimsStruct, actualW::Float64, actualH::Float64)
    viewportW = Float64(calcDimsStruct.windowWidth)
    viewportH = Float64(calcDimsStruct.windowHeight)
    
    glX = (x * 2.0 / viewportW) - 1.0
    glY = ((actualH - y) * 2.0 / viewportH) - 1.0
    
    verts = calcDimsStruct.mainImageQuadVert
    glLeft   = Float64(min(verts[17], verts[25]))
    glRight  = Float64(max(verts[1], verts[9]))
    glBottom = Float64(min(verts[10], verts[18]))
    glTop    = Float64(max(verts[2], verts[26]))
    
    s = clamp((glX - glLeft) / (glRight - glLeft), 0.0, 1.0)
    t = clamp((glY - glBottom) / (glTop - glBottom), 0.0, 1.0)
    
    texW = Float64(calcDimsStruct.imageTextureWidth)
    texH = Float64(calcDimsStruct.imageTextureHeight)
    zoom = Float64(calcDimsStruct.zoom)
    
    # Reverse the data-level zoom/pan transform to get original voxel coords
    viewW = texW / zoom
    viewH = texH / zoom
    cx = texW / 2.0 + Float64(calcDimsStruct.panX) * texW
    cy = texH / 2.0 + Float64(calcDimsStruct.panY) * texH
    x1 = clamp(cx - viewW / 2.0, 1.0, texW - viewW + 1.0)
    y1 = clamp(cy - viewH / 2.0, 1.0, texH - viewH + 1.0)
    
    texX = clamp(round(Int, x1 + s * viewW), 1, Int(texW))
    texY = clamp(round(Int, y1 + t * viewH), 1, Int(texH))
    
    return texX, texY
end

"""
Apply zoom and pan to a 2D slice by cropping a subregion and upscaling via nearest-neighbor.
zoom: zoom factor (1.0 = no zoom, 2.0 = 2x zoom)
panX, panY: pan offsets in normalized coordinates (-0.5 to 0.5)
"""
function applyZoomPan(slice::AbstractMatrix{T}, zoom::Float32, panX::Float32, panY::Float32) where T
    if zoom <= 1.0f0 && panX == 0.0f0 && panY == 0.0f0
        return slice  # No-op fast path
    end
    
    h, w = size(slice)
    z = max(0.1f0, zoom)
    
    # Center in source coordinates: panning shifts the sample center
    # panY shifts horizontal center (columns w), panX shifts vertical center (rows h)
    cx = Float64(w) / 2.0 + (Float64(panY) * Float64(w))
    cy = Float64(h) / 2.0 + (Float64(panX) * Float64(h))
    
    result = fill(zero(T), h, w)
    
    for j in 1:w, i in 1:h
        vi = (Float64(i) - 0.5 - Float64(h) / 2.0) / Float64(z)
        vj = (Float64(j) - 0.5 - Float64(w) / 2.0) / Float64(z)
        
        si = round(Int, cy + vi + 0.5)
        sj = round(Int, cx + vj + 0.5)
        
        if si >= 1 && si <= h && sj >= 1 && sj <= w
            result[i, j] = slice[si, sj]
        end
    end
    
    return result
end
```@doc
given two dim dat it sets points in given coordinates in given slice to given value
coords - coordinates in a plane of chosen slice to modify
value - value to set for given points
return reference to modified slice
```
function modSlice!(data::TwoDimRawDat{T}, coords::Vector{CartesianIndex{2}}, value::T) where {T}
  data.dat[coords] .= value
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
function threeToTwoDimm(typ::Type{T}, sliceInner::Int, sliceDim::Int, threedimDat::ThreeDimRawDat{T})::TwoDimRawDat{T} where {T}

  maxSlice = size(threedimDat.dat)[sliceDim]
  slice = sliceInner
  if (sliceInner > maxSlice)
    slice = maxSlice
  end
  return TwoDimRawDat{T}(typ, threedimDat.name, selectdim(threedimDat.dat, sliceDim, slice))
end#ThreeToTwoDimm


modifySliceFull!Str = """
 modifies given slice in given coordinates of given data - queried by name
 data - full data we work on and modify
 coords - coordinates in a plane of chosen slice to modify (so list of x and y coords)
 value - value to set for given points
 return reference to modified slice
 """
@doc modifySliceFull!Str
function modifySliceFull!(data::FullScrollableDat, slice::Int, coords::Vector{CartesianIndex{2}}, name::String, value)

  threeDimDat = data.nameIndexes[name] |>
                (ind) -> data.dataToScroll[ind]
  if (typeof(value) != threeDimDat.type)
    throw(DomainError(value, "supplied value should be of compatible type - $(threeDimDat.type )"))
  end #if

  return threeToTwoDimm(threeDimDat.type, slice, data.dimensionToScroll, threeDimDat) |>
         (twoDimDat) -> modSlice!(twoDimDat, coords, value)
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
function getHeightToWidthRatio(calcDim::CalcDimsStruct, dataToScrollDims::DataToScrollDims)::CalcDimsStruct
  toSelect = filter(it -> it != dataToScrollDims.dimensionToScroll, [1, 2, 3])# will be used to get texture width and height

  return setproperties(calcDim, (imageTextureWidth=dataToScrollDims.imageSize[toSelect[1]], imageTextureHeight=dataToScrollDims.imageSize[toSelect[2]], heightToWithRatio=dataToScrollDims.voxelSize[toSelect[2]] / dataToScrollDims.voxelSize[toSelect[1]], textTextureZeros=calcDim.textTextureZeros
  ))
end#getHeightToWidthRatio



```@doc
Based on DataToScrollDims ,2 dim cartesian coordinate and  slice number it gives 3 dimensional coordinate of mouse position
```
function cartTwoToThree(dataToScrollDims::DataToScrollDims, sliceNumber::Int, cartIn::CartesianIndex{2})::CartesianIndex{3}
  dimToScroll = dataToScrollDims.dimensionToScroll
  toSelect = filter(it -> it != dimToScroll, [1, 2, 3])# will be used to get texture width and height
  resArr = [1, 1, 1]


  resArr[toSelect[1]] = cartIn[1]
  resArr[toSelect[2]] = cartIn[2]

  resArr[dimToScroll] = Int64(sliceNumber)

  return CartesianIndex(resArr[1], resArr[2], resArr[3])
end#cartTwoToThree




```@doc
Given function and actor it passes the function to forUndoVector -
   in case the length of the vector is too big the last element woill be removed
```
function addToforUndoVector(stateObject::StateDataFields, fun)

  push!(stateObject.forUndoVector, fun)

  if (length(stateObject.forUndoVector) > stateObject.maxLengthOfForUndoVector)
    popfirst!(stateObject.forUndoVector)
  end

end#addToforUndoVector


```@doc
utility function to create series of ThreeDimRawDat from list of tuples where
first entry is String and second entry is 3 dimensional array with data
```
function getThreeDims(list)
  return map(tupl -> ThreeDimRawDat{typeof(tupl[2][1])}(typeof(tupl[2][1]), tupl[1], tupl[2]), list)
end#getThreeDims


# parameter_type(x::TextureSpec) = parameter_type(typeof(x))



"""
Calculates OpenGL quad vertices (32 floats: 4 vertices * (X, Y, Z, R, G, B, U, V)) for a panel
in `SingleImage`, `MultiImage` (Compare Mode), or `QuadImage` (4-view layout).

Guarantees exact physical anatomical aspect ratio preservation:
  ratio_desired = heightToWithRatio * (imageTextureHeight / imageTextureWidth) = (H_mm / W_mm)

Within the panel's allocated NDC bounding box [x_min, x_max] x [y_min, y_max]:
- If ratio_actual > ratio_desired (panel is taller than image):
    scale_y = ratio_desired / ratio_actual, scale_x = 1.0 (fills width, centers vertically)
- If ratio_actual <= ratio_desired (panel is wider than image):
    scale_x = ratio_actual / ratio_desired, scale_y = 1.0 (fills height, centers horizontally, zero top gap)

Returns enriched CalcDimsStruct.
"""
function getMainVerticies(calcDimStruct::CalcDimsStruct, displayMode::DisplayMode, imagePos::Int64)::CalcDimsStruct
  # 0. If QuadImage and imagePos > 4, panel is hidden (only active as :RightHalf in compare mode)
  if displayMode == QuadImage && imagePos > 4
    res = zeros(Float32, 32)
    return setproperties(calcDimStruct, (
      mainImageQuadVert = res,
      mainQuadVertSize  = sizeof(res),
      widthCorr         = 0.0f0,
      heightCorr        = 0.0f0,
      imagePos          = imagePos
    ))
  end

  total_w = Float64(calcDimStruct.windowWidth)
  total_h = Float64(calcDimStruct.windowHeight)
  frac = Float64(calcDimStruct.fractionOfMainIm)

  local panel_w::Float64
  local panel_h::Float64
  local x_min::Float64, x_max::Float64
  local y_min::Float64, y_max::Float64

  ndc_right_edge = -1.0 + 2.0 * frac

  if displayMode == SingleImage
    panel_w = total_w * frac
    panel_h = total_h
    x_min = -1.0
    x_max = ndc_right_edge
    y_min = -1.0
    y_max = 1.0
  elseif displayMode == MultiImage
    # 2 side-by-side panels
    panel_w = (total_w * frac) / 2.0
    panel_h = total_h
    ndc_mid_x = -1.0 + frac
    if imagePos == 1  # Left Panel
      x_min = -1.0
      x_max = ndc_mid_x
    else              # Right Panel
      x_min = ndc_mid_x
      x_max = ndc_right_edge
    end
    y_min = -1.0
    y_max = 1.0
  else # QuadImage
    # 4 quadrants: Top-Left (1), Top-Right (2), Bottom-Left (3), Bottom-Right (4)
    panel_w = (total_w * frac) / 2.0
    panel_h = total_h / 2.0
    ndc_mid_x = -1.0 + frac
    if imagePos == 1 || imagePos == 3  # Left column
      x_min = -1.0
      x_max = ndc_mid_x
    else                              # Right column
      x_min = ndc_mid_x
      x_max = ndc_right_edge
    end
    if imagePos == 1 || imagePos == 2  # Top row
      y_min = 0.0
      y_max = 1.0
    else                              # Bottom row
      y_min = -1.0
      y_max = 0.0
    end
  end

  # 2. Desired anatomical physical aspect ratio (height_mm / width_mm)
  tex_w = max(1.0, Float64(calcDimStruct.imageTextureWidth))
  tex_h = max(1.0, Float64(calcDimStruct.imageTextureHeight))
  ratio_desired = Float64(calcDimStruct.heightToWithRatio) * (tex_h / tex_w)

  # 3. Actual aspect ratio of the allocated panel (pixel height / pixel width)
  ratio_actual = panel_h / max(1.0, panel_w)

  # 4. Aspect-ratio preserving scale factors within the panel
  local scale_x::Float64, scale_y::Float64
  local widthCorr::Float64, heightCorr::Float64

  if ratio_actual > ratio_desired
    scale_y = ratio_desired / ratio_actual
    scale_x = 1.0
    heightCorr = 1.0 - scale_y
    widthCorr = 0.0
  else
    scale_x = ratio_actual / ratio_desired
    scale_y = 1.0
    widthCorr = 1.0 - scale_x
    heightCorr = 0.0
  end

  # 5. Compute NDC coordinates centered in panel [x_min, x_max] x [y_min, y_max]
  x_center = (x_min + x_max) / 2.0
  half_span_x = ((x_max - x_min) / 2.0) * scale_x
  left_x  = Float32(x_center - half_span_x)
  right_x = Float32(x_center + half_span_x)

  y_center = (y_min + y_max) / 2.0
  half_span_y = ((y_max - y_min) / 2.0) * scale_y
  bottom_y = Float32(y_center - half_span_y)
  top_y    = Float32(y_center + half_span_y)

  # 6. Build 32-element OpenGL vertex array (4 vertices * 8 floats)
  # Layout: X, Y, Z, R, G, B, U, V
  res = Float32[
    right_x, top_y,    0.0f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0, 1.0f0,  # top right
    right_x, bottom_y, 0.0f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0, 0.0f0,  # bottom right
    left_x,  bottom_y, 0.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0, 0.0f0,  # bottom left
    left_x,  top_y,    0.0f0, 1.0f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0   # top left
  ]

  windowWidthCorr = Int32(round((widthCorr / 2.0) * panel_w))
  windowHeightCorr = Int32(round((heightCorr / 2.0) * panel_h))

  return setproperties(calcDimStruct, (
    widthCorr                 = Float32(widthCorr),
    heightCorr                = Float32(heightCorr),
    mainImageQuadVert         = res,
    mainQuadVertSize          = sizeof(res),
    windowWidthCorr           = windowWidthCorr,
    windowHeightCorr          = windowHeightCorr,
    corrected_width           = panel_w,
    correCtedWindowQuadHeight = Int32(round(panel_h * scale_y)),
    correCtedWindowQuadWidth  = Int32(round(panel_w * scale_x)),
    imagePos                  = imagePos
  ))
end #getMainVerticies


# function correctRatios(texel_ratio, heightToWidthRatio, windowHeight, corrected_width)




#   heightCorr = 0.0
#   widthCorr = 0.0

#   # Do not modify below this line
#   restSpaceHeight = 1 - heightCorr
#   restSpaceWidth = 1 - widthCorr
#   multipliedHeight = restSpaceHeight * windowHeight #and why are we multiplying with the calcDimStruct.windowHeight specifically?
#   mulitipliedWidth = restSpaceWidth * corrected_width #can u explain why me multiply with corrected_width here?
#   recalc_texel_ratio = multipliedHeight / mulitipliedWidth

#   return restSpaceHeight, restSpaceWidth, multipliedHeight, mulitipliedWidth, recalc_texel_ratio
# end


####################### adapted from https://github.com/biaslab/Rocket.jl/blob/8aa557c90717bed9d24e36cc6b147dcc076d6b67/src/schedulers/async.jl










# import Base: show, similar

# """
#     AsyncScheduler_spawned

# `AsyncScheduler_spawned` executes scheduled actions asynchronously and uses `Channel` object to order different actions on a single asynchronous task
# """
# struct AsyncScheduler_spawned{N} <: Rocket.AbstractScheduler end

# Base.show(io::IO, ::AsyncScheduler_spawned) = print(io, "AsyncScheduler_spawned()")

# function AsyncScheduler_spawned(size::Int = typemax(Int))
#     return AsyncScheduler_spawned{size}()
# end

# Base.similar(::AsyncScheduler_spawned{N}) where N = AsyncScheduler_spawned{N}()

# makeinstance(::Type{D}, ::AsyncScheduler_spawned{N}) where { D, N } = AsyncScheduler_spawnedInstance{D}(N)

# instancetype(::Type{D}, ::Type{<:AsyncScheduler_spawned}) where D = AsyncScheduler_spawnedInstance{D}

# struct AsyncScheduler_spawnedDataMessage{D}
#     data :: D
# end

# struct AsyncScheduler_spawnedErrorMessage
#     err
# end

# struct AsyncScheduler_spawnedCompleteMessage end

# const AsyncScheduler_spawnedMessage{D} = Union{AsyncScheduler_spawnedDataMessage{D}, AsyncScheduler_spawnedErrorMessage, AsyncScheduler_spawnedCompleteMessage}

# mutable struct AsyncScheduler_spawnedInstance{D}
#     channel        :: Channel{AsyncScheduler_spawnedMessage{D}}
#     isunsubscribed :: Bool
#     subscription   :: Teardown

#     AsyncScheduler_spawnedInstance{D}(size::Int = typemax(Int)) where D = begin
#         return new(Channel{AsyncScheduler_spawnedMessage{D}}(size, spawn=true), false, voidTeardown)
#     end
# end

# isunsubscribed(instance::AsyncScheduler_spawnedInstance) = instance.isunsubscribed
# getchannel(instance::AsyncScheduler_spawnedInstance) = instance.channel

# function dispose(instance::AsyncScheduler_spawnedInstance)
#     if !isunsubscribed(instance)
#         instance.isunsubscribed = true
#         close(instance.channel)
#         @async begin
#             unsubscribe!(instance.subscription)
#         end
#     end
# end

# function __process_channeled_message(instance::AsyncScheduler_spawnedInstance{D}, message::AsyncScheduler_spawnedDataMessage{D}, actor) where D
#     on_next!(actor, message.data)
# end

# function __process_channeled_message(instance::AsyncScheduler_spawnedInstance, message::AsyncScheduler_spawnedErrorMessage, actor)
#     on_error!(actor, message.err)
#     dispose(instance)
# end

# function __process_channeled_message(instance::AsyncScheduler_spawnedInstance, message::AsyncScheduler_spawnedCompleteMessage, actor)
#     on_complete!(actor)
#     dispose(instance)
# end

# struct AsyncScheduler_spawnedSubscription{ H <: AsyncScheduler_spawnedInstance } <: Teardown
#     instance :: H
# end

# Base.show(io::IO, ::AsyncScheduler_spawnedSubscription) = print(io, "AsyncScheduler_spawnedSubscription()")

# as_teardown(::Type{ <: AsyncScheduler_spawnedSubscription}) = UnsubscribableTeardownLogic()

# function on_unsubscribe!(subscription::AsyncScheduler_spawnedSubscription)
#     dispose(subscription.instance)
#     return nothing
# end

# function scheduled_subscription!(source, actor, instance::AsyncScheduler_spawnedInstance)
#     subscription = AsyncScheduler_spawnedSubscription(instance)

#     channeling_task = @async begin
#         while !isunsubscribed(instance)
#             message = take!(getchannel(instance))
#             if !isunsubscribed(instance)
#                 __process_channeled_message(instance, message, actor)
#             end
#         end
#     end

#     subscription_task = @async begin
#         if !isunsubscribed(instance)
#             tmp = on_subscribe!(source, actor, instance)
#             if !isunsubscribed(instance)
#                 subscription.instance.subscription = tmp
#             else
#                 unsubscribe!(tmp)
#             end
#         end
#     end

#     bind(getchannel(instance), channeling_task)

#     return subscription
# end

# function scheduled_next!(actor, value::D, instance::AsyncScheduler_spawnedInstance{D}) where { D }
#     put!(getchannel(instance), AsyncScheduler_spawnedDataMessage{D}(value))
# end

# function scheduled_error!(actor, err, instance::AsyncScheduler_spawnedInstance)
#     put!(getchannel(instance), AsyncScheduler_spawnedErrorMessage(err))
# end

# function scheduled_complete!(actor, instance::AsyncScheduler_spawnedInstance)
#     put!(getchannel(instance), AsyncScheduler_spawnedCompleteMessage())
# end




# import Base: show, similar

# ##

# struct SubjectListener{I}
#     schedulerinstance :: I
#     actor
# end

# Base.show(io::IO, ::SubjectListener) = print(io, "SubjectListener()")


# """
#     Subject(::Type{D}; scheduler::H = AsapScheduler())

# A Subject is a special type of Observable that allows values to be multicasted to many Observers. Subjects are like EventEmitters.
# Every Subject is an Observable and an Actor. You can subscribe to a Subject, and you can call `next!` to feed values as well as `error!` and `complete!`.

# Note: By convention, every actor subscribed to a Subject observable is not allowed to throw exceptions during `next!`, `error!` and `complete!` calls.
# Doing so would lead to undefined behaviour. Use `safe()` operator to bypass this rule.

# See also: [`SubjectFactory`](@ref), [`ReplaySubject`](@ref), [`BehaviorSubject`](@ref), [`safe`](@ref)
# """
# mutable struct Subject{D, H, I} <: Rocket.AbstractSubject{D}
#     listeners   :: Rocket.List{SubjectListener{I}}
#     scheduler   :: H
#     isactive    :: Bool
#     iscompleted :: Bool
#     isfailed    :: Bool
#     lasterror   :: Any

#     Subject{D, H, I}(scheduler::H) where { D, H <: Rocket.AbstractScheduler, I } = new(Rocket.List(SubjectListener{I}), scheduler, true, false, false, nothing)
# end

# function Subject(::Type{D}; scheduler::H = AsapScheduler()) where { D, H <: Rocket.AbstractScheduler }
#     return Subject{D, H, instancetype(D, H)}(scheduler)
# end


# ##
# function convert(::Rocket.Subject, subj::Subject)
#     return subj
# end


# Base.show(io::IO, ::Subject{D, H}) where { D, H } = print(io, "Subject($D, $H)")

# Base.similar(subject::Subject{D, H}) where { D, H } = Subject(D; scheduler = similar(subject.scheduler))

# ##

# isactive(subject::Subject)    = subject.isactive
# iscompleted(subject::Subject) = subject.iscompleted
# isfailed(subject::Subject)    = subject.isfailed
# lasterror(subject::Subject)   = subject.lasterror

# setinactive!(subject::Subject)       = subject.isactive    = false
# setcompleted!(subject::Subject)      = subject.iscompleted = true
# setfailed!(subject::Subject)         = subject.isfailed    = true
# setlasterror!(subject::Subject, err) = subject.lasterror   = err

# ##

# function on_next!(subject::Subject{D, H, I}, data::D) where { D, H, I }
#     for listener in subject.listeners
#         scheduled_next!(listener.actor, data, listener.schedulerinstance)
#     end
# end

# function on_error!(subject::Subject, err)
#     if isactive(subject)
#         setinactive!(subject)
#         setfailed!(subject)
#         setlasterror!(subject, err)
#         for listener in subject.listeners
#             scheduled_error!(listener.actor, err, listener.schedulerinstance)
#         end
#         empty!(subject.listeners)
#     end
# end

# function on_complete!(subject::Subject)
#     if isactive(subject)
#         setinactive!(subject)
#         setcompleted!(subject)
#         for listener in subject.listeners
#             scheduled_complete!(listener.actor, listener.schedulerinstance)
#         end
#         empty!(subject.listeners)
#     end
# end

# ##

# function on_subscribe!(subject::Subject{D}, actor) where { D }
#     if isfailed(subject)
#         error!(actor, lasterror(subject))
#         return SubjectSubscription(nothing)
#     elseif iscompleted(subject)
#         complete!(actor)
#         return SubjectSubscription(nothing)
#     else
#         instance = makeinstance(D, subject.scheduler)
#         return scheduled_subscription!(subject, actor, instance)
#     end
# end

# function on_subscribe!(subject::Subject, actor, instance)
#     listener      = SubjectListener(instance, actor)
#     listener_node = pushnode!(subject.listeners, listener)
#     return SubjectSubscription(listener_node)
# end

# ##

# mutable struct SubjectSubscription <: Rocket.Teardown
#     listener_node :: Union{Nothing, Rocket.ListNode}
# end

# as_teardown(::Type{ <: SubjectSubscription }) = UnsubscribableTeardownLogic()

# function on_unsubscribe!(subscription::SubjectSubscription)
#     if subscription.listener_node !== nothing
#         remove(subscription.listener_node)
#         subscription.listener_node = nothing
#     end
#     return nothing
# end

# Base.show(io::IO, ::SubjectSubscription) = print(io, "SubjectSubscription()")

# ##

# """
#     SubjectFactory(scheduler::H) where { H <: Rocket.AbstractScheduler }

# A base subject factory that creates an instance of Subject with specified scheduler.

# See also: [`AbstractSubjectFactory`](@ref), [`Subject`](@ref)
# """
# struct SubjectFactory{ H <: Rocket.Rocket.AbstractScheduler } <: AbstractSubjectFactory
#     scheduler :: H
# end

# create_subject(::Type{L}, factory::SubjectFactory) where L = Subject(L, scheduler = similar(factory.scheduler))

# Base.show(io::IO, ::SubjectFactory{H}) where H = print(io, "SubjectFactory($H)")






















end#StructsManag


