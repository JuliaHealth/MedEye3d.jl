"""
Main module controlling displaying segmentations image and data

"""
module SegmentationDisplay
export loadRegisteredImages, displayImage, coordinateDisplay, passDataForScrolling, close_window, set_window_title, resize_window, GLOBAL_OPENGL_LOCK, synchronized_makie_renderloop
using Dates
using ColorTypes, MedImages, GLFW, Dictionaries, Logging, Setfield, FreeTypeAbstraction, Statistics, Observables, FileIO
using ..ForDisplayStructs, ..distinctColorsSaved
using ..VulkanBackend: VulkanContext, VulkanPipeline, VulkanRender, VulkanTextures, VulkanScreenshot, VulkanShaders, VulkanBuffers, VulkanStaging
using Vulkan
using ..ReactingToInput, ..ReactToScroll, ..DataStructs, ..StructsManag
using ..ReactOnKeyboard, ..ReactOnMouseClickAndDrag, ..DisplayDataManag
using ..PrepareWindow, ..PrepareWindowHelpers, ..TextureManag, ..OpenGLDisplayUtils, ..Uniforms, ..DisplayWords
using ..MakieEvents
include("MakieEventHandlers.jl")
using .MakieEventHandlers

const GLOBAL_OPENGL_LOCK = ReentrantLock()

function synchronized_makie_renderloop(screen)
    # Find GLMakie from loaded modules — it may not be in Main scope
    # (e.g., when imported inside a submodule like LesionMetadataWindow)
    local GLMakie_mod = nothing
    if isdefined(Main, :GLMakie)
        GLMakie_mod = Main.GLMakie
    else
        for (k, v) in Base.loaded_modules
            if string(k.name) == "GLMakie"
                GLMakie_mod = v
                break
            end
        end
    end
    if GLMakie_mod === nothing
        @warn "synchronized_makie_renderloop: GLMakie not found in loaded modules, renderloop disabled"
        return
    end
    GLMakie = GLMakie_mod
    Makie = isdefined(GLMakie, :Makie) ? GLMakie.Makie : nothing
    if Makie === nothing
        @warn "synchronized_makie_renderloop: Makie not found via GLMakie"
        return
    end
    @info "synchronized_makie_renderloop: STARTED (GLMakie found via loaded_modules)"
    tick_state = Ref(Makie.UnknownTickState)
    loop_count = Ref(0)
    while isopen(screen) && !screen.stop_renderloop[]
        if isdefined(GLMakie, :GLAbstraction)
            lock(GLOBAL_OPENGL_LOCK) do
                GLMakie.GLAbstraction.with_context(screen.glscreen) do
                    GLMakie.pollevents(screen, tick_state[])
                    GLMakie.poll_updates(screen)
                    if !screen.config.pause_renderloop && GLMakie.requires_update(screen)
                        tick_state[] = Makie.RegularRenderTick
                        GLMakie.render_frame(screen)
                        GLFW.SwapBuffers(GLMakie.to_native(screen))
                    else
                        tick_state[] = ifelse(screen.config.pause_renderloop, Makie.PausedRenderTick, Makie.SkippedRenderTick)
                    end
                end
            end
        end
        loop_count[] += 1
        if loop_count[] % 600 == 0
            @info "RENDERLOOP alive: iteration $(loop_count[])"
        end
        GC.safepoint()
        sleep(screen.timer)
    end
    @info "synchronized_makie_renderloop: STOPPED"
end

# switch_gl_context! removed — Vulkan doesn't use GL context switching

function reactToResizeWindow(data::ResizeWindowEvent, stateObjects::Vector{StateDataFields})
    if data.width > 0 && data.height > 0
        for state in stateObjects
            state.calcDimsStruct.windowWidth = Int64(data.width)
            state.calcDimsStruct.windowHeight = Int64(data.height)
            state.calcDimsStruct.avWindWidtForMain = Int32(round(data.width * state.calcDimsStruct.fractionOfMainIm))
            state.calcDimsStruct.avWindHeightForMain = Int32(data.height)
            state.calcDimsStruct.avMainImRatio = Float32(data.height / max(1, state.calcDimsStruct.avWindWidtForMain))
            
            try
                state.calcDimsStruct = StructsManag.getMainVerticies(state.calcDimsStruct, state.displayMode, state.calcDimsStruct.imagePos)
            catch e
                @warn "Error updating quad vertices on window resize: $e"
            end
        end
        # Recreate Vulkan swapchain for new size
        try
            obj = stateObjects[1].mainForDisplayObjects
            if obj.vulkanCtx !== nothing
                VulkanContext.recreate_swapchain!(obj.vulkanCtx, data.width, data.height)
            end
        catch e
            @warn "Error recreating Vulkan swapchain: $e"
        end
    end
end

function reactToSetWindowTitle(data::SetWindowTitleEvent, stateObjects::Vector{StateDataFields})
    if !isempty(stateObjects) && stateObjects[1].mainForDisplayObjects.window.handle != C_NULL
        GLFW.SetWindowTitle(stateObjects[1].mainForDisplayObjects.window, data.title)
    end
end

#  do not copy it into the consumer function
"""
configuring consumer function on_next! function using multiple dispatch mechanism in order to connect input to proper functions
"""
on_next!(stateObjects::Vector{StateDataFields}, data::Int64) = reactToScroll(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ScrollZoomEvent) = ReactToScroll.reactToScrollZoom(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::forDisplayObjects) = setUpMainDisplay(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ForWordsDispStruct) = setUpWordsDisplay(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::CalcDimsStruct) = setUpCalcDimsStruct(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::valueForMasToSetStruct) = setUpvalueForMasToSet(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::FullScrollableDat) = setUpForScrollData(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::SingleSliceDat) = updateSingleImagesDisplayedSetUp(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::Vector{MouseStruct}) = react_to_draw(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::MouseStruct) = reactToMouseDrag(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::DoubleClickEvent) = reactToDoubleClick(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::KeyInputFields) = reactToKeyInput(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::DisplayedVoxels) = retrieveVoxelArray(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::CustomDisplayedVoxels) = depositVoxelArray(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ChangePlaneEvent) = reactToChangePlane(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::CompareTimePointsEvent) = reactToCompareTimePoints(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ShowSingleLesionEvent) = reactToShowSingleLesion(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::WindowingEvent) = reactToWindowing(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::PaintValEvent) = reactToPaintVal(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ChangeBrushSizeEvent) = reactToChangeBrushSize(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::SyncLesionEvent) = reactToSyncLesion(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ChangeTimePointEvent) = reactToChangeTimePoint(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ToggleLesionEvent) = reactToToggleLesion(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::PetBlendEvent) = reactToPetBlend(data, stateObjects)

on_next!(stateObjects::Vector{StateDataFields}, data::RefreshListEvent) = reactToRefreshList(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::AddAutoPetEvent) = reactToAddAutoPet(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::AIInferenceResultEvent) = reactToAIInferenceResult(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::SyncMissingEvent) = reactToSyncMissing(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::GenManualEvent) = reactToGenManual(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::MapLinkEvent) = reactToMapLink(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::AutoRunPreprocessEvent) = reactToAutoRunPreprocess(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::RunPreprocessEvent) = reactToRunPreprocess(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ShowBoneMaskEvent) = reactToShowBoneMask(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ShowMaskLayerEvent) = reactToShowMaskLayer(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::SaveMRBEvent) = reactToSaveMRB(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::CloseWindowEvent) = nothing
on_next!(stateObjects::Vector{StateDataFields}, data::ResizeWindowEvent) = reactToResizeWindow(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::SetWindowTitleEvent) = reactToSetWindowTitle(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ToggleMoveLesionModeEvent) = reactToToggleMoveLesionMode(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::AIStatusUpdateEvent) = (MakieEventHandlers.ai_status_text[] = data.text)
on_next!(stateObjects::Vector{StateDataFields}, data::BoneSubsegResultEvent) = MakieEventHandlers.reactToBoneSubsegResult(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::ScreenshotEvent) = reactToScreenshot(data, stateObjects)
on_error!(stateObjects::Vector{StateDataFields}, err) = error(err)
on_complete!(stateObjects::Vector{StateDataFields}) = ""

"""
    reactToScreenshot(event::ScreenshotEvent, stateObjects)

Captures the current OpenGL framebuffer and saves it as a PNG file.
Must be called on the GL thread (via on_next! dispatch) after rendering.
Signals event.done_channel when complete.
"""
function reactToScreenshot(event::ScreenshotEvent, stateObjects::Vector{StateDataFields})
    try
        obj = stateObjects[1].mainForDisplayObjects
        if obj.vulkanCtx !== nothing
            img = VulkanScreenshot.capture_screenshot(obj.vulkanCtx)
            mkpath(dirname(event.path))
            FileIO.save(event.path, img)
            println("Screenshot saved: $(event.path)")
            flush(stdout)
            put!(event.done_channel, true)
        else
            @warn "Screenshot failed: no Vulkan context"
            put!(event.done_channel, false)
        end
    catch e
        @warn "Screenshot failed: $e"
        put!(event.done_channel, false)
    end
end

"""
is used to pass into the actor data that will be used for scrolling
onScrollData - struct holding between others list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)
"""
function passDataForScrolling(
    mainMedEye3dInstance::MainMedEye3d,
    onScrollData::Union{FullScrollableDat,Vector{FullScrollableDat}}
)
    # """
    # put data onto the channel, matching types with on_next.

    # put the data in the onScrollData which screen need to be
    # """

    #modify here the data for passing onto the channel
    if typeof(onScrollData) == FullScrollableDat
        put!(mainMedEye3dInstance.channel, onScrollData)
    elseif typeof(onScrollData) == Vector{FullScrollableDat}
        foreach(enumerate(onScrollData)) do (index, onScrollDataInstance)
            onScrollDataInstance.imagePos = index
            put!(mainMedEye3dInstance.channel, onScrollDataInstance)
        end
    end

    #we can put the vector of onScrollData into the channel.
end



"""
is using the actor that is instantiated in this module and connects it to GLFW context
by invoking appropriate registering functions and passing to it to the main Actor controlling input
"""
function registerInteractions(
    window::GLFW.Window,
    mainMedEye3dInstance::MainMedEye3d,
    calcDimStruct::Union{CalcDimsStruct,Vector{CalcDimsStruct}}
)
    if typeof(calcDimStruct) == CalcDimsStruct
        subscribeGLFWtoActor(window, mainMedEye3dInstance, calcDimStruct)
    elseif typeof(calcDimStruct) == Vector{CalcDimsStruct}
        foreach(calcDimStruct) do currentCalcDim
            subscribeGLFWtoActor(window, mainMedEye3dInstance, currentCalcDim)
        end
    end
    # subscribeGLFWtoActor(window, mainMedEye3dInstance, calcDimStruct)
end

"""
Preparing ForWordsDispStruct that will be needed for proper displaying of texts
    numberOfActiveTextUnits - number of textures already used - so we we will know what is still free
    fragment_shader_words - reference to fragment shader used to display text
    vbo_words - vertex buffer object used to display words
    shader_program_words - shader program associated with displaying text
    widthh, heightt - size of the texture - the bigger the higher resolution, but higher computation cost

return prepared for displayStruct
"""
function prepareForDispStruct(
    numberOfActiveTextUnits::Int,
    fragment_shader_words::UInt32,
    vbo_words::Base.RefValue{UInt32},
    shader_program_words::UInt32,
    window,
    widthh::Int32=Int32(1),
    heightt::Int32=Int32(1),
    forDispObj::forDisplayObjects=forDisplayObjects()
)::ForWordsDispStruct

    res = ForWordsDispStruct(
        fontFace=FTFont(joinpath(dirname(dirname(pathof(FreeTypeAbstraction))), "test", "hack_regular.ttf")), textureSpec=createTextureForWords(numberOfActiveTextUnits, widthh, heightt, getProperGL_TEXTURE(numberOfActiveTextUnits + 1)), fragment_shader_words=fragment_shader_words, vbo_words=vbo_words, shader_program_words=shader_program_words
    )

    return res
end#prepereForDispStruct



"""
Returns the display mode of the visualizer
"""
function getDisplayMode(listOfTextSpecs::Union{Vector{TextureSpec},Vector{Vector{TextureSpec}}}, quadView::Bool=false)::DisplayMode
    if typeof(listOfTextSpecs) == Vector{TextureSpec}
        return SingleImage
    elseif typeof(listOfTextSpecs) == Vector{Vector{TextureSpec}}
        if length(listOfTextSpecs) == 1 && !quadView
            return SingleImage
        elseif length(listOfTextSpecs) >= 4 || quadView
            return QuadImage
        else
            return MultiImage
        end
    end
end



"""
Carries out the initialization of shader and buffers for
SuperVoxels — stub, OpenGL rendering removed.
"""
function initializeSupervoxels(vertex_shader, vao, ebo, vboVector, allSupervoxels)
    return (GlShaderAndBufferFields(), GlShaderAndBufferFields())
end


"""
Carries out the initialization of shader and buffers for
crosshair — stub, OpenGL rendering removed.
"""
function createCrosshairFields(vertex_shader)
    return GlShaderAndBufferFields()
end

function initializeCrosshair(vertex_shader, vao, ebo, vboVector, fragment_shader_words, vbo_words, shader_program_words)
    crosshairs = [GlShaderAndBufferFields() for _ in 1:length(vboVector)]
    mainRects = [GlShaderAndBufferFields() for _ in 1:length(vboVector)]
    textFields = GlShaderAndBufferFields()
    return (crosshairs, mainRects, textFields)
end


"""
coordinating displaying - sets needed constants that are storeds in  forDisplayConstants; and configures interactions from GLFW events
listOfTextSpecs - holds required data needed to initialize textures
keeps also references to needed ..Uniforms etc.
windowWidth::Int,windowHeight::Int - GLFW window dimensions
fractionOfMainIm - how much of width should be taken by the main image
heightToWithRatio - needed for proper display of main texture - so it would not be stretched ...
"""
function coordinateDisplay(
    listOfTextSpecsPrim::Union{Vector{TextureSpec},Vector{Vector{TextureSpec}}},
    fractionOfMainIm::Float32,
    dataToScrollDims::Union{DataToScrollDims,Vector{DataToScrollDims}}=DataToScrollDims(),
    spacing::Union{Vector{Tuple{Float64,Float64,Float64}},Vector{Vector{Tuple{Float64,Float64,Float64}}}}=Vector{Tuple{Float64,Float64,Float64}}(),
    origin::Union{Vector{Tuple{Float64,Float64,Float64}},Vector{Vector{Tuple{Float64,Float64,Float64}}}}=Vector{Tuple{Float64,Float64,Float64}}(),
    svVertAndInd::Dict{String,Vector}=Dict{String,Vector}("supervoxel_vertices" => [], "supervoxel_indices" => []),
    allSupervoxels::Dict{Int,Dict{Int, Dict{String, Any}}} = Dict{Int, Dict{Int, Dict{String, Any}}}(),
    windowWidth::Int=1200,
    windowHeight::Int=Int(round(windowWidth * fractionOfMainIm)),
    textTexturewidthh::Int32=Int32(2000),
    textTextureheightt::Int32= fractionOfMainIm >= 1.0 ? Int32(1) : Int32(round((windowHeight / (windowWidth * (1 - fractionOfMainIm)))) * textTexturewidthh),
    windowControlStruct::WindowControlStruct=WindowControlStruct();
    quadView::Bool=false
)


    displayMode = getDisplayMode(listOfTextSpecsPrim, quadView)
    #setting number to texture that will be needed in shader configuration
    #enumerate function returns index,value pair of each item in an array, here for the TextureSpecStruct, setting the whichCreated field to the current index
    listOfTextSpecs::Union{Vector{TextureSpec{Float32}},Vector{Vector{TextureSpec{Float32}}}} = begin
        if typeof(listOfTextSpecsPrim) == Vector{TextureSpec}
            # For single image mode with potential overlaid images
            counter = 1
            map(textSpec -> begin
                result = setproperties(textSpec, (whichCreated = counter))
                counter += 1
                result
            end, listOfTextSpecsPrim)
        else
            # For multi-image mode
            map(innerVector -> begin
                counter = 1
                map(textSpec -> begin
                    result = setproperties(textSpec, (whichCreated = counter))
                    counter += 1
                    result
                end, innerVector)
            end, listOfTextSpecsPrim)
        end
    end

    #calculations of necessary constants needed to calculate window size , mouse position ...

    #we need multiple calcDims if the current display mode is multi Image display, evident from the inner vectors
    calcDimStructs::Vector{CalcDimsStruct} = Vector{CalcDimsStruct}()


    if typeof(dataToScrollDims) == DataToScrollDims
        push!(calcDimStructs, CalcDimsStruct(
            windowWidth=windowWidth,
            windowHeight=windowHeight,
            fractionOfMainIm=fractionOfMainIm,
            wordsImageQuadVert=Float32[0,0,0,0,0,0,0,0],
            wordsQuadVertSize=0,
            textTexturewidthh=textTexturewidthh,
            textTextureheightt=textTextureheightt) |>
                              (calcDim) -> getHeightToWidthRatio(calcDim, dataToScrollDims) |>
                                           (calcDim) -> getMainVerticies(calcDim, displayMode, 1)) #passing Image index as 1 for Single Image display mode
    elseif typeof(dataToScrollDims) == Vector{DataToScrollDims}
        foreach(enumerate(dataToScrollDims)) do (imageIndex, scrollDims)
            push!(calcDimStructs, CalcDimsStruct(
                windowWidth=windowWidth,
                windowHeight=windowHeight,
                fractionOfMainIm=fractionOfMainIm,
                wordsImageQuadVert=Float32[0,0,0,0,0,0,0,0],
                wordsQuadVertSize=0,
                textTexturewidthh=textTexturewidthh,
                textTextureheightt=textTextureheightt,
                imagePos=imageIndex) |>
                                  (calcDim) -> getHeightToWidthRatio(calcDim, scrollDims) |>
                                               (calcDim) -> getMainVerticies(calcDim, displayMode, imageIndex))
        end
    end

    #    put!(mainMedEye3dInstance.channel, calcDimStruct)


    #creating window and event listening loop

    #we can pass the first calcDimStruct here, since we need to get the window height and width which is same across all the calcDims
    window, vertex_shader, vao, ebo, fragment_shader_words, vbo_words, shader_program_words, gslsStr, stopChannel = PrepareWindow.displayAll(calcDimStructs[1])

    # ═══ Vulkan Initialization ═══
    # Initialize Vulkan context (instance, device, swapchain, render pass, etc.)
    vk_ctx = VulkanContext.init_vulkan_context(window, calcDimStructs[1].windowWidth, calcDimStructs[1].windowHeight)

    # Determine texture list per panel
    listOfTextSpecsMapped::Vector{Vector{TextureSpec}} = if typeof(listOfTextSpecs) == Vector{TextureSpec{Float32}}
        [listOfTextSpecs for _ in 1:length(calcDimStructs)]
    else
        listOfTextSpecs
    end

    # Build Vulkan pipeline and textures per panel
    initializedTextures::Vector{Vector{TextureSpec}} = []
    forDispObjs::Vector{forDisplayObjects} = Vector{forDisplayObjects}()

    foreach(enumerate(calcDimStructs)) do (index, calcDimStruct)
        textSpecVec = listOfTextSpecsMapped[index]
        w = calcDimStruct.imageTextureWidth
        h = calcDimStruct.imageTextureHeight

        # Create Vulkan textures for this panel
        vk_textures = Any[]
        panel_text_specs = TextureSpec[]

        for (tex_idx, spec) in enumerate(textSpecVec)
            T = parameter_type(spec)
            initial_data = zeros(Float32, Int(w), Int(h))

            filter_mode = (spec.isMultiDiscreteMask || spec.isEditable) ? :nearest : :linear

            vk_tex = VulkanTextures.create_vulkan_texture(
                vk_ctx, Int(w), Int(h),
                Vulkan.FORMAT_R32_SFLOAT,
                initial_data;
                filter_mode=filter_mode,
                name=spec.name
            )
            push!(vk_textures, vk_tex)

            updated_spec = setproperties(spec, (
                associatedActiveNumer = tex_idx - 1,
                actTextrureNumb = UInt32(tex_idx - 1),
                ID = Ref(UInt32(tex_idx)),
                colorMask = RGBA(spec.color.r, spec.color.g, spec.color.b, 1.0)
            ))
            push!(panel_text_specs, updated_spec)
        end
        push!(initializedTextures, panel_text_specs)

        # Generate Vulkan shaders
        color = index == 1 ? "green" : "red"
        frag_src = VulkanShaders.generate_vulkan_fragment_shader(textSpecVec, color)
        vert_src = VulkanShaders.generate_vulkan_zerovbo_vertex_shader()

        # Create pipeline state
        vk_pipeline = VulkanPipeline.create_pipeline_state(
            vk_ctx, vert_src, frag_src, length(textSpecVec)
        )

        # Bind textures to descriptor set
        VulkanPipeline.update_descriptor_textures!(vk_ctx, vk_pipeline, VulkanTextures.VkTexture[vk_textures...])

        # Initial UBO update
        VulkanPipeline.update_ubo!(vk_ctx, vk_pipeline, panel_text_specs)

        # Build numbDicts
        filteredTextures = filter(x -> x.numb >= 0, panel_text_specs)
        usedNumbers = Set{Int32}()
        fixedTextures = map(tex -> begin
            if tex.numb in usedNumbers
                newNumb = tex.numb
                while newNumb in usedNumbers
                    newNumb += 1
                end
                push!(usedNumbers, newNumb)
                setproperties(tex, (numb = newNumb))
            else
                push!(usedNumbers, tex.numb)
                tex
            end
        end, filteredTextures)
        numbDict = Dictionary(map(it -> it.numb, fixedTextures), collect(eachindex(fixedTextures)))

        push!(forDispObjs, forDisplayObjects(
            listOfTextSpecifications=panel_text_specs,
            window=window,
            TextureIndexes=Dictionary(map(it -> it.name, panel_text_specs), collect(eachindex(panel_text_specs))),
            numIndexes=numbDict,
            windowControlStruct=windowControlStruct,
            imagePos=index,
            renderBackend=VulkanBackend,
            vulkanCtx=vk_ctx,
            vulkanPipelineState=vk_pipeline,
            vulkanTextures=vk_textures
        ))
    end
    #finding some texture that can be modifid and set as one active for modifications
    # put!(mainMedEye3dInstance.channel, forDispObj)
    #in order to clean up all resources while closing



    # Create minimal text display structs (text rendering removed for Vulkan)
    forTextDispStructs::Vector{ForWordsDispStruct} = [ForWordsDispStruct() for _ in 1:length(initializedTextures)]

    states = map(x -> StateDataFields(
            textDispObj=forTextDispStructs[x],
            mainForDisplayObjects=forDispObjs[typeof(dataToScrollDims) == Vector{DataToScrollDims} ? x : 1],
            calcDimsStruct=calcDimStructs[x],
            displayMode=displayMode,
            imagePosition=x,
            switchIndex=x,
            mainRectFields=GlShaderAndBufferFields(),
            crosshairFields=GlShaderAndBufferFields(),
            textFields=GlShaderAndBufferFields(),
            spacingsValue=spacing[x],
            originValue=origin[x],
            supervoxelFields=GlShaderAndBufferFields(),
            allSupervoxels=allSupervoxels
        ), 1:length(forDispObjs))

    stateInstances::Vector{StateDataFields} = states

    if (length(stateInstances) > 1)
        #we need to mark the crosshair for the first state as invisible
        # stateInstances[1].mainRectFields.isVisible = false
        for i in 2:length(stateInstances)
            stateInstances[i].switchIndex = 0
        end
    end

    #Setting second state information to be 0, because we need to access information from the first state only
    if length(stateInstances) > 1 && displayMode == MultiImage
        stateInstances[2].switchIndex = 0
    end
    if length(stateInstances) > 1 && displayMode == QuadImage
        for i in 2:length(stateInstances)
            stateInstances[i].switchIndex = 0
        end
    end

    foreach(enumerate(stateInstances)) do (index, stateInstance)
        stateInstance.textureToModifyVec = filter(it -> it.isEditable, initializedTextures[index])
    end

    # Start the single persistent inference worker thread
    MakieEventHandlers.start_inference_worker()

    # Preload initial CT into Docker nnInteractive GPU (fire-and-forget)
    try
        if !isempty(stateInstances) && !isempty(stateInstances[1].onScrollData.dataToScroll)
            for dat in stateInstances[1].onScrollData.dataToScroll
                if dat.name == "CT"
                    InferenceClient.preload_ct_for_nninteractive(Array{Float32,3}(dat.dat))
                    println("[Display] Initial CT preload into nnInteractive GPU initiated"); flush(stdout)
                    break
                end
            end
        end
    catch e
        println("[Display] Initial CT preload skipped: $e"); flush(stdout)
    end

    shouldStop = [false]
    
    #    in case we are recreating all we need to destroy old textures ... generally simplest is destroy window


    function cleanUp()
        # Post CloseWindowEvent to serialize all OpenGL teardown and window destruction on the consumer task
        try
            shouldStop[1] = true
            put!(stopChannel, true)
        catch
        end
    end

    # Set callback to trigger channel shutdown
    GLFW.SetWindowCloseCallback(window, (_) -> begin
        try
            put!(mainMedEye3dInstance.channel, CloseWindowEvent())
        catch
            cleanUp()
        end
    end)

    function consumer(mainChannel::Base.Channel{Any})

        while !shouldStop[1]
            try
                channelData = take!(mainChannel)
                
                # Coalesce SyncLesionEvent: skip intermediate events, keep only the latest
                if channelData isa SyncLesionEvent
                    while isready(mainChannel)
                        peeked = fetch(mainChannel)
                        if peeked isa SyncLesionEvent
                            channelData = take!(mainChannel)  # skip intermediate, keep latest
                        else
                            break
                        end
                    end
                end
                # Coalesce scroll events: sum all pending scroll deltas into one
                if channelData isa Int64
                    while isready(mainChannel)
                        peeked = fetch(mainChannel)
                        if peeked isa Int64
                            channelData += take!(mainChannel)
                        else
                            break
                        end
                    end
                end
                if channelData isa CloseWindowEvent
                    println("CloseWindowEvent received: shutting down window")
                    flush(stdout)
                    shouldStop[1] = true
                    
                    # Destroy Vulkan resources
                    try
                        for state in stateInstances
                            obj = state.mainForDisplayObjects
                            if obj.vulkanCtx !== nothing
                                VulkanPipeline.destroy_pipeline_state!(obj.vulkanCtx, obj.vulkanPipelineState)
                                VulkanContext.destroy_vulkan_context!(obj.vulkanCtx)
                            end
                        end
                    catch e
                        @warn "Error cleaning Vulkan resources: $e"
                    end
                    
                    try put!(stopChannel, true) catch end
                    
                    try
                        if window.handle != C_NULL
                            GLFW.SetWindowShouldClose(window, true)
                            GLFW.DestroyWindow(window)
                            window.handle = C_NULL
                        end
                    catch e
                        @warn "Error destroying GLFW window: $e"
                    end
                    break
                end

                # get the aggregation here, only when the type is mouseStruct.
                if typeof(channelData) == MouseStruct
                    # Left-button drag aggregation for mask painting ONLY when painting is active.
                    if channelData.isLeftButtonDown && stateInstances[1].valueForMasToSet.is_painting_active
                        mouseStructAggregationArray::Vector{MouseStruct} = [channelData]
                        while !isempty(mainChannel) && typeof(fetch(mainChannel)) == MouseStruct
                            peeked = fetch(mainChannel)
                            if !peeked.isLeftButtonDown
                                break
                            end
                            push!(mouseStructAggregationArray, take!(mainChannel))
                        end
                        channelData = mouseStructAggregationArray
                    else
                        # Coalesce rapid non-painting mouse movements: drain stale intermediate moves,
                        # but preserve button state transitions (press / release).
                        while !isempty(mainChannel) && typeof(fetch(mainChannel)) == MouseStruct
                            peeked = fetch(mainChannel)
                            if peeked.isRightButtonDown != channelData.isRightButtonDown ||
                               peeked.isLeftButtonDown != channelData.isLeftButtonDown
                                break
                            end
                            channelData = take!(mainChannel)  # discard stale move, keep latest
                        end
                    end

                elseif typeof(channelData) == CalcDimsStruct || typeof(channelData) == forDisplayObjects || typeof(channelData) == FullScrollableDat
                    stateInstances[1].switchIndex = channelData.imagePos
                end

                if !(channelData isa MouseStruct) && !(channelData isa Vector{MouseStruct}) && !(channelData isa Int64)
                    println("CONSUMER: processing $(typeof(channelData))")
                    flush(stdout)
                end
                on_next!(stateInstances, channelData)
                
                if !shouldStop[1]
                    # Build panel render data for Vulkan
                    vk_panels = VulkanRender.PanelRenderData[]
                    for state in stateInstances
                        if state.calcDimsStruct.mainQuadVertSize <= 0 || all(state.calcDimsStruct.mainImageQuadVert .== 0.0f0)
                            continue
                        end
                        if state.currentlyDispDat.sliceNumber == 0
                            continue
                        end
                        
                        obj = state.mainForDisplayObjects
                        if obj.vulkanPipelineState === nothing
                            continue
                        end
                        
                        # Upload texture data from currentlyDispDat to Vulkan textures
                        if state.currentlyDispDat !== nothing && !isempty(state.currentlyDispDat.listOfDataAndImageNames)
                            for updateDat in state.currentlyDispDat.listOfDataAndImageNames
                                for (i, vk_tex) in enumerate(obj.vulkanTextures)
                                    if hasproperty(vk_tex, :name) && vk_tex.name == updateDat.name
                                        try
                                            upload_data = Float32.(updateDat.dat)
                                            VulkanTextures.update_vulkan_texture!(obj.vulkanCtx, vk_tex, upload_data)
                                        catch e
                                            @warn "Vulkan texture upload failed for $(updateDat.name)" exception=e
                                        end
                                        break
                                    end
                                end
                            end
                        end
                        
                        # Update UBO with current texture specs
                        VulkanPipeline.update_ubo!(obj.vulkanCtx, obj.vulkanPipelineState, obj.listOfTextSpecifications)
                        # Build push constants for zoom/pan + NDC quad bounds
                        zoom = max(0.1f0, state.calcDimsStruct.zoom)
                        scale = 1.0f0 / zoom
                        offsetX = state.calcDimsStruct.panY
                        offsetY = state.calcDimsStruct.panX
                        
                        # Extract NDC quad bounds from mainImageQuadVert
                        # Each vertex has 8 floats: X, Y, Z, R, G, B, U, V
                        verts = state.calcDimsStruct.mainImageQuadVert
                        if length(verts) >= 32
                            xs = Float32[verts[1], verts[9], verts[17], verts[25]]
                            ys = Float32[verts[2], verts[10], verts[18], verts[26]]
                            ndc_left   = minimum(xs)
                            ndc_right  = maximum(xs)
                            ndc_bottom = minimum(ys)
                            ndc_top    = maximum(ys)
                        else
                            ndc_left   = -1.0f0
                            ndc_bottom = -1.0f0
                            ndc_right  =  1.0f0
                            ndc_top    =  1.0f0
                        end
                        
                        # Push constants: uvScale(2) + uvOffset(2) + ndcMin(2) + ndcMax(2) = 8 floats
                        push_consts = Float32[scale, scale, offsetX, offsetY, 
                                              ndc_left, ndc_bottom, ndc_right, ndc_top]
                        
                        w = Float32(obj.vulkanCtx.width)
                        h = Float32(obj.vulkanCtx.height)
                        
                        panel = VulkanRender.PanelRenderData(
                            obj.vulkanPipelineState,
                            push_consts,
                            Float32(0), Float32(0), w, h
                        )
                        push!(vk_panels, panel)
                    end
                    
                    if !isempty(vk_panels)
                        VulkanRender.render_frame!(stateInstances[1].mainForDisplayObjects.vulkanCtx, vk_panels)
                    end
                end
            catch e
                if e isa InterruptException || (e isa GLFW.GLFWError && e.code == GLFW.NOT_INITIALIZED)
                    println("CONSUMER FATAL ERROR: $e")
                    println(sprint(showerror, e, catch_backtrace()))
                    flush(stdout)
                    shouldStop[1] = true
                else
                    println("CONSUMER ERROR (continuing): $e")
                    println(sprint(showerror, e, catch_backtrace()))
                    flush(stdout)
                    # Update AI status label so user sees the error
                    try
                        MakieEventHandlers.ai_status_text[] = MakieEventHandlers.safe_status_text("[Error] $(sprint(showerror, e))")
                    catch; end
                    # Log to file for post-mortem analysis
                    try
                        open("/tmp/medeye3d_errors.log", "a") do f
                            println(f, "$(Dates.now()) CONSUMER ERROR: $(sprint(showerror, e))")
                            println(f, sprint(showerror, e, catch_backtrace()))
                            println(f, "---")
                        end
                    catch; end
                end
            end
        end
    end #end of consumer

    # Release context from the main thread so the background consumer task can claim it (or just release it)
    # Run consumer task (Vulkan context is thread-safe, no context switching needed)
    mainMedEye3dInstance = MainMedEye3d(channel=Base.Channel{Any}(consumer, 1000; spawn=false), textDispObj=forTextDispStructs[1], displayMode=displayMode, states=stateInstances)
    
    # Register main channel for background tasks to dispatch events back
    MakieEventHandlers.register_main_channel!(mainMedEye3dInstance.channel)

    foreach(calcDimStructs) do currentCalcDim
        put!(mainMedEye3dInstance.channel, currentCalcDim)
    end

    foreach(forDispObjs) do currentDispObj
        put!(mainMedEye3dInstance.channel, currentDispObj)
    end




    put!(mainMedEye3dInstance.channel, forTextDispStructs[1])



    registerInteractions(window, mainMedEye3dInstance, calcDimStructs)#passing needed subscriptions from GLFW
    
    # Set initial window title
    idx = MakieEventHandlers.current_tp_index[]
    label = get(MakieEventHandlers.tp_labels, idx, "TP $idx")
    GLFW.SetWindowTitle(window, "MedEye3d - Viewing: $label")

    return mainMedEye3dInstance
end #coordinateDisplay

"""
Close the MedEye3d viewer window gracefully through the channel.
"""
close_window(main::MainMedEye3d) = put!(main.channel, CloseWindowEvent())

"""
Update the window title through the channel.
"""
set_window_title(main::MainMedEye3d, title::String) = put!(main.channel, SetWindowTitleEvent(title))

"""
Resize the window viewport and framebuffer dimensions through the channel.
"""
resize_window(main::MainMedEye3d, width::Int, height::Int) = put!(main.channel, ResizeWindowEvent(width, height))


"""
Defining some default textures for PET, CT and ManualModif, subject to change
"""
function getDefaultTexture(
    studyType::Union{MedImages.MedImage_data_struct.Image_type,String},
    numbIndex::Int32
)
    if studyType == MedImages.MedImage_data_struct.PET_type
        return TextureSpec{Float32}(
            name="PET",
            studyType="PET",
            isContinuusMask=true,
            numb=numbIndex,
            colorSet=[RGB(0.0, 0.0, 0.0), RGB(1.0, 1.0, 0.0), RGB(1.0, 0.5, 0.0), RGB(1.0, 0.0, 0.0), RGB(1.0, 0.0, 0.0)],
            minAndMaxValue=Float32.([200, 8000])
        )
    elseif studyType == MedImages.MedImage_data_struct.CT_type
        return TextureSpec{Float32}(
            name="CTIm",
            studyType="CT",
            numb=numbIndex,
            color=RGB(1.0, 1.0, 1.0),
            minAndMaxValue=Float32.([0, 100])
        )
    elseif studyType == "ManualModif"
        colors_mapped = map(c -> RGB(c[1]/255, c[2]/255, c[3]/255), distinctColorsSaved.listOfColors)
        return TextureSpec{Float32}(
            name="manualModif",
            numb=Int32(2),
            isMultiDiscreteMask=true,
            colorSet=colors_mapped,
            minAndMaxValue=Float32.([0, length(colors_mapped)]),
            isEditable=true
        )

    else
        return TextureSpec{Float32}(
            name="default",
            studyType="CT",
            numb=Int32(4),
            colorSet=[RGB(0.0, 0.0, 0.0), RGB(1.0, 1.0, 1.0)]
        )
    end
end

"""
Loading Nifti volumes or Dicom Series with MedImages.jl package.
Single Image or Multi-Image display supported.
"""
function loadRegisteredImages(
    studySrc::Union{Vector{Tuple{String,String}},Tuple{String,String},Vector{Vector{Tuple{String,String}}}}
)

    medImageDataInstances::Union{Vector{MedImages.MedImage},Vector{Vector{MedImages.MedImage}}} = typeof(studySrc) == Vector{Vector{Tuple{String,String}}} ? Vector{Vector{MedImages.MedImage}}() : Vector{MedImages.MedImage}()

    # FIRST: Load all the images
    if typeof(studySrc) == Tuple{String,String}
        push!(medImageDataInstances, MedImages.Load_and_save.load_image(studySrc[1], studySrc[2]))
    elseif typeof(studySrc) == Vector{Tuple{String,String}}
        for studySrcPath in studySrc
            push!(medImageDataInstances, MedImages.Load_and_save.load_image(studySrcPath[1], studySrcPath[2]))
        end
    elseif typeof(studySrc) == Vector{Vector{Tuple{String,String}}}
        for studySrcVector in studySrc
            medImageInnerVector::Vector{MedImages.MedImage} = Vector{MedImages.MedImage}()
            for studySrcPath in studySrcVector
                push!(medImageInnerVector, MedImages.Load_and_save.load_image(studySrcPath[1], studySrcPath[2]))
            end
            push!(medImageDataInstances, medImageInnerVector)
        end
    end

    # SECOND: Now handle resampling for overlaid images (AFTER loading)
    if typeof(studySrc) == Vector{Tuple{String,String}} && length(studySrc) > 1
        @info "Detected overlaid images with $(length(studySrc)) images"
        
        # Use the first image (typically CT) as reference
        reference_image = medImageDataInstances[1] 
        reference_size = size(reference_image.voxel_data)
        
        for i in 2:length(medImageDataInstances)
            current_size = size(medImageDataInstances[i].voxel_data)
            if current_size != reference_size
                @info "Image $(i) size $(current_size) differs from reference $(reference_size)"
                @info "Resampling using SimpleITK..."
                medImageDataInstances[i] = resample_to_reference_sitk(medImageDataInstances[i], reference_image)
                @info "SimpleITK resampling completed for image $(i)"
            else
                @info "Image $(i) already matches reference dimensions"
            end
        end
        
        @info "All images now have matching dimensions: $(size(medImageDataInstances[1].voxel_data))"
    end

    # INTERMEDIATE: Increase resolution by 2 times by default for all images loaded
    if typeof(medImageDataInstances) == Vector{MedImages.MedImage}
        for (i, medImage) in enumerate(medImageDataInstances)
            @info "Applying default 2x resolution increase to image $(i) by resampling..."
            new_sp = (medImage.spacing[1]/2.0, medImage.spacing[2]/2.0, medImage.spacing[3]/2.0)
            medImageDataInstances[i] = MedImages.Resample_to_target.resample_to_spacing(medImage, new_sp, MedImages.MedImage_data_struct.Linear_en, 0.0)
        end
    elseif typeof(medImageDataInstances) == Vector{Vector{MedImages.MedImage}}
        for i in 1:length(medImageDataInstances)
            for j in 1:length(medImageDataInstances[i])
                @info "Applying default 2x resolution increase to image $(i),$(j) by resampling..."
                medImage = medImageDataInstances[i][j]
                new_sp = (medImage.spacing[1]/2.0, medImage.spacing[2]/2.0, medImage.spacing[3]/2.0)
                medImageDataInstances[i][j] = MedImages.Resample_to_target.resample_to_spacing(medImage, new_sp, MedImages.MedImage_data_struct.Linear_en, 0.0)
            end
        end
    end

    # THIRD: Apply orientation corrections and type conversions
    if typeof(medImageDataInstances) == Vector{MedImages.MedImage}
        for medImageDataInstance in medImageDataInstances
            #permuting the voxelData to some default orientation, such that the image is not inverted or sideways
            sizeInfo = size(medImageDataInstance.voxel_data)
            for outerNum in 1:sizeInfo[1]
                for innerNum in 1:sizeInfo[3]
                    medImageDataInstance.voxel_data[outerNum, :, innerNum] = reverse(medImageDataInstance.voxel_data[outerNum, :, innerNum])
                end
            end
            #Float conversion happens here for voxelData, currently only Floats are supported to keep it simple
            medImageDataInstance.voxel_data = Float32.(medImageDataInstance.voxel_data)
        end

    elseif typeof(medImageDataInstances) == Vector{Vector{MedImages.MedImage}}
        for medImageInnerVector in medImageDataInstances
            for medImageDataInstance in medImageInnerVector
                #medImageDataInstance.voxel_data = permutedims(medImageDataInstance.voxel_data, (3, 2, 1)) #previously in the test script the default was (3, 2, 1)
                sizeInfo = size(medImageDataInstance.voxel_data)
                for outerNum in 1:sizeInfo[1]
                    for innerNum in 1:sizeInfo[3]
                        medImageDataInstance.voxel_data[outerNum, :, innerNum] = reverse(medImageDataInstance.voxel_data[outerNum, :, innerNum])
                    end
                end
                #Float conversion happens here for voxelData, currently only Floats are supported to keep it simple
                medImageDataInstance.voxel_data = Float32.(medImageDataInstance.voxel_data)
            end
        end
    end

    return medImageDataInstances #returns the vector of MedImages or a Vector of Vector of MedImages
end



"""
    displayImage(studySrc; kwargs...)

High-Level entry point for the MedEye3d Visualizer.

# Arguments
- `studySrc`: Path(s) and metadata for the volume data (e.g. `Vector{Tuple{String,String}}` where tuples represent path and image type).

# Keyword Arguments
- `textureSpecArray`: Definitions of masks, colors, and visibility defaults.
- `voxelDataTupleVector`: Explicitly passed dense voxel data (used by `run_interactive_mrb.jl`).
- `spacings` & `origins`: Patient anatomical spatial scaling vectors.
- `fractionOfMainImage`: Width ratio occupied by the OpenGL panel (0.0 - 1.0).
- `windowWidth`: Starting horizontal resolution of the GLFW window.
- `quadView`: Pass `true` to enable the 4-panel multi-planar layout.
- `dimensionsToScroll`: The primary slicing dimension.

# Returns
`MainMedEye3d`: A handle representing the running viewer, containing the `channel` for posting GUI events.
"""
function displayImage(
    studySrc::Union{Vector{Tuple{String,String}},Tuple{String,String},Vector{Vector{Tuple{String,String}}}}
    ; textureSpecArray::Union{Vector{TextureSpec},Vector{Vector{TextureSpec}}}=Vector{TextureSpec}(),
    voxelDataTupleVector::Union{Vector{Any},Vector{Vector{Any}}}=[],
    spacings::Union{Vector{Tuple{Float64,Float64,Float64}},Vector{Vector{Tuple{Float64,Float64,Float64}}}}=Vector{Tuple{Float64,Float64,Float64}}(),
    origins::Union{Vector{Tuple{Float64,Float64,Float64}},Vector{Vector{Tuple{Float64,Float64,Float64}}}}=Vector{Tuple{Float64,Float64,Float64}}(),
    fractionOfMainImage::Float32=Float32(0.8),
    windowWidth::Int=1000,
    svVertAndInd::Dict{String,Vector}=Dict{String,Vector}("supervoxel_vertices" => [], "supervoxel_indices" => []),
    all_supervoxels::Dict{Int,Dict{Int, Dict{String, Any}}} = Dict{Int, Dict{Int, Dict{String, Any}}}(),
    quadView::Bool=false,
    dimensionsToScroll::Union{Int, Vector{Int}}=3
)


    #asserting that the length of the studySrc is 2, if it is a multi-dimensions vector
    if typeof(studySrc) == Vector{Vector{Tuple{String,String}}} && !quadView
        try
            @assert isempty(studySrc) || length(studySrc) == 2 || length(studySrc) == 4
        catch assertionError
            @error "MedEye3d.jl currently do not support more than 2 images for comparison, unless quadView is true." assertionError
        end
    end

    medImageData::Union{Vector{MedImages.MedImage},Vector{Vector{MedImages.MedImage}}} = loadRegisteredImages(studySrc)
    #NOTE : for overlaid images, they need to be resampled first

    if isempty(textureSpecArray) && isempty(voxelDataTupleVector) && isempty(spacings) && isempty(origins)
        #Reassigning textureSpecArray, voxelDataTupleVector, spacings  depending upong the typeof studySrc
        textureSpecArray = typeof(studySrc) == Vector{Vector{Tuple{String,String}}} ? Vector{Vector{TextureSpec}}() : Vector{TextureSpec}()
        voxelDataTupleVector = typeof(studySrc) == Vector{Vector{Tuple{String,String}}} ? Vector{Vector{Any}}() : Vector{Any}()
        spacings = typeof(studySrc) == Vector{Vector{Tuple{String,String}}} ? Vector{Vector{Tuple{Float64,Float64,Float64}}}() : Vector{Tuple{Float64,Float64,Float64}}()
        origins = typeof(studySrc) == Vector{Vector{Tuple{String,String}}} ? Vector{Vector{Tuple{Float64,Float64,Float64}}}() : Vector{Tuple{Float64,Float64,Float64}}()

        if typeof(medImageData) == Vector{MedImages.MedImage}
            for (index, medImage) in enumerate(medImageData)
                if medImage.image_type == MedImages.MedImage_data_struct.PET_type
                    push!(textureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.PET_type, Int32(index)))
                    push!(voxelDataTupleVector, ("PET", medImage.voxel_data))
                    push!(spacings, medImage.spacing)
                    push!(origins, medImage.origin)
                elseif medImage.image_type == MedImages.MedImage_data_struct.CT_type
                    push!(textureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.CT_type, Int32(index)))
                    push!(voxelDataTupleVector, ("CTIm", medImage.voxel_data))
                    push!(spacings, medImage.spacing)
                    push!(origins, medImage.origin)
                end

            end

        elseif typeof(medImageData) == Vector{Vector{MedImages.MedImage}}
            for medImageInnerVector in medImageData
                for (innerIndex, medImage) in enumerate(medImageInnerVector)


                    #==
                     check to ensure No texture has number 2, since it is reserved for ManualModif
                    ==#
                    innerIndex = innerIndex == 2 ? innerIndex + 1 : innerIndex

                    innerTextureSpecArray::Vector{TextureSpec} = Vector{TextureSpec}()
                    innerVoxelDataTupleVector::Vector{Any} = Vector{Any}()
                    innerSpacings::Vector{Tuple{Float64,Float64,Float64}} = Vector{Tuple{Float64,Float64,Float64}}()
                    innerOrigins::Vector{Tuple{Float64,Float64,Float64}} = Vector{Tuple{Float64,Float64,Float64}}()

                    if medImage.image_type == MedImages.MedImage_data_struct.PET_type
                        push!(innerTextureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.PET_type, Int32(innerIndex)))
                        push!(innerVoxelDataTupleVector, ("PET", medImage.voxel_data))
                        push!(innerSpacings, medImage.spacing)
                        push!(innerOrigins, medImage.origin)
                    elseif medImage.image_type == MedImages.MedImage_data_struct.CT_type
                        push!(innerTextureSpecArray, getDefaultTexture(MedImages.MedImage_data_struct.CT_type, Int32(innerIndex)))
                        push!(innerVoxelDataTupleVector, ("CTIm", medImage.voxel_data))
                        push!(innerSpacings, medImage.spacing)
                        push!(innerOrigins, medImage.origin)
                    end

                    push!(textureSpecArray, innerTextureSpecArray)
                    push!(voxelDataTupleVector, innerVoxelDataTupleVector)
                    push!(spacings, innerSpacings)
                    push!(origins, innerOrigins)
                end
            end

        end
    end

    #for correct display and windowing for PET we do (median -std /2) for min and (median + std * 2) for max

    if typeof(textureSpecArray) == Vector{TextureSpec}
        for textur in textureSpecArray
            if textur.studyType == "PET"
                textur.minAndMaxValue = Float32.([median(voxelDataTupleVector[1][2]) - std(voxelDataTupleVector[1][2]) / 2, median(voxelDataTupleVector[1][2]) + std(voxelDataTupleVector[1][2]) * 2])
            end
        end

    elseif typeof(textureSpecArray) == Vector{Vector{TextureSpec}}

        for (index, texturVector) in enumerate(textureSpecArray)
            for textur in texturVector
                if textur.studyType == "PET"
                    textur.minAndMaxValue = Float32.([median(voxelDataTupleVector[index][1][2]) - std(voxelDataTupleVector[index][1][2]) / 2, median(voxelDataTupleVector[index][1][2]) + std(voxelDataTupleVector[index][1][2]) * 2])
                end
            end
        end

    end


    #Texture specification for manual modification Mask
    if typeof(textureSpecArray) == Vector{TextureSpec}
        insert!(textureSpecArray, 2, getDefaultTexture("ManualModif", Int32(2)))
    elseif typeof(textureSpecArray) == Vector{Vector{TextureSpec}}
        for texturVector in textureSpecArray
            insert!(texturVector, 2, getDefaultTexture("ManualModif", Int32(2)))
        end
    end


        # // In displayImage function, after manual modification insertion:
    if typeof(textureSpecArray) == Vector{TextureSpec}
            if typeof(studySrc) == Vector{Tuple{String, String}} && length(studySrc) > 1
            # Fix the PET texture to have numb=3 instead of 2
            for (index, texture) in enumerate(textureSpecArray)
                if texture.studyType == "PET"
                    textureSpecArray[index] = setproperties(texture, (numb = Int32(3),))  # Give PET unique number
                    @info "Fixed PET texture numb to 3"
                end
            end
            
            # Fix PET intensity range
            for (index, textur) in enumerate(textureSpecArray)
                if textur.studyType == "PET"
                    pet_data = voxelDataTupleVector[2][2]  # PET is second image
                    positive_pet = pet_data[pet_data .> 0]
                    if !isempty(positive_pet)
                        # Use a higher threshold to exclude very low values
                        pet_threshold = quantile(positive_pet, 0.05)  # Bottom 5% threshold
                        pet_max = quantile(positive_pet, 0.95)        # Top 5% threshold
                        textureSpecArray[index] = setproperties(textur, (
                minAndMaxValue = Float32.([pet_threshold, pet_max]),
                maskContribution = Float32(1.0),  # Full contribution
                isVisible = true,
                colorSet = [
                    RGB(0.0, 0.0, 0.0),      # Transparent for very low values
                    RGB(0.0, 0.0, 0.8),      # Dark blue
                    RGB(0.0, 0.8, 0.8),      # Cyan
                    RGB(0.0, 1.0, 0.0),      # Green
                    RGB(1.0, 1.0, 0.0),      # Yellow
                    RGB(1.0, 0.5, 0.0),      # Orange
                    RGB(1.0, 0.0, 0.0)       # Bright red
                ]
            ))
        
                        @info "Fixed PET range to: $(textureSpecArray[index].minAndMaxValue)"
                    end
                end
            end
        end
    end

    # @info "look here" typeof(voxelDataTupleVector) typeof(voxelDataTupleVector[1]) voxelDataTupleVector[1]
    # voxelDataForUniforms::Union{Vector{Array{Float32,3}},Vector{Vector{Array{Float32,3}}}} = map(x -> map(tup -> tup[2], x), voxelDataTupleVector)
    voxelDataForUniforms::Union{Vector{Array{Float32,3}},Vector{Vector{Array{Float32,3}}}} = typeof(voxelDataTupleVector) == Vector{Any} ? map(tuple -> tuple[2], voxelDataTupleVector) : map(innerVector -> map(tuple -> tuple[2], innerVector), voxelDataTupleVector)

    if typeof(voxelDataTupleVector) == Vector{Any}
        voxelDataForUniforms = map(tuple -> tuple[2], voxelDataTupleVector)
    elseif typeof(voxelDataTupleVector) == Vector{Vector{Any}}
        voxelDataForUniforms = map(innerVector -> map(tuple -> tuple[2], innerVector), voxelDataTupleVector)
    end

    #Since there are repeating Tuples for Manual Modif, we need to ensure only a unique ones exist based on the first loaded image

    if typeof(voxelDataTupleVector) == Vector{Any}
        if !isempty(voxelDataTupleVector) && voxelDataTupleVector[1][1] != "manualModif" && (length(voxelDataTupleVector) < 2 || voxelDataTupleVector[2][1] != "manualModif")
            insert!(voxelDataTupleVector, 2, ("manualModif", zeros(Float32, size(voxelDataForUniforms[1]))))
        end
    elseif typeof(voxelDataTupleVector) == Vector{Vector{Any}}
        for (vectorIndex, innerVector) in enumerate(voxelDataTupleVector)
            if !isempty(innerVector) && innerVector[1][1] != "manualModif" && (length(innerVector) < 2 || innerVector[2][1] != "manualModif")
                insert!(innerVector, 2, ("manualModif", zeros(Float32, size(voxelDataForUniforms[vectorIndex][1]))))
            end
        end
    end


    datToScrollDimsB::Union{DataToScrollDims,Vector{DataToScrollDims}} = typeof(voxelDataTupleVector) == Vector{Vector{Any}} ? Vector{DataToScrollDims}() : DataToScrollDims()
    mainLines::Union{Vector{Vector{SimpleLineTextStruct}},Vector{SimpleLineTextStruct}} = Vector{SimpleLineTextStruct}() #Subject to change
    supplLines::Union{Vector{Vector{Vector{SimpleLineTextStruct}}},Vector{Vector{SimpleLineTextStruct}}} = Vector{Vector{SimpleLineTextStruct}}() #Subject to change


    if typeof(voxelDataForUniforms) == Vector{Array{Float32,3}} #Our data is in Float32 format in 3 dimensions
        dimToScroll = typeof(dimensionsToScroll) <: Vector ? dimensionsToScroll[1] : dimensionsToScroll
        datToScrollDimsB = DataToScrollDims(imageSize=size(voxelDataForUniforms[1]), voxelSize=spacings[1], dimensionToScroll=dimToScroll)
        mainLines = textLinesFromStrings(["main line 1", "main line 2"])
        supplLines = map(x -> textLinesFromStrings(["sub line 1 in $(x)", "sub line 2 in $(x)"]), 1:size(voxelDataForUniforms[1])[dimToScroll])

    elseif typeof(voxelDataForUniforms) == Vector{Vector{Array{Float32,3}}}
        for (index, innerVector) in enumerate(voxelDataForUniforms)
            dimToScroll = typeof(dimensionsToScroll) <: Vector ? dimensionsToScroll[index] : dimensionsToScroll
            push!(datToScrollDimsB, DataToScrollDims(imageSize=size(innerVector[1]), voxelSize=spacings[index][1], dimensionToScroll=dimToScroll))
        end
        mainLines = textLinesFromStrings(["main line 1", "main line 2"])
        dimToScroll0 = typeof(dimensionsToScroll) <: Vector ? dimensionsToScroll[1] : dimensionsToScroll
        supplLines = map(x -> textLinesFromStrings(["sub line 1 in $(x)", "sub line 2 in $(x)"]), 1:size(voxelDataForUniforms[1][1])[dimToScroll0]) #change this added [1] bcuz to get the first vector

    end



    sliceData::Union{Vector{ThreeDimRawDat{Float32}},Vector{Vector{ThreeDimRawDat{Float32}}}} = typeof(voxelDataTupleVector) == Vector{Vector{Any}} ? Vector{Vector{ThreeDimRawDat{Float32}}}() : Vector{ThreeDimRawDat{Float32}}()
    mainScrollData::Union{FullScrollableDat,Vector{FullScrollableDat}} = typeof(voxelDataTupleVector) == Vector{Vector{Any}} ? Vector{FullScrollableDat}() : FullScrollableDat()
    if typeof(voxelDataTupleVector) == Vector{Any}
        sliceDatad = getThreeDims(voxelDataTupleVector)
        # @info typeof(sliceDatad)
        sliceData = sliceDatad
        dimToScroll = typeof(dimensionsToScroll) <: Vector ? dimensionsToScroll[1] : dimensionsToScroll
        mainScrollData = FullScrollableDat(dataToScrollDims=datToScrollDimsB, dimensionToScroll=dimToScroll, dataToScroll=sliceData, mainTextToDisp=mainLines, sliceTextToDisp=supplLines)
    elseif typeof(voxelDataTupleVector) == Vector{Vector{Any}}
        for (index, innerVector) in enumerate(voxelDataTupleVector)
            push!(sliceData, getThreeDims(innerVector))
        end
    end

    if typeof(voxelDataForUniforms[1]) <: Vector
        # so we have multiple planes per image
        mainScrollData = Vector{FullScrollableDat}()
        for (index, innerVector) in enumerate(voxelDataForUniforms)
            dimToScroll = typeof(dimensionsToScroll) <: Vector ? dimensionsToScroll[index] : dimensionsToScroll
            push!(mainScrollData, FullScrollableDat(dataToScrollDims=datToScrollDimsB[index], dimensionToScroll=dimToScroll, dataToScroll=sliceData[index], mainTextToDisp=mainLines, sliceTextToDisp=supplLines))
        end
    end


    # Few assertions to ensure correct types between the textureSpecification type and the voxel data type

    if typeof(textureSpecArray) == Vector{TextureSpec}

        for (textureSpec, tupleVector) in zip(textureSpecArray, voxelDataTupleVector)
            @assert typeof(textureSpec) == TextureSpec{Float32}
            # @info typeof(voxelData)
            @assert typeof(tupleVector[2]) == Array{Float32,3}

            @assert textureSpec.name == tupleVector[1]
            # @info typeof(voxelData)
        end

    elseif typeof(textureSpecArray) == Vector{Vector{TextureSpec}}
        for (textureSpecVector, tupleVector) in zip(textureSpecArray, voxelDataTupleVector)
            for (textureSpec, tuple) in zip(textureSpecVector, tupleVector)
                @assert typeof(textureSpec) == TextureSpec{Float32}
                @assert typeof(tuple[2]) == Array{Float32,3}
                @assert textureSpec.name == tuple[1]
            end
        end
    end



    medEye3dChannelInstance = coordinateDisplay(textureSpecArray, fractionOfMainImage, datToScrollDimsB, spacings, origins, svVertAndInd, all_supervoxels, windowWidth; quadView=quadView)



    # Populating the fields for mainMedEye3dInstance
    # try
    displayMode = getDisplayMode(textureSpecArray, quadView)

    if displayMode == SingleImage
        medEye3dChannelInstance.voxelArrayShapes = map(x -> size(x[2]), voxelDataTupleVector)
        medEye3dChannelInstance.voxelArrayTypes = map(x -> typeof(x[2][1, 1, 1]), voxelDataTupleVector) #getting the type of the first element

        @info "!! Crosshair rendering is currently only supported in Multi image display mode !!"
    else
        @info "!! On Screen Voxel modification is currently only supported in Single image display mode !!"
    end



    passDataForScrolling(medEye3dChannelInstance, mainScrollData)
    return medEye3dChannelInstance
end




####################################################################################################
end #SegmentationDisplay


"""
DOCS::
Usage of interactive thread
In mutli-image only one image modality at a time can be visualized simultaneously. Either pet or ct.
During the initilization of states in consumer not all the fields of GlShaderAndBufferFields are populated. (for eg mainRectFields.shaderProgram)
Annotations are not saved and are cannot be undone.
Crosshair rendering is only supported in multi-image display mode.
Annotations are only supported in single-image display mode.
Disabling the concept of overlaid images in multi-image display mode. Thought manual-modification masks are working.
Advise Users to restart their Julia REPL session once they are done with the visualization
Advise Users to only change the plane of the left image in multi-image display for crosshair display.
ADvise users when willing to display hdf5 data first convert into nifti with the function and then display normally
Loading of DICOM Series is Supported now

NOTS:
return stuff similar to words_display for each calcDimStruct in the vector of calcDims
make changes to put forTextDispStruct in the mainMedEye3dInstance
allow user access to voxel modification in the case of single Image display  [DONE]
fix text rendering when in multi image
Add support for dynamic crosshair rendering on passive image
add support for supervoxel line rendering and sobel filter
Dynamic allocation of texture number no matter if the images are overlaid or not
Correct windowing for ct images f1, f2,f3
Test overlaid images in single Image and multi-image
With Crosshair rendering added, the keymaps for setting visiblity does not work
With Crosshair rendering added, the keymaps for changing windowing does not work
Allow People to load and visualize custom annotations masks [manual modifications]
In shader and vertices for supervoxels, during the calculation of vertices for supervoxels, make sure to use Float32 for calculation.
Add a sample nifti file in the supervoxel directory, since the function seem to be modifying the original nifti input image
Fix text rendering in single Image display , which is sprouting from reactToScroll.jl file with rendering for supervoxels.
"""
