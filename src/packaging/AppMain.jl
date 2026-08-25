module MedEye3dApp

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using MedImages
using ColorTypes
using Logging
using Statistics
using LinearAlgebra
using Dates
import GLFW
import ModernGL

export julia_main

"""
    get_writable_log_dir() -> String

Returns a writable directory path for application log files on Windows.
Prefers `%APPDATA%\\MedEye3D\\logs` or `%LOCALAPPDATA%\\MedEye3D\\logs`, falling back to `tempdir()`.
"""
function get_writable_log_dir()::String
    appdata = get(ENV, "APPDATA", "")
    base_dir = if !isempty(appdata) && isdir(appdata)
        joinpath(appdata, "MedEye3D", "logs")
    else
        localappdata = get(ENV, "LOCALAPPDATA", "")
        if !isempty(localappdata) && isdir(localappdata)
            joinpath(localappdata, "MedEye3D", "logs")
        else
            joinpath(tempdir(), "MedEye3D_logs")
        end
    end
    try
        mkpath(base_dir)
        return base_dir
    catch
        return tempdir()
    end
end

"""
    launch_demo(; quad::Bool=true)

Launches a standalone interactive 3D medical visualizer with synthetic CT and segmentation mask.
Supports single panel or multi-planar 4-panel (Axial, Coronal, Sagittal) quad view.
"""
function launch_demo(; quad::Bool=true)
    println("Initializing MedEye3D Demo Visualization...")

    # Create synthetic CT volume (ellipsoid structure + simulated tissue)
    dim_x, dim_y, dim_z = 128, 128, 64
    vol_ct = zeros(Float32, dim_x, dim_y, dim_z)
    vol_mask = zeros(Float32, dim_x, dim_y, dim_z)

    cx, cy, cz = dim_x ÷ 2, dim_y ÷ 2, dim_z ÷ 2
    for z in 1:dim_z, y in 1:dim_y, x in 1:dim_x
        # Simulated tissue background
        vol_ct[x, y, z] = -100.0f0 + 40.0f0 * sin(0.08f0 * x) * cos(0.08f0 * y)
        
        # Ellipsoidal organ / phantom
        dx = (x - cx) / (dim_x * 0.35f0)
        dy = (y - cy) / (dim_y * 0.35f0)
        dz = (z - cz) / (dim_z * 0.35f0)
        r2 = dx*dx + dy*dy + dz*dz
        if r2 < 1.0f0
            vol_ct[x, y, z] = 40.0f0 + 120.0f0 * (1.0f0 - Float32(sqrt(r2)))
        end

        # Synthetic focal lesion
        lx = (x - (cx + 16)) / 10.0f0
        ly = (y - (cy + 8)) / 10.0f0
        lz = (z - (cz - 4)) / 8.0f0
        if (lx*lx + ly*ly + lz*lz) < 1.0f0
            vol_mask[x, y, z] = 1.0f0
        end
    end

    spacing = (1.0, 1.0, 1.5)

    textureSpec_ct = TextureSpec{Float32}(
        name="CT",
        isMainImage=true,
        color=RGB(1.0, 1.0, 1.0),
        minAndMaxValue=Float32.([-150, 250])
    )

    textureSpec_mask = TextureSpec{Float32}(
        name="Mask",
        isMainImage=false,
        color=RGB(1.0, 0.2, 0.2),
        minAndMaxValue=Float32.([0, 1]),
        maskContribution=0.5f0,
        isEditable=true
    )

    if quad
        # Axial, Coronal, Sagittal
        vol_ct_axial = vol_ct
        vol_mask_axial = vol_mask
        spacing_axial = spacing

        vol_ct_coronal = permutedims(vol_ct, (1, 3, 2))
        vol_mask_coronal = permutedims(vol_mask, (1, 3, 2))
        spacing_coronal = (spacing[1], spacing[3], spacing[2])

        vol_ct_sagittal = permutedims(vol_ct, (2, 3, 1))
        vol_mask_sagittal = permutedims(vol_mask, (2, 3, 1))
        spacing_sagittal = (spacing[2], spacing[3], spacing[1])

        voxelDataTupleVector = Vector{Vector{Any}}([
            Any[("CT", vol_ct_axial), ("Mask", vol_mask_axial)],
            Any[("Mask", vol_mask_axial)],
            Any[("CT", vol_ct_coronal), ("Mask", vol_mask_coronal)],
            Any[("CT", vol_ct_sagittal), ("Mask", vol_mask_sagittal)]
        ])

        textureSpecArray = Vector{Vector{TextureSpec}}([
            TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)]
        ])

        spacings = [[spacing_axial], [spacing_axial], [spacing_coronal], [spacing_sagittal]]
        origins = [[(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]]
    else
        voxelDataTupleVector = Vector{Vector{Any}}([
            Any[("CT", vol_ct), ("Mask", vol_mask)]
        ])
        textureSpecArray = Vector{Vector{TextureSpec}}([
            TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)]
        ])
        spacings = [[spacing]]
        origins = [[(0.0, 0.0, 0.0)]]
    end

    svVertAndInd = Dict{String, Vector}()
    dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

    println("Opening MedEye3D window...")
    mainViewer = SegmentationDisplay.displayImage(
        dummyStudySrc;
        textureSpecArray=textureSpecArray,
        voxelDataTupleVector=voxelDataTupleVector,
        spacings=spacings,
        origins=origins,
        fractionOfMainImage=Float32(1.0),
        windowWidth=1280,
        svVertAndInd=svVertAndInd,
        quadView=quad
    )

    println("MedEye3D window active. Entering event wait loop.")
    # Keep application alive while viewer channel is active
    try
        while isopen(mainViewer.channel)
            sleep(0.1)
        end
    catch
        # Channel closed or window terminated
    end
    println("MedEye3D session finished.")
end

"""
    launch_from_file(file_path::String; quad::Bool=true)

Loads and visualizes a medical image (NIfTI, DICOM, etc.) from `file_path`.
"""
function launch_from_file(file_path::String; quad::Bool=true)
    println("Loading medical image: ", file_path)
    if !isfile(file_path)
        @error "Provided file path does not exist: $file_path"
        return
    end

    med_img = MedImages.load_image(file_path, "")
    vol_img = Float32.(med_img.voxel_data)
    spacing = Tuple(Float64.(med_img.spacing))
    origin = Tuple(Float64.(med_img.origin))

    vol_mask = zeros(Float32, size(vol_img)...)

    min_val, max_val = Float32(minimum(vol_img)), Float32(maximum(vol_img))
    if min_val == max_val
        max_val += 1.0f0
    end

    textureSpec_img = TextureSpec{Float32}(
        name="MainImage",
        isMainImage=true,
        color=RGB(1.0, 1.0, 1.0),
        minAndMaxValue=Float32.([min_val, max_val])
    )

    textureSpec_mask = TextureSpec{Float32}(
        name="Mask",
        isMainImage=false,
        color=RGB(1.0, 0.0, 0.0),
        minAndMaxValue=Float32.([0, 1]),
        maskContribution=0.5f0,
        isEditable=true
    )

    if quad
        vol_img_coronal = permutedims(vol_img, (1, 3, 2))
        vol_mask_coronal = permutedims(vol_mask, (1, 3, 2))
        spacing_coronal = (spacing[1], spacing[3], spacing[2])

        vol_img_sagittal = permutedims(vol_img, (2, 3, 1))
        vol_mask_sagittal = permutedims(vol_mask, (2, 3, 1))
        spacing_sagittal = (spacing[2], spacing[3], spacing[1])

        voxelDataTupleVector = Vector{Vector{Any}}([
            Any[("MainImage", vol_img), ("Mask", vol_mask)],
            Any[("Mask", vol_mask)],
            Any[("MainImage", vol_img_coronal), ("Mask", vol_mask_coronal)],
            Any[("MainImage", vol_img_sagittal), ("Mask", vol_mask_sagittal)]
        ])

        textureSpecArray = Vector{Vector{TextureSpec}}([
            TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)],
            TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)]
        ])

        spacings = [[spacing], [spacing], [spacing_coronal], [spacing_sagittal]]
        origins = [[origin], [origin], [origin], [origin]]
    else
        voxelDataTupleVector = Vector{Vector{Any}}([
            Any[("MainImage", vol_img), ("Mask", vol_mask)]
        ])
        textureSpecArray = Vector{Vector{TextureSpec}}([
            TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)]
        ])
        spacings = [[spacing]]
        origins = [[origin]]
    end

    svVertAndInd = Dict{String, Vector}()
    dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

    println("Opening MedEye3D window for: ", basename(file_path))
    mainViewer = SegmentationDisplay.displayImage(
        dummyStudySrc;
        textureSpecArray=textureSpecArray,
        voxelDataTupleVector=voxelDataTupleVector,
        spacings=spacings,
        origins=origins,
        fractionOfMainImage=Float32(1.0),
        windowWidth=1360,
        svVertAndInd=svVertAndInd,
        quadView=quad
    )

    try
        while isopen(mainViewer.channel)
            sleep(0.1)
        end
    catch
    end
end

"""
    run_app(args::Vector{String})

Dispatches CLI actions: help, version, image loading, or interactive study selector GUI.
"""
function run_app(args::Vector{String})
    println("MedEye3D standalone runtime initializing...")
    println("Arguments: ", args)
    println("Julia version: ", VERSION)
    println("Threads available: ", Threads.nthreads())

    if "--help" in args || "-h" in args
        println("""
        MedEye3D - High-Performance 3D Medical Image Annotation & Visualization
        
        Usage:
          MedEye3D.exe [options] [image_file]

        Options:
          -h, --help        Show this help message
          -v, --version     Show version information
          --demo            Launch synthetic 3D phantom viewer
          [file_path]       Open medical image file (.nii, .nii.gz, .mha, .h5)
        """)
        return
    end

    if "--version" in args || "-v" in args
        println("MedEye3D Version 0.5.8 (x86_64-w64-mingw32)")
        return
    end

    file_args = filter(a -> !startswith(a, "-"), args)
    if !isempty(file_args) && isfile(file_args[1])
        launch_from_file(file_args[1]; quad=true)
    elseif "--demo" in args
        launch_demo(; quad=true)
    else
        println("Opening MedEye3D File Selector / Demo Launcher...")
        action, path = MedEye3d.StudySelectorWindow.prompt_open_or_demo()
        if action == :file && isfile(path)
            println("Selected file: ", path)
            launch_from_file(path; quad=true)
        else
            println("No file selected. Launching interactive 3D demo phantom visualizer...")
            launch_demo(; quad=true)
        end
    end
end

"""
    julia_main()::Cint

Standard C-compatible entrypoint function required by PackageCompiler.jl.
Captures stdio and stderr into real-time flushed log files in `%APPDATA%\\MedEye3D\\logs`,
initializes loggers, catches unhandled exceptions, and returns exit code 0.
"""
function julia_main()::Cint
    log_dir = get_writable_log_dir()
    out_log_path = joinpath(log_dir, "medeye3d_output.log")
    err_log_path = joinpath(log_dir, "medeye3d_error.log")
    session_log_path = joinpath(log_dir, "medeye3d_session.log")

    # Rotate previous logs safely
    try
        if isfile(out_log_path)
            mv(out_log_path, joinpath(log_dir, "medeye3d_output_prev.log"), force=true)
        end
        if isfile(err_log_path)
            mv(err_log_path, joinpath(log_dir, "medeye3d_error_prev.log"), force=true)
        end
    catch
    end

    out_file = open(out_log_path, "w")
    err_file = open(err_log_path, "w")

    init_msg = """
    =======================================================
     MedEye3D Session Started: $(Dates.now())
     Log Directory: $log_dir
     Output Log:    $out_log_path
     Error Log:     $err_log_path
     Julia Version: $VERSION
     Threads:       $(Threads.nthreads())
     Arguments:     $ARGS
    =======================================================
    """
    println(out_file, init_msg)
    flush(out_file)

    try
        open(session_log_path, "a") do sess
            println(sess, init_msg)
            flush(sess)
        end
    catch
    end

    exit_code = Cint(0)
    redirect_stdio(stdout=out_file, stderr=err_file) do
        logger = SimpleLogger(err_file, Logging.Info)
        with_logger(logger) do
            try
                run_app(ARGS)
            catch e
                exit_code = Cint(1)
                err_msg = "FATAL UNHANDLED EXCEPTION: $(sprint(showerror, e))\n$(sprint(Base.show_backtrace, catch_backtrace()))"
                println(stderr, err_msg)
                flush(stderr)
                try
                    open(session_log_path, "a") do sess
                        println(sess, "ERROR [$(Dates.now())]: $err_msg")
                        flush(sess)
                    end
                catch
                end
            finally
                end_msg = "MedEye3D Session Ended: $(Dates.now()) (Exit code: $exit_code)\n"
                println(stdout, end_msg)
                flush(stdout)
                flush(stderr)
                try
                    open(session_log_path, "a") do sess
                        println(sess, end_msg)
                        flush(sess)
                    end
                catch
                end
                try close(out_file) catch end
                try close(err_file) catch end
            end
        end
    end

    return exit_code
end

end # module MedEye3dApp
