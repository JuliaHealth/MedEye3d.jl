# build_app.jl
# Standalone application build script utilizing PackageCompiler.jl to compile
# MedEye3D into a relocatable Windows application bundle (bin, share, sysimage.dll).

using Pkg

println("===============================================================")
println(" MedEye3D - Standalone Windows Application Build Pipeline")
println("===============================================================")

# 1. Ensure PackageCompiler is available in the current environment
println("\n[Step 1/4] Checking build dependencies (PackageCompiler.jl)...")
try
    using PackageCompiler
    println("✓ PackageCompiler is already available.")
catch
    println("Installing PackageCompiler into project environment...")
    Pkg.add("PackageCompiler")
    using PackageCompiler
    println("✓ PackageCompiler installed successfully.")
end

# 2. Define build paths
project_root = normpath(joinpath(@__DIR__, "..", ".."))
build_output_dir = joinpath(project_root, "build", "MedEye3D_dist")
precompile_script = joinpath(@__DIR__, "precompile_app.jl")
app_icon_path = joinpath(@__DIR__, "app_icon.ico")
license_path = joinpath(project_root, "LICENSE")

println("\n[Step 2/4] Configuration:")
println("  - Source Root:        $project_root")
println("  - Output Directory:   $build_output_dir")
println("  - Precompile Script:  $precompile_script")
println("  - Icon:               $app_icon_path")

if !isfile(app_icon_path)
    println("App icon not found, generating icon...")
    include(joinpath(@__DIR__, "generate_icon.ps1"))
end

# Ensure JULIA_NUM_THREADS does not hang non-incremental sysimage builds
if haskey(ENV, "JULIA_NUM_THREADS")
    println("Clearing JULIA_NUM_THREADS environment variable to avoid compiler hangs...")
    delete!(ENV, "JULIA_NUM_THREADS")
end

# 3. Compile standalone application
println("\n[Step 3/4] Compiling Standalone Application with PackageCompiler.create_app()...")
println("This step compiles Julia runtime, dependencies, sysimage, and C-entrypoint.")
println("Please wait, this process may take several minutes...\n")

t_start = time()

PackageCompiler.create_app(
    project_root,
    build_output_dir;
    executables=["MedEye3D" => "julia_main"],
    precompile_execution_file=precompile_script,
    include_transitive_dependencies=true,
    include_lazy_artifacts=true,
    force=true,
    incremental=false,
    filter_stdlibs=false
)

elapsed_min = round((time() - t_start) / 60.0, digits=2)
println("\n✓ Compilation completed successfully in $elapsed_min minutes!")

# 4. Post-processing & Asset Copying
println("\n[Step 4/4] Copying assets and resources to distribution bundle...")
bin_dir = joinpath(build_output_dir, "bin")

if isfile(app_icon_path)
    cp(app_icon_path, joinpath(bin_dir, "app_icon.ico"); force=true)
    cp(app_icon_path, joinpath(build_output_dir, "app_icon.ico"); force=true)
    println("✓ Copied application icon.")
end

if isfile(license_path)
    cp(license_path, joinpath(build_output_dir, "LICENSE.txt"); force=true)
    println("✓ Copied LICENSE.")
end

# Create a default configuration / info file
info_file = joinpath(build_output_dir, "app_info.json")
write(info_file, """
{
    "name": "MedEye3D",
    "version": "0.5.8",
    "arch": "x86_64",
    "entrypoint": "bin/MedEye3D.exe",
    "description": "High-Performance 3D Medical Image Annotation & Visualization Software"
}
""")
println("✓ Generated app_info.json.")

println("\n===============================================================")
println(" MedEye3D Application successfully built at:")
println(" $build_output_dir")
println(" Executable: $(joinpath(bin_dir, "MedEye3D.exe"))")
println("===============================================================")
