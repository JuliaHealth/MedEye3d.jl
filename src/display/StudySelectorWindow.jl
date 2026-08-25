module StudySelectorWindow

using GLMakie
using Observables
using Dates
import GLFW

export open_study_selector, scan_medical_files

"""
    scan_medical_files(dir::String) -> Vector{String}

Scans `dir` for medical imaging files (.nii, .nii.gz, .mha, .mhd, .h5, .hdf5, .dcm).
"""
function scan_medical_files(dir::String)::Vector{String}
    if !isdir(dir)
        return String[]
    end
    valid_exts = [".nii", ".nii.gz", ".mha", ".mhd", ".h5", ".hdf5", ".dcm"]
    found = String[]
    try
        for (root, _, files) in walkdir(dir)
            for f in files
                fl = lowercase(f)
                if any(ext -> endswith(fl, ext), valid_exts)
                    rel = relpath(joinpath(root, f), dir)
                    push!(found, rel)
                end
            end
            # Do not recurse too deep to prevent long pauses
            if length(found) > 100
                break
            end
        end
    catch e
        @warn "Error scanning directory $dir: $e"
    end
    return sort(found)
end

"""
    browse_folder_dialog(initial_dir::String="") -> String

Opens a native Windows FolderBrowserDialog to pick a directory.
"""
function browse_folder_dialog(initial_dir::String="")::String
    if Sys.iswindows()
        init_cmd = isdir(initial_dir) ? "\$dialog.SelectedPath = '$initial_dir';" : ""
        cmd = """
        Add-Type -AssemblyName System.Windows.Forms
        \$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        \$dialog.Description = 'Select Medical Dataset Folder'
        $init_cmd
        if (\$dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            Write-Output \$dialog.SelectedPath
        }
        """
        try
            out = read(`powershell -NoProfile -Command $cmd`, String)
            selected = strip(out)
            if isdir(selected)
                return selected
            end
        catch e
            @warn "Failed to launch native folder dialog: $e"
        end
    end
    return initial_dir
end

"""
    browse_file_dialog(initial_dir::String="") -> String

Opens a native Windows OpenFileDialog to pick a medical image file.
"""
function browse_file_dialog(initial_dir::String="")::String
    if Sys.iswindows()
        init_cmd = isdir(initial_dir) ? "\$dialog.InitialDirectory = '$initial_dir';" : ""
        cmd = """
        Add-Type -AssemblyName System.Windows.Forms
        \$dialog = New-Object System.Windows.Forms.OpenFileDialog
        \$dialog.Title = 'Select Medical Image File'
        \$dialog.Filter = 'Medical Images (*.nii;*.nii.gz;*.mha;*.h5)|*.nii;*.nii.gz;*.mha;*.h5|All files (*.*)|*.*'
        $init_cmd
        if (\$dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            Write-Output \$dialog.FileName
        }
        """
        try
            out = read(`powershell -NoProfile -Command $cmd`, String)
            selected = strip(out)
            if isfile(selected)
                return selected
            end
        catch e
            @warn "Failed to launch native file dialog: $e"
        end
    end
    return ""
end

"""
    open_study_selector(; on_select::Function=(path, quad)->nothing, on_demo::Function=(quad)->nothing)

Opens an interactive GLMakie Study and Dataset Selection panel.
"""
function open_study_selector(; on_select::Union{Function,Nothing}=nothing, on_demo::Union{Function,Nothing}=nothing)
    default_dir = normpath(joinpath(@__DIR__, "..", ".."))
    if !isdir(default_dir)
        default_dir = homedir()
    end

    # Observables
    current_dir = Observable(default_dir)
    files_list = Observable(scan_medical_files(default_dir))
    selected_file_index = Observable(1)
    status_text = Observable("Ready. Select a folder or file to visualize.")
    quad_view_obs = Observable(true)

    # Initial options for Menu
    menu_options = Observable(isempty(files_list[]) ? ["(No medical files found in current directory)"] : files_list[])

    on(files_list) do fl
        if isempty(fl)
            menu_options[] = ["(No medical files found in current directory)"]
            selected_file_index[] = 1
        else
            menu_options[] = fl
            selected_file_index[] = 1
        end
    end

    # GLMakie Figure
    fig = Figure(size=(850, 520), backgroundcolor=RGBf(0.12, 0.13, 0.16))

    # Header
    Label(fig[1, 1:3], "MedEye3D - Study & Data Selector",
        fontsize=22, font=:bold, color=RGBf(0.95, 0.95, 0.98), halign=:left)
    Label(fig[2, 1:3], "Select a medical dataset folder, open an image file, or launch a synthetic 3D phantom.",
        fontsize=13, color=RGBf(0.70, 0.73, 0.78), halign=:left)

    # Folder Input Row
    Label(fig[3, 1], "Folder Path:", fontsize=13, font=:bold, color=RGBf(0.85, 0.88, 0.92), halign=:left)
    folder_tb = Textbox(fig[3, 2], placeholder="Enter or paste folder path...", stored_string=current_dir[],
        fontsize=13, width=480, halign=:left)

    btn_browse_folder = Button(fig[3, 3], label="Browse Folder...", fontsize=12,
        buttoncolor=RGBf(0.22, 0.26, 0.32), labelcolor=RGBf(0.9, 0.9, 0.95))

    # File / Quick actions Row
    Label(fig[4, 1], "Detected Files:", fontsize=13, font=:bold, color=RGBf(0.85, 0.88, 0.92), halign=:left)
    file_menu = Menu(fig[4, 2], options=menu_options, default=menu_options[][1], fontsize=13, width=480, halign=:left)

    btn_browse_file = Button(fig[4, 3], label="Browse File...", fontsize=12,
        buttoncolor=RGBf(0.22, 0.26, 0.32), labelcolor=RGBf(0.9, 0.9, 0.95))

    # Options Row: QuadView Toggle
    Label(fig[5, 1], "Display Mode:", fontsize=13, font=:bold, color=RGBf(0.85, 0.88, 0.92), halign=:left)
    quad_toggle = Toggle(fig[5, 2], active=quad_view_obs[], halign=:left)
    Label(fig[5, 2], "  Multi-Planar QuadView (Axial, Coronal, Sagittal, 3D)",
        fontsize=12, color=RGBf(0.8, 0.82, 0.85), halign=:left, padding=(35, 0, 0, 0))

    connect!(quad_view_obs, quad_toggle.active)

    # Status Label
    status_label = Label(fig[6, 1:3], status_text, fontsize=12, color=RGBf(0.6, 0.8, 1.0), halign=:left)

    # Bottom Actions Grid
    btn_grid = GridLayout(fig[7, 1:3])
    
    btn_demo = Button(btn_grid[1, 1], label="Launch 3D Demo", fontsize=13, font=:bold,
        buttoncolor=RGBf(0.18, 0.42, 0.70), labelcolor=RGBf(1.0, 1.0, 1.0), height=42)

    btn_open = Button(btn_grid[1, 2], label="Open Selected Study", fontsize=13, font=:bold,
        buttoncolor=RGBf(0.16, 0.58, 0.36), labelcolor=RGBf(1.0, 1.0, 1.0), height=42)

    btn_logs = Button(btn_grid[1, 3], label="View Logs", fontsize=12,
        buttoncolor=RGBf(0.24, 0.28, 0.35), labelcolor=RGBf(0.88, 0.90, 0.95), height=42)

    # Interactions
    on(folder_tb.stored_string) do new_path
        if isdir(new_path)
            current_dir[] = normpath(new_path)
            files_list[] = scan_medical_files(new_path)
            status_text[] = "Found $(length(files_list[])) medical files in: $new_path"
        else
            status_text[] = "Specified path is not a valid directory: $new_path"
        end
    end

    on(btn_browse_folder.clicks) do _
        picked = browse_folder_dialog(current_dir[])
        if !isempty(picked) && isdir(picked)
            folder_tb.stored_string[] = picked
            current_dir[] = picked
            files_list[] = scan_medical_files(picked)
            status_text[] = "Found $(length(files_list[])) medical files in: $picked"
        end
    end

    on(btn_browse_file.clicks) do _
        picked = browse_file_dialog(current_dir[])
        if !isempty(picked) && isfile(picked)
            parent_dir = dirname(picked)
            folder_tb.stored_string[] = parent_dir
            current_dir[] = parent_dir
            files_list[] = scan_medical_files(parent_dir)
            
            # Auto-select picked file in menu
            rel = relpath(picked, parent_dir)
            idx = findfirst(==(rel), files_list[])
            if idx !== nothing
                file_menu.selection[] = rel
            end
            status_text[] = "Selected file: $picked"
        end
    end

    on(btn_logs.clicks) do _
        appdata = get(ENV, "APPDATA", "")
        log_dir = !isempty(appdata) && isdir(joinpath(appdata, "MedEye3D", "logs")) ?
            joinpath(appdata, "MedEye3D", "logs") : tempdir()
        status_text[] = "Opening log folder: $log_dir"
        if Sys.iswindows()
            try
                run(`explorer.exe $log_dir`, wait=false)
            catch e
                @warn "Could not open explorer: $e"
            end
        end
    end

    screen = display(fig)

    # Open Selected Button handler
    on(btn_open.clicks) do _
        selected_rel = file_menu.selection[]
        if selected_rel !== nothing && !startswith(selected_rel, "(")
            full_path = joinpath(current_dir[], selected_rel)
            if isfile(full_path)
                status_text[] = "Opening study: $full_path..."
                if on_select !== nothing
                    on_select(full_path, quad_view_obs[])
                end
            else
                status_text[] = "File does not exist: $full_path"
            end
        else
            status_text[] = "Please select a valid medical file from the list or click 'Browse File...'."
        end
    end

    # Launch Demo Button handler
    on(btn_demo.clicks) do _
        status_text[] = "Launching 3D synthetic phantom visualizer..."
        if on_demo !== nothing
            on_demo(quad_view_obs[])
        end
    end

    return fig, screen
end

end # module StudySelectorWindow
