module StudySelectorWindow

using Dates
import GLFW

export open_study_selector, scan_medical_files, browse_file_dialog, browse_folder_dialog, prompt_open_or_demo

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
            # Prevent scanning excessive files
            if length(found) > 200
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
                return String(selected)
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
        \$dialog.Title = 'MedEye3D - Select Medical Image to Visualize'
        \$dialog.Filter = 'Medical Images (*.nii;*.nii.gz;*.mha;*.h5;*.dcm)|*.nii;*.nii.gz;*.mha;*.h5;*.dcm|NIfTI (*.nii;*.nii.gz)|*.nii;*.nii.gz|MetaImage (*.mha;*.mhd)|*.mha;*.mhd|HDF5 (*.h5;*.hdf5)|*.h5;*.hdf5|All files (*.*)|*.*'
        \$dialog.FilterIndex = 1
        \$dialog.RestoreDirectory = \$true
        $init_cmd
        if (\$dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            Write-Output \$dialog.FileName
        }
        """
        try
            out = read(`powershell -NoProfile -Command $cmd`, String)
            selected = strip(out)
            if isfile(selected)
                return String(selected)
            end
        catch e
            @warn "Failed to launch native file dialog: $e"
        end
    end
    return ""
end

"""
    prompt_open_or_demo(default_dir::String="") -> Tuple{Symbol, String}

Prompts the user via native Windows file dialog to choose a file or launch the demo phantom.
Returns `(:file, path)` or `(:demo, "")`.
"""
function prompt_open_or_demo(default_dir::String="")::Tuple{Symbol, String}
    selected_file = browse_file_dialog(default_dir)
    if !isempty(selected_file) && isfile(selected_file)
        return (:file, selected_file)
    else
        return (:demo, "")
    end
end

"""
    open_study_selector(; on_select::Union{Function,Nothing}=nothing, on_demo::Union{Function,Nothing}=nothing)

Opens the native medical file selector or launches the 3D demo.
"""
function open_study_selector(; on_select::Union{Function,Nothing}=nothing, on_demo::Union{Function,Nothing}=nothing)
    action, path = prompt_open_or_demo()
    if action == :file && on_select !== nothing
        on_select(path, true)
    elseif on_demo !== nothing
        on_demo(true)
    end
end

end # module StudySelectorWindow
