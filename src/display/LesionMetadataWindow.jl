module LesionMetadataWindow

using GLMakie
using Observables

export create_metadata_window

const CATEGORIES = ["Bone Meta", "Lymph Node Meta", "Prostate", "Organ Meta"]
const MANAGEMENT = ["Monitor", "Next study in 1 month", "Next study in 2 months", "Next study in 6 months", "Next study in 13 months", "Next study in 18 months", "Next study in 24 months", "Next study in 36 months", "FDG PET CT", "MRI", "Contrast CT", "Biopsy", "None"]
const SUV_Q = ["SUV max < liver", "SUV max > liver", "SUV max = liver", "Unspecified bone uptake"]
const INSIDE = ["Sclerotic/Blastic/Ivory (>1000 HU)", "Lytic/Lucent", "Mixed Lytic & Sclerotic", "Ground-Glass / Fibrous (70-130 HU)", "Fluid-Filled/Cystic (Water Density)", "Fat Density / Trapped Fat", "Central Necrosis (Low Density)"]
const BORDERS = ["Smooth / Well-Defined Margins", "Spiculated / Feathered", "Moth-Eaten", "Reactive Sclerotic Rim", "Ill-Defined / Permeative", "Serpiginous (Snake-like) Margin", "Overhanging Edges (Apple Core)"]
const AROUND_A = ["Infiltrative (Replaces Marrow Fat)", "Non-Infiltrative (Spares Marrow Fat)"]
const AROUND_B = ["None", "Thick / Solid Reaction (Callus)", "Aggressive / Sunburst / Spiculated", "Codman Triangle"]
const AROUND_C = ["Cortical Breakthrough", "Soft-Tissue Edema (Halo Sign)", "Perivesical Fat Stranding", "Direct Bladder/Rectal Infiltration", "Dural Tail Sign", "Vacuum Cleft Sign"]
const SHAPES = ["Oval / Bean-Shaped", "Round", "Teardrop / Comma-Shaped", "Parallel to the Long Axis of Bone", "Horizontal"]

function create_metadata_window(active_lesion_id::Observable{String}, ui_hooks::Dict{Symbol, Observable})
    fig = Figure(resolution = (800, 1000), backgroundcolor = :white)

    # Main Layout
    gl = GridLayout(fig[1, 1], alignmode = Outside(10))

    Label(gl[1, 1:3], "Slicer Lesion Text Extension (Julia Port)", textsize = 24, font = "bold", halign = :center)

    row = 2
    # Viewport & Windowing Panel
    Label(gl[row, 1:3], "Viewport Controls", font="bold")
    row += 1
    btn_prev_slice = Button(gl[row, 1], label="<< Prev Slice")
    btn_next_slice = Button(gl[row, 2], label="Next Slice >>")
    btn_toggle_lesion = Button(gl[row, 3], label="Toggle Lesion")
    on(btn_prev_slice.clicks) do _ ui_hooks[:scroll][] = -1 end
    on(btn_next_slice.clicks) do _ ui_hooks[:scroll][] = 1 end

    row += 1
    btn_axial = Button(gl[row, 1], label="Axial")
    btn_coronal = Button(gl[row, 2], label="Coronal")
    btn_sagittal = Button(gl[row, 3], label="Sagittal")

    row += 1
    Label(gl[row, 1:3], "CT Window Presets", font="bold")
    row += 1
    btn_soft_tissue = Button(gl[row, 1], label="Soft Tissue")
    btn_bone = Button(gl[row, 2], label="Bone")
    btn_lung = Button(gl[row, 3], label="Lung")
    
    # Soft Tissue (W:400, L:40) -> Min: -160, Max: 240
    on(btn_soft_tissue.clicks) do _ ui_hooks[:windowing][] = (-160.0f0, 240.0f0) end
    # Bone (W:1500, L:300) -> Min: -450, Max: 1050
    on(btn_bone.clicks) do _ ui_hooks[:windowing][] = (-450.0f0, 1050.0f0) end
    # Lung (W:1500, L:-600) -> Min: -1350, Max: 150
    on(btn_lung.clicks) do _ ui_hooks[:windowing][] = (-1350.0f0, 150.0f0) end

    row += 1
    Label(gl[row, 1:3], "Segmentation Mini Manager", font="bold")
    row += 1
    btn_add_auto = Button(gl[row, 1], label="Add Auto-PET")
    btn_gen_manual = Button(gl[row, 2], label="Gen Manual")
    btn_sync = Button(gl[row, 3], label="Sync Lesion")
    on(btn_sync.clicks) do _ ui_hooks[:sync_lesion][] = true end

    row += 1
    btn_paint = Button(gl[row, 1], label="Paint")
    btn_erase = Button(gl[row, 2], label="Erase")
    on(btn_paint.clicks) do _ ui_hooks[:paint_val][] = 1 end
    on(btn_erase.clicks) do _ ui_hooks[:paint_val][] = 0 end

    row += 1
    Label(gl[row, 1], "Active Lesion ID:", font="bold", halign=:right)
    Label(gl[row, 2], @lift(string($active_lesion_id)), halign=:left)

    row += 1
    # Dropdowns for metadata
    Label(gl[row, 1], "Category:", halign=:right)
    menu_category = Menu(gl[row, 2], options = CATEGORIES)

    row += 1
    Label(gl[row, 1], "Management:", halign=:right)
    menu_management = Menu(gl[row, 2], options = MANAGEMENT)

    row += 1
    Label(gl[row, 1], "SUV Q:", halign=:right)
    menu_suvq = Menu(gl[row, 2], options = SUV_Q)

    row += 1
    Label(gl[row, 1], "Inside (Matrix/Density):", halign=:right)
    menu_inside = Menu(gl[row, 2], options = INSIDE)

    row += 1
    Label(gl[row, 1], "Borders (Margins):", halign=:right)
    menu_borders = Menu(gl[row, 2], options = BORDERS)

    row += 1
    Label(gl[row, 1], "Around A (Marrow):", halign=:right)
    menu_around_a = Menu(gl[row, 2], options = AROUND_A)

    row += 1
    Label(gl[row, 1], "Around B (Periosteum):", halign=:right)
    menu_around_b = Menu(gl[row, 2], options = AROUND_B)

    row += 1
    Label(gl[row, 1], "Around C (Extramural):", halign=:right)
    menu_around_c = Menu(gl[row, 2], options = AROUND_C)

    row += 1
    Label(gl[row, 1], "Shape:", halign=:right)
    menu_shape = Menu(gl[row, 2], options = SHAPES)

    row += 1
    Label(gl[row, 1], "Other (RadLex):", halign=:right)
    textbox_radlex = Textbox(gl[row, 2], placeholder = "e.g., RID5961 - Sclerotic")

    row += 1
    # Save button
    save_button = Button(gl[row, 1:2], label = "Save Metadata to Memory")
    
    row += 1
    # Textbox for AI Radiological Report Output
    Label(gl[row, 1:2], "Radiological Report Output:", font="bold", halign=:left)
    row += 1
    report_output = Textbox(gl[row, 1:2], width = 600, height = 150, text = "Click Generate to create report...")
    row += 1
    generate_button = Button(gl[row, 1:2], label = "Generate Report (Mock AI)")

    # Example interactions / data storage dict (In reality this maps to MRML/Segments)
    lesion_db = Dict{String, Dict{String, Any}}()

    # Callback when Active Lesion Changes
    on(active_lesion_id) do id
        if !haskey(lesion_db, id)
            lesion_db[id] = Dict{String, Any}()
        end
        # Load from DB
        data = lesion_db[id]
        menu_category.selection[] = get(data, "Category", nothing)
        menu_management.selection[] = get(data, "Management", nothing)
        menu_suvq.selection[] = get(data, "SUV Q", nothing)
        menu_inside.selection[] = get(data, "Inside", nothing)
        menu_borders.selection[] = get(data, "Borders", nothing)
        menu_around_a.selection[] = get(data, "Around A", nothing)
        menu_around_b.selection[] = get(data, "Around B", nothing)
        menu_around_c.selection[] = get(data, "Around C", nothing)
        menu_shape.selection[] = get(data, "Shape", nothing)
        textbox_radlex.stored_string[] = get(data, "Other", "")
    end

    # Callback for Save button
    on(save_button.clicks) do _
        id = active_lesion_id[]
        lesion_db[id] = Dict(
            "Category" => menu_category.selection[],
            "Management" => menu_management.selection[],
            "SUV Q" => menu_suvq.selection[],
            "Inside" => menu_inside.selection[],
            "Borders" => menu_borders.selection[],
            "Around A" => menu_around_a.selection[],
            "Around B" => menu_around_b.selection[],
            "Around C" => menu_around_c.selection[],
            "Shape" => menu_shape.selection[],
            "Other" => textbox_radlex.stored_string[]
        )
        println("Saved metadata for lesion \$id")
    end

    # Callback for Generate button
    on(generate_button.clicks) do _
        id = active_lesion_id[]
        report_output.stored_string[] = "The lesion (\$id) located in the \$(menu_category.selection[]) shows a \$(menu_borders.selection[]) margin and a \$(menu_shape.selection[]) shape. Management plan: \$(menu_management.selection[])."
    end

    display(fig)
    return fig
end

end
