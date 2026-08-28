#!/usr/bin/env julia
# prefill_anatomy.jl — Pre-fill all metadata, anatomy, measurements, and clinical rules for all lesions
#
# This script:
# 1. Loads the scene hierarchy and all timepoints
# 2. Loads the max_anatomy atlas, labels, and ontology mapping
# 3. For every lesion across all timepoints:
#    - Computes centroid [cx, cy, cz], volume_mm3, volume_cc, diameter_mm, bounding box [z_min, z_max]
#    - Samples anatomy atlas at centroid (and mode over lesion voxels) -> BaseAnatomy, BaseAnatomySide, Anatomic Location, Anatomical Sublocation, LesionType
#    - Applies Bone subsegmentation / overlap check -> Bone Meta classification
#    - Applies Edge Slices Rule: if z_min <= 2 or z_max >= total_z - 1 -> Technical Artifact, Certainty = 0
#    - Applies Physiological Excretion Rule: if bladder / kidney -> Urine / Renal Excretion, Certainty = 0
#    - Computes SUV max at centroid & background organ SUV means (liver, parotid, blood pool)
#    - Sets tracking name, CT correlate default ("false"), and initial button states
# 4. Saves complete lesion database and _GlobalAppState into HDF5 and JSON

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using JSON, NIfTI, HDF5, Statistics, LinearAlgebra, MedImages

# ── Paths ────────────────────────────────────────────────────────────────────
data_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "data", "pat_6_files")
ontology_path = joinpath(@__DIR__, "..", "..", "data", "max_anatomy_to_ontology.json")
h5_annot_path = joinpath(homedir(), "medeye3d_lesion_annotations.h5")
json_annot_path = joinpath(homedir(), "medeye3d_lesion_annotations.json")
case_h5_path = joinpath(data_dir, "medeye3d_lesion_annotations.h5")
case_json_path = joinpath(data_dir, "medeye3d_lesion_annotations.json")

if !isfile(ontology_path)
    error("Ontology mapping not found: $ontology_path")
end

# ── Load ontology ────────────────────────────────────────────────────────────
ontology = JSON.parsefile(ontology_path)
println("Loaded ontology: $(length(ontology)) entries")

# ── Load scene hierarchy ────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "lib", "SceneHierarchy.jl"))
using .SceneHierarchy
studies = parse_studies_from_hierarchy(data_dir)
println("Found $(length(studies)) timepoints in $(data_dir)")

# ── Load existing database if available ─────────────────────────────────────
using MedEye3d
using MedEye3d.LesionMetadataWindow
using MedEye3d.LesionAssociation

existing_db = load_annotations(json_annot_path)
if isempty(existing_db) && isfile(h5_annot_path)
    existing_db = load_annotations_hdf5(h5_annot_path)
end
println("Loaded $(length(existing_db)) existing annotation entries for preserving user notes")
db = Dict{String, Any}()

total_lesions_processed = 0
total_artifacts_flagged = 0

for (s_idx, study) in enumerate(studies)
    modality, orig_tp, date_str = study[1], study[2], study[3]
    ct_fname = study[4]
    pet_fname = study[5]
    mask_fname = study[6]
    node_name = study[7]
    tfm_fname = study[8]
    max_anat_src = study[10]
    max_anat_lbl = study[11]
    skellytour_source = study[12]
    
    println("\n=== Processing Study $s_idx: $modality TP $orig_tp ($node_name) ===")
    
    # Load mask
    if isempty(mask_fname)
        println("  No mask for study $s_idx, skipping")
        continue
    end
    mask_path = joinpath(data_dir, mask_fname)
    if !isfile(mask_path)
        println("  Mask file not found: $mask_path, skipping")
        continue
    end
    
    mask_nii = NIfTI.niread(mask_path)
    mask_vol = Int16.(mask_nii.raw)
    mask_vol = reverse(mask_vol, dims=2)
    sz = size(mask_vol)
    total_z = sz[3]
    
    # Load CT / Spacing
    ct_path = joinpath(data_dir, ct_fname)
    spacing = (1.0, 1.0, 1.0)
    if isfile(ct_path)
        try
            ct_nii = NIfTI.niread(ct_path)
            hdr = ct_nii.header
            spacing = (Float64(hdr.pixdim[2]), Float64(hdr.pixdim[3]), Float64(hdr.pixdim[4]))
        catch
        end
    end
    voxel_vol_mm3 = spacing[1] * spacing[2] * spacing[3]
    
    # Load PET if available
    pet_path = joinpath(data_dir, pet_fname)
    pet_vol = nothing
    if isfile(pet_path)
        try
            pet_nii = NIfTI.niread(pet_path)
            pet_raw = Float32.(pet_nii.raw)
            pet_vol = reverse(pet_raw, dims=2)
        catch e
            println("  PET load error: $e")
        end
    end
    
    # Load Atlas & Labels
    atlas = nothing
    labels = Dict{Int,String}()
    if !isempty(max_anat_src) && !isempty(max_anat_lbl)
        anat_path = joinpath(data_dir, max_anat_src)
        lbl_path = joinpath(data_dir, max_anat_lbl)
        if isfile(anat_path) && isfile(lbl_path)
            anat_nii = NIfTI.niread(anat_path)
            atlas_raw = UInt16.(anat_nii.raw)
            atlas = reverse(atlas_raw, dims=2)
            labels = Dict{Int,String}(parse(Int, k) => v for (k, v) in JSON.parsefile(lbl_path))
            println("  Loaded max_anatomy atlas $(size(atlas)) with $(length(labels)) labels")
        end
    end
    
    # Load Skellytour for bone overlap check
    skelly_vox = nothing
    if !isempty(skellytour_source)
        sk_path = joinpath(data_dir, skellytour_source)
        if isfile(sk_path)
            sk_nii = NIfTI.niread(sk_path)
            sk_raw = Float32.(sk_nii.raw)
            skelly_vox = reverse(sk_raw, dims=2)
        end
    end
    
    # Reference background SUVs (liver, parotid, blood pool)
    bg_suvs = Dict{String, Float32}("liver" => 2.0f0, "parotid" => 1.5f0, "blood" => 1.8f0)
    if pet_vol !== nothing && atlas !== nothing && !isempty(labels)
        try
            bg_computed = LesionMetadataWindow.compute_background_suvs(pet_vol, atlas, labels)
            for (k, v) in bg_computed
                if v > 0.1f0
                    bg_suvs[k] = v
                end
            end
            println("  Background SUVs: liver=$(round(bg_suvs["liver"], digits=2)), parotid=$(round(bg_suvs["parotid"], digits=2)), blood=$(round(bg_suvs["blood"], digits=2))")
        catch e
            println("  Background SUV computation warning: $e")
        end
    end
    
    # Map all lesions in this timepoint to anatomical organs using expanding radius search
    organ_mapping = Dict{Int,String}()
    if atlas !== nothing && !isempty(labels)
        try
            organ_mapping = LesionAssociation.map_lesions_to_organs(mask_vol, atlas, labels)
            println("  Mapped $(length(organ_mapping)) lesions to anatomy organs")
        catch e
            println("  Organ mapping error: $e")
        end
    end
    
    unique_ids = sort(filter(x -> x > 0, unique(mask_vol)))
    println("  Found $(length(unique_ids)) lesions in mask")
    
    for lid in unique_ids
        global total_lesions_processed += 1
        voxels = findall(x -> x == lid, mask_vol)
        if isempty(voxels)
            continue
        end
        
        # Spatial measurements
        xs = [v[1] for v in voxels]
        ys = [v[2] for v in voxels]
        zs = [v[3] for v in voxels]
        
        cx = round(Int, mean(xs))
        cy = round(Int, mean(ys))
        cz = round(Int, mean(zs))
        
        z_min = minimum(zs)
        z_max = maximum(zs)
        
        vol_voxels = length(voxels)
        vol_mm3 = vol_voxels * voxel_vol_mm3
        vol_cc = vol_mm3 / 1000.0
        diameter_mm = 2.0 * (3.0 * vol_mm3 / (4.0 * pi))^(1.0 / 3.0)
        
        # Anatomy lookup from mapped organs or direct atlas sampling
        organ_name = get(organ_mapping, Int(lid), "")
        if isempty(organ_name) && atlas !== nothing
            cx_c = clamp(cx, 1, size(atlas, 1))
            cy_c = clamp(cy, 1, size(atlas, 2))
            cz_c = clamp(cz, 1, size(atlas, 3))
            anat_val = Int(atlas[cx_c, cy_c, cz_c])
            if anat_val > 0
                organ_name = get(labels, anat_val, "")
            end
        end
        
        # Ontology lookup
        entry = nothing
        if !isempty(organ_name)
            entry = get(ontology, lowercase(organ_name), get(ontology, organ_name, nothing))
        end
        
        base_anatomy = entry !== nothing ? get(entry, "detailed", titlecase(replace(organ_name, "_" => " "))) : (isempty(organ_name) ? "Lesion" : titlecase(replace(organ_name, "_" => " ")))
        base_side = entry !== nothing ? get(entry, "side", "") : ""
        anatomic_loc = entry !== nothing ? get(entry, "anatomic_location", "Solid Organ / Viscera") : "Solid Organ / Viscera"
        anatomic_subloc = entry !== nothing ? get(entry, "anatomical_sublocation", "N/A (General Organ)") : "N/A (General Organ)"
        lesion_type = entry !== nothing ? get(entry, "lesion_type", "Organ Meta") : "Organ Meta"
        
        # Check bone overlap with Skellytour
        is_bone = false
        if skelly_vox !== nothing
            sk_overlap = count([skelly_vox[clamp(v[1],1,size(skelly_vox,1)), clamp(v[2],1,size(skelly_vox,2)), clamp(v[3],1,size(skelly_vox,3))] > 0 for v in voxels])
            if sk_overlap >= min(5, round(Int, 0.05 * vol_voxels))
                is_bone = true
            end
        end
        bone_kws = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula", "humerus", "scapula", "sternum", "skull", "palate", "bone", "spine", "ilium", "ischium", "pubis"]
        if any(kw -> occursin(kw, lowercase(organ_name)), bone_kws) || any(kw -> occursin(kw, lowercase(base_anatomy)), bone_kws)
            is_bone = true
        end
        
        if is_bone
            lesion_type = "Bone Meta"
            if !occursin("Skeleton", anatomic_loc)
                anatomic_loc = (occursin("femur", lowercase(base_anatomy)) || occursin("humerus", lowercase(base_anatomy)) || occursin("tibia", lowercase(base_anatomy))) ? 
                    "Appendicular Skeleton (Limbs, Scapulae, Hands, Feet)" : 
                    "Axial Skeleton (Spine, Pelvis, Ribs, Skull, Sternum, Clavicles)"
            end
            if isempty(anatomic_subloc) || anatomic_subloc == "N/A (General Organ)"
                anatomic_subloc = "Medullary Cavity (Intramedullary/Marrow)"
            end
        elseif occursin("lymph", lowercase(organ_name)) || occursin("node", lowercase(organ_name))
            lesion_type = "Lymph Node Meta"
            anatomic_loc = occursin("pelv", lowercase(organ_name)) ? "Pelvic Lymph Node" : "Distant Lymph Node (Common Iliac, Retroperitoneal, Inguinal, Supraclavicular, Axillary)"
        elseif occursin("prostate", lowercase(organ_name))
            lesion_type = "Prostate"
            anatomic_loc = "Prostate Gland"
            anatomic_subloc = "Prostate Peripheral Zone (PZ)"
        end
        
        # ── Rules Engine ─────────────────────────────────────────────────────
        alt_hypothesis = "None"
        certainty = "3"
        artifact_reason = ""
        
        # Rule 1: Edge-slice reconstruction artifact (first 2 or last 2 axial slices)
        if z_min <= 2 || z_max >= total_z - 1 || cz <= 2 || cz >= total_z - 1
            alt_hypothesis = "Technical Artifact"
            certainty = "0"
            artifact_reason = "Edge slice reconstruction / partial volume artifact (z=$cz/$total_z)"
            global total_artifacts_flagged += 1
        # Rule 2: Urinary excretion / Bladder contamination
        elseif occursin("bladder", lowercase(organ_name)) || occursin("urinary_bladder", lowercase(organ_name))
            alt_hypothesis = "Urine Excretion / Bladder Contamination"
            certainty = "0"
            artifact_reason = "Urinary bladder physiological tracer pooling"
            global total_artifacts_flagged += 1
        # Rule 3: Renal excretion (calyces / renal pelvis)
        elseif occursin("kidney", lowercase(organ_name))
            alt_hypothesis = "Renal Excretion"
            certainty = "0"
            artifact_reason = "Renal parenchymal/pelvicalyceal physiological excretion"
            global total_artifacts_flagged += 1
        # Rule 4: Normal salivary gland uptake
        elseif occursin("parotid", lowercase(organ_name)) || occursin("submandibular", lowercase(organ_name))
            alt_hypothesis = "Normal Physiological Uptake"
            certainty = "0"
            artifact_reason = "Salivary gland physiological uptake"
            global total_artifacts_flagged += 1
        end
        
        # SUV max computation
        suv_max = 0.0f0
        if pet_vol !== nothing
            cx_p = clamp(cx, 1, size(pet_vol, 1))
            cy_p = clamp(cy, 1, size(pet_vol, 2))
            cz_p = clamp(cz, 1, size(pet_vol, 3))
            
            for dz in -1:1, dy in -1:1, dx in -1:1
                nx, ny, nz = cx_p + dx, cy_p + dy, cz_p + dz
                if 1 <= nx <= size(pet_vol, 1) && 1 <= ny <= size(pet_vol, 2) && 1 <= nz <= size(pet_vol, 3)
                    v = pet_vol[nx, ny, nz]
                    if v > suv_max; suv_max = v; end
                end
            end
        end
        suv_str = "Max: $(round(suv_max, digits=1)) ; Parotid: $(round(bg_suvs["parotid"], digits=1)) ; Liver: $(round(bg_suvs["liver"], digits=1)) ; Blood: $(round(bg_suvs["blood"], digits=1))"
        
        # Generate tracking name
        pat_id = "pat6"
        tracking_name = "$(replace(base_anatomy, " " => "_"))_L$(lid)_$(modality)_TP$(orig_tp)_$(pat_id)"
        
        # Build lesion record
        lesion_record = Dict{String, Any}(
            "LesionType" => lesion_type,
            "BaseAnatomy" => base_anatomy,
            "BaseAnatomySide" => base_side,
            "Anatomic Location" => anatomic_loc,
            "Anatomical Sublocation" => anatomic_subloc,
            "Alternative Hypothesis (False Positive)" => alt_hypothesis,
            "Certainty" => certainty,
            "Lesion tracking name?" => tracking_name,
            "SUV max" => suv_str,
            "NoCTCorrelate" => "false",
            "Radioligand Type" => "68Ga-PSMA-11",
            "_Volume_mm3" => string(round(vol_mm3, digits=1)),
            "_Volume_cc" => string(round(vol_cc, digits=3)),
            "_Diameter_mm" => string(round(diameter_mm, digits=1)),
            "_Centroid" => "[$(cx), $(cy), $(cz)]",
            "_Slice_Z" => string(cz),
            "_TimePoint" => string(orig_tp),
            "_Modality" => modality,
            "_NodeName" => node_name
        )
        if !isempty(artifact_reason)
            lesion_record["_ArtifactReason"] = artifact_reason
        end
        
        # Key naming: format as "$lid: $organ_name" and also ensure "$lid" is available
        display_key = isempty(organ_name) ? "$lid: Lesion $lid" : "$lid: $organ_name"
        
        # Merge with existing data if present (preserve user-edited clinical notes/custom fields)
        existing = get_lesion_state(existing_db, display_key)
        for (k, v) in existing
            if startswith(k, "Custom:") || k == "RadiologicalDictation" || k == "RadiologicalDictationEN" || 
               k == "RadiologicalReportOutput" || k == "Comment" || k == "ClinicalNotes" || k == "UserNotes" ||
               (k == "Certainty" && v ∉ ["3", "0"]) || (k == "NoCTCorrelate" && v == "true")
                lesion_record[k] = v
            end
        end
        
        # Store in db under timepoint-specific and display keys
        if s_idx == 1  # Baseline TP 0 (primary viewer study)
            db[display_key] = lesion_record
            db[string(lid)] = lesion_record
            db["TP$(orig_tp)_L$lid"] = lesion_record
            db["$(node_name)_L$lid"] = lesion_record
        else
            db["$(display_key) (TP $orig_tp)"] = lesion_record
            db["TP$(orig_tp)_L$lid"] = lesion_record
            db["$(node_name)_L$lid"] = lesion_record
            if !haskey(db, display_key)
                db[display_key] = lesion_record
            end
            if !haskey(db, string(lid))
                db[string(lid)] = lesion_record
            end
        end
        
        println("    L$lid: $base_anatomy | $anatomic_loc | $lesion_type | Certainty=$certainty | Alt=$alt_hypothesis")
    end
end

# ── Set Global App State ─────────────────────────────────────────────────────
defaults = Dict{String, Any}(
    "CT_Min" => "-150",
    "CT_Max" => "250",
    "PET_Min" => "0",
    "PET_Max" => "10",
    "SPECT_Min" => "0",
    "SPECT_Max" => "10",
    "vis_lesion" => "true",
    "vis_surface" => "true",
    "vis_marrow" => "true",
    "vis_anatomy" => "false"
)
if haskey(existing_db, "_GLOBAL_APP_STATE")
    gst = existing_db["_GLOBAL_APP_STATE"]
    # Merge: keep existing values, fill missing with defaults
    for (k, v) in defaults
        if !haskey(gst, k)
            gst[k] = v
        end
    end
    # Remove legacy keys
    delete!(gst, "windowing")
    db["_GLOBAL_APP_STATE"] = gst
else
    db["_GLOBAL_APP_STATE"] = defaults
end

# ── Save to HDF5 and JSON (both user home and data_dir) ──────────────────────
println("\n=== Saving complete metadata database ===")
save_annotations(db, json_annot_path)
save_annotations_hdf5(db, h5_annot_path)
save_annotations(db, case_json_path)
save_annotations_hdf5(db, case_h5_path)

println("✓ Saved $(length(db)) entries to:")
println("  - $h5_annot_path (HDF5)")
println("  - $json_annot_path (JSON)")
println("  - $case_h5_path (HDF5)")
println("  - $case_json_path (JSON)")
println("Total lesions processed: $total_lesions_processed")
println("Total artifacts flagged by rules: $total_artifacts_flagged")
println("Done!")
