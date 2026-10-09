using JSON
d = JSON.parsefile("data/max_anatomy_to_ontology.json")

# Fix 1: Appendicular bones mapping
app_bones = ["femur", "humerus", "tibia", "scapula", "radius", "ulna", "fibula", "patella", "carpal", "tarsal", "metacarpal", "metatarsal", "phalange", "clavicula", "clavicle", "hip", "ilium", "ischium", "pubis"]

for (k, v) in d
    k_lc = lowercase(k)
    
    # Cartilages and joints shouldn't be bones
    if occursin("cartilage", k_lc) || occursin("joint", k_lc)
        v["lesion_type"] = "Organ Meta"
        v["anatomic_location"] = "General Soft Tissue (Muscles, Subcutaneous)"
        continue
    end
    
    # Fix Axial -> Appendicular for clavicles, hips, ilium, ischium, pubis
    if v["lesion_type"] == "Bone Meta"
        if any(kw -> occursin(kw, k_lc), app_bones)
            v["anatomic_location"] = "Appendicular Skeleton (Limbs, Pelvis, Scapulae, Clavicles, Hands, Feet)"
        else
            v["anatomic_location"] = "Axial Skeleton (Spine, Ribs, Skull, Sternum)"
        end
    end
    
    # Fix Lymph Nodes misclassifications
    if v["lesion_type"] == "Lymph Node"
        if k_lc in ["ribs_combined", "fused_spine", "thoracic_sternum_bridge", "thoracic_sternum_exclusion_zone"]
            v["lesion_type"] = "Bone Meta"
            v["anatomic_location"] = "Axial Skeleton (Spine, Ribs, Skull, Sternum)"
        elseif k_lc in ["portal_vein", "splenic_vein", "aortic_arch_exported", "abdominal_superior_mesenteric_artery"]
            v["lesion_type"] = "Organ Meta"
            v["anatomic_location"] = "Blood Vessel"
        elseif k_lc in ["tissue_fat", "thorax_wall", "deep_fascial_floor_left", "deep_fascial_floor_right"]
            v["lesion_type"] = "Organ Meta"
            v["anatomic_location"] = "General Soft Tissue (Muscles, Subcutaneous)"
        elseif occursin("helper", k_lc) || occursin("exclusion", k_lc) || k_lc == "lev_vert" || k_lc == "spinal_cord_dilated_1cm"
            v["lesion_type"] = "Technical Artifact"
            v["anatomic_location"] = "General Soft Tissue (Muscles, Subcutaneous)"
        end
    end
end

open("data/max_anatomy_to_ontology.json", "w") do f
    JSON.print(f, d, 2)
end
