with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    code = f.read()

old_func = """function _force_mri_show_all!(tp::Int, stateObjects::Vector{StateDataFields})
    panel_mod = uppercase(get(tp_modalities, tp, "PET"))
    if !(panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI"))
        return
    end
    # Check anatomy toggle state from LesionMetadataWindow
    anatomy_on = try
        LMW = _get_lmw()
        LMW !== nothing ? LMW.is_anatomy_visible() : false
    catch; false; end
    max_label = anatomy_on ? 1000 : 3  # 1-3 = lesions, 4+ = gland/anatomy

    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation" || (textSpec.isMultiDiscreteMask && textSpec.name != "Anatomy" && textSpec.name != "Bone_Overlay")
                T_mm = eltype(textSpec.minAndMaxValue)
                textSpec.minAndMaxValue = T_mm.([1, max_label])
                textSpec.isVisible = true
                # Ensure mask is rendered with non-zero opacity
                if textSpec.maskContribution <= 0.0f0
                    textSpec.maskContribution = 0.5f0
                end
            end
        end
    end
end"""

new_func = """function _force_mri_show_all!(stateObjects::Vector{StateDataFields})
    anatomy_on = try
        LMW = _get_lmw()
        LMW !== nothing ? LMW.is_anatomy_visible() : false
    catch; false; end
    max_label = anatomy_on ? 1000 : 3  # 1-3 = lesions, 4+ = gland/anatomy

    for (i, stateObject) in enumerate(stateObjects)
        tp = (i == 5 && compare_mode[]) ? compare_right_tp[] : current_tp_index[]
        panel_mod = uppercase(get(tp_modalities, tp, "PET"))
        if !(panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI"))
            continue
        end
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation" || (textSpec.isMultiDiscreteMask && textSpec.name != "Anatomy" && textSpec.name != "Bone_Overlay")
                T_mm = eltype(textSpec.minAndMaxValue)
                textSpec.minAndMaxValue = T_mm.([1, max_label])
                textSpec.isVisible = true
                # Ensure mask is rendered with non-zero opacity
                if textSpec.maskContribution <= 0.0f0
                    textSpec.maskContribution = 0.5f0
                end
            end
        end
    end
end"""

code = code.replace(old_func, new_func)
code = code.replace("_force_mri_show_all!(tp, stateObjects)", "_force_mri_show_all!(stateObjects)")

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.write(code)
