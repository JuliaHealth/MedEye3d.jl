with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    code = f.read()

old_func = """function _mri_clamp_mask_range!(stateObjects::Vector{StateDataFields})
    tp = current_tp_index[]
    panel_mod = uppercase(get(tp_modalities, tp, "PET"))
    if !(panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI"))
        return  # Not MRI, no clamping needed
    end
    anatomy_on = try
        LMW = _get_lmw()
        LMW !== nothing ? LMW.is_anatomy_visible() : false
    catch; false; end
    anatomy_on && return  # Anatomy ON → allow all labels

    # Clamp max label to 3 (hide gland = label 4+)
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation" || (textSpec.isMultiDiscreteMask && textSpec.name != "Anatomy" && textSpec.name != "Bone_Overlay")
                T_mm = eltype(textSpec.minAndMaxValue)
                cur_max = textSpec.minAndMaxValue[2]
                if cur_max > T_mm(3)
                    textSpec.minAndMaxValue = T_mm.([textSpec.minAndMaxValue[1], 3])
                end
            end
        end
    end
end"""

new_func = """function _mri_clamp_mask_range!(stateObjects::Vector{StateDataFields})
    anatomy_on = try
        LMW = _get_lmw()
        LMW !== nothing ? LMW.is_anatomy_visible() : false
    catch; false; end
    anatomy_on && return  # Anatomy ON → allow all labels

    # Check each panel individually based on its actual TP
    for (i, stateObject) in enumerate(stateObjects)
        tp = (i == 5 && compare_mode[]) ? compare_right_tp[] : current_tp_index[]
        panel_mod = uppercase(get(tp_modalities, tp, "PET"))
        if !(panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI"))
            continue
        end
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation" || (textSpec.isMultiDiscreteMask && textSpec.name != "Anatomy" && textSpec.name != "Bone_Overlay")
                T_mm = eltype(textSpec.minAndMaxValue)
                cur_max = textSpec.minAndMaxValue[2]
                if cur_max > T_mm(3)
                    textSpec.minAndMaxValue = T_mm.([textSpec.minAndMaxValue[1], 3])
                end
            end
        end
    end
end"""

code = code.replace(old_func, new_func)

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.write(code)
