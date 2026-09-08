import re

with open("src/display/LesionMetadataWindow.jl", "r") as f:
    code = f.read()

if "is_crosshair_visible()" not in code:
    code = code.replace("const _anatomy_visible_global = Ref(false)\n", "const _anatomy_visible_global = Ref(false)\nconst _crosshair_visible_global = Ref(false)\nis_crosshair_visible() = _crosshair_visible_global[]\n")
    
    # Add toggle button next to other toggles
    toggle_target = """    btn_vis_marrow = Button(controls_layout[6, 2], label="Marrow: OFF", buttoncolor=BG_PNL, halign=:left, width=140)"""
    new_toggle = """    btn_vis_marrow = Button(controls_layout[6, 2], label="Marrow: OFF", buttoncolor=BG_PNL, halign=:left, width=140)
    btn_vis_crosshair = Button(controls_layout[7, 1], label="Crosshair: OFF", buttoncolor=BG_PNL, halign=:left, width=140)
    vis_crosshair_active = Ref(false)
"""
    code = code.replace(toggle_target, new_toggle)
    
    # Add click handler
    click_target = """    on(btn_vis_anatomy.clicks) do _"""
    new_click = """    on(btn_vis_crosshair.clicks) do _
        vis_crosshair_active[] = !vis_crosshair_active[]
        _crosshair_visible_global[] = vis_crosshair_active[]
        btn_vis_crosshair.label[] = vis_crosshair_active[] ? "Crosshair: ON" : "Crosshair: OFF"
        btn_vis_crosshair.buttoncolor[] = vis_crosshair_active[] ? RGBf(0.0, 0.8, 0.0) : BG_PNL
        put!(channel, ShowMaskLayerEvent(5, vis_crosshair_active[])) # Send a dummy layer to trigger re-render
    end

    on(btn_vis_anatomy.clicks) do _"""
    code = code.replace(click_target, new_click)

with open("src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(code)
