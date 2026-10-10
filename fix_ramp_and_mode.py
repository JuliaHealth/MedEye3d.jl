import re

with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    code = f.read()

# 1. Slow down the initial growth display more
# Decrease HEAT_DISPLAY_RAMP_RATE from 3.0f0 to 1.0f0
code = re.sub(r"const HEAT_DISPLAY_RAMP_RATE  = 3\.0f0", "const HEAT_DISPLAY_RAMP_RATE  = 0.5f0", code)
code = re.sub(r"const HEAT_DISPLAY_RAMP_RATE  = 2\.5f0", "const HEAT_DISPLAY_RAMP_RATE  = 0.5f0", code) # Just in case

# 2. Get out of editing mode when toggling OFF HeatGDT
# In reactToOneShotHeatGDT, toggle OFF branch
off_branch_find = """        if is_active
            # Toggle OFF — refresh commit theta constants
            romcad._heatgdt_mode_active[] = false
            HEAT_GDT_COMMIT_THETA[] = heatgdt_theta[]           # Refresh to current default
            HEAT_GDT_LAST_DISPLAY_THETA[] = HEAT_DISPLAY_RAMP_START  # Reset ramp
            set_ai_status!("[Heat-GDT Wand] Disabled.")
        else"""

off_branch_replace = """        if is_active
            # Toggle OFF — refresh commit theta constants
            romcad._heatgdt_mode_active[] = false
            HEAT_GDT_COMMIT_THETA[] = heatgdt_theta[]           # Refresh to current default
            HEAT_GDT_LAST_DISPLAY_THETA[] = HEAT_DISPLAY_RAMP_START  # Reset ramp
            set_ai_status!("[Heat-GDT Wand] Disabled.")
            
            # Exit editing mode
            reactToViewMode(MakieEvents.ViewModeEvent(), stateObjects)
        else"""

code = code.replace(off_branch_find, off_branch_replace)

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.write(code)

