import re

with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    code = f.read()

find_str = """    # Snapshot the commit theta and display theta at the moment of release
    commit_theta = HEAT_GDT_COMMIT_THETA[]
    last_display_theta = HEAT_GDT_LAST_DISPLAY_THETA[]"""

replace_str = """    # Snapshot the commit theta and display theta at the moment of release
    # The user requested to keep ONLY the part that is visible at the moment of mouse release
    # so we use HEAT_GDT_LAST_DISPLAY_THETA[] instead of HEAT_GDT_COMMIT_THETA[].
    commit_theta = HEAT_GDT_LAST_DISPLAY_THETA[]
    last_display_theta = HEAT_GDT_LAST_DISPLAY_THETA[]"""

code = code.replace(find_str, replace_str)

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.write(code)

