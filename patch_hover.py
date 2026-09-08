with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    code = f.read()

if "const current_hovered_panel = Ref{Int}(0)" not in code:
    code = code.replace("const current_viewer_position = Ref((0, 0, 0))", "const current_viewer_position = Ref((0, 0, 0))\nconst current_hovered_panel = Ref{Int}(0)")
    code = code.replace("export cursor_info_text, cursor_study_text, set_ai_status!, current_viewer_position", "export cursor_info_text, cursor_study_text, set_ai_status!, current_viewer_position, current_hovered_panel")

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.write(code)

with open("src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl", "r") as f:
    code2 = f.read()

if "MEH.current_hovered_panel[] = hovered_panel" not in code2:
    code2 = code2.replace("MEH.current_viewer_position[] = (ix, iy, currentSlice)", "MEH.current_viewer_position[] = (ix, iy, currentSlice)\n                MEH.current_hovered_panel[] = hovered_panel")

with open("src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl", "w") as f:
    f.write(code2)
