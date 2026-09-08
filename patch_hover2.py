with open("src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl", "r") as f:
    code = f.read()

target = "MEH.current_viewer_position[] = (ix, iy, currentSlice)"
new_code = """MEH.current_viewer_position[] = (ix, iy, currentSlice)
                        MEH.current_hovered_panel[] = mainState.switchIndex"""

if "MEH.current_hovered_panel[] =" not in code:
    code = code.replace(target, new_code)

with open("src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl", "w") as f:
    f.write(code)
