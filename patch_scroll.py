with open("src/display/reactingToMouseKeyboard/ReactToScroll.jl", "r") as f:
    code = f.read()

target = """        if lastSlice < 1
            continue
        end"""

new_code = """        if lastSlice < 1 || panelState.calcDimsStruct.mainQuadVertSize <= 0 || all(iszero, panelState.calcDimsStruct.mainImageQuadVert)
            continue
        end"""

code = code.replace(target, new_code)

with open("src/display/reactingToMouseKeyboard/ReactToScroll.jl", "w") as f:
    f.write(code)
