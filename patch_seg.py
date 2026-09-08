with open("src/display/GLFW/SegmentationDisplay.jl", "r") as f:
    code = f.read()

code = code.replace("const _push_consts = zeros(Float32, 8)", "const _push_consts = zeros(Float32, 12)")

with open("src/display/GLFW/SegmentationDisplay.jl", "w") as f:
    f.write(code)
