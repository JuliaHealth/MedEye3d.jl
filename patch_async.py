with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if 'if data.algorithm == "NNInteractive"' in line:
        start_idx = i
    if 'function reactToSyncMissing' in line:
        end_idx = i
        break

block = lines[start_idx:end_idx]
new_block = ["    @async begin\n"]
for line in block:
    if line.strip() == "end" and line == block[-1]:
        new_block.append("    end\n")
        new_block.append("end\n")
    elif line == "end\n" and block[-1] == "end\n" and line is block[-1]:
        pass
    else:
        new_block.append("    " + line)
        
# wait, actually let's just indent everything and wrap it.
lines = lines[:start_idx] + ["    @async begin\n"] + ["    " + l for l in block[:-1]] + ["    end\n", "end\n"] + lines[end_idx:]
with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.writelines(lines)
