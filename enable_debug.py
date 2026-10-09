import re
with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    content = f.read()

# Replace `if DEBUG_VERBOSE[]; println("  [BONE] Cache HIT ... ); flush(stdout); end`
# with `println("  [BONE] Cache HIT ... ); flush(stdout);`
content = re.sub(r"if DEBUG_VERBOSE\[\];\s*(println\(\"  \[BONE\] Cache HIT.*?\);\s*flush\(stdout\);)\s*end", r"\1", content)

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.write(content)
