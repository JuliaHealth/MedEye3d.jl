using Pkg
Pkg.activate(".")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.MakieEvents
using MedEye3d.DataStructs

# Verify code compiles correctly and ShowSingleLesionEvent works
println("Testing ShowSingleLesionEvent compilation...")
event1 = ShowSingleLesionEvent(1)
event0 = ShowSingleLesionEvent(0)
println("  ShowSingleLesionEvent(1) = $event1")
println("  ShowSingleLesionEvent(0) = $event0")

# Verify the reactToShowSingleLesion function signature is correct by checking it loads
println("\nLoading MakieEventHandlers module...")
# MakieEventHandlers is loaded via SegmentationDisplay which is loaded via MedEye3d
# We can check that reactToShowSingleLesion is defined:
hasmethod_check = hasmethod(MedEye3d.SegmentationDisplay.on_next!, Tuple{Vector{StateDataFields}, ShowSingleLesionEvent})
println("  on_next!(stateObjects, ShowSingleLesionEvent) defined: $hasmethod_check")

println("\nVerifying reactToShowSingleLesion does NOT contain reactToScroll...")
src_file = joinpath(@__DIR__, "..", "src", "display", "GLFW", "MakieEventHandlers.jl")
content = read(src_file, String)

# Find the function body
func_start = findfirst("function reactToShowSingleLesion", content)
func_end = findfirst(r"^end"m, content[func_start.start:end])
func_body = content[func_start.start:func_start.start + func_end.stop - 1]

has_react_to_scroll = occursin("reactToScroll", func_body)
println("  Contains reactToScroll: $has_react_to_scroll")

if has_react_to_scroll
    println("  FAIL: reactToShowSingleLesion still calls reactToScroll!")
    exit(1)
else
    println("  PASS: reactToScroll removed from reactToShowSingleLesion")
end

has_uniform_update = occursin("coontrolMinMaxUniformVals", func_body)
println("  Contains uniform update: $has_uniform_update")

if !has_uniform_update
    println("  FAIL: reactToShowSingleLesion doesn't update uniforms!")
    exit(1)
else
    println("  PASS: Uniform updates present")
end

println("\n=== ALL COMPILATION AND STRUCTURE TESTS PASSED ===")
