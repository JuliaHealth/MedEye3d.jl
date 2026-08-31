using MedEye3d
using MedEye3d.SegmentationDisplay
using MedEye3d.MakieEvents
using MedEye3d.DataStructs
using MedEye3d.ForDisplayStructs
using MedEye3d.VulkanBackend
using MedEye3d.VulkanBackend.VulkanStaging
using Dates
using FileIO
using Statistics

println("=== BATCHED VULKAN DATA LOADING & TEXTURE UPLOAD BENCHMARK ===")

# Include the app script
include("/workspaces/MedEye3d.jl/scripts/app/run_interactive_mrb.jl")
sleep(3.0)

ch = mainMedEye3dInstance.channel

# Helper to sync and capture
function capture_synced(path::String, desc::String)
    done_ch = Channel{Bool}(1)
    put!(ch, ScreenshotEvent(path, done_ch))
    success = take!(done_ch)
    println("Captured $desc: success=$success -> $path")
    @assert success "Screenshot capture failed for $desc"
    return success
end

# 1. Capture baseline frame
capture_synced("/workspaces/MedEye3d.jl/data/scr/opt_01_baseline.png", "Baseline Frame")

# 2. Benchmark Rapid Multi-Panel Scrolling (Batched uploads)
println("--- Benchmarking 30 Rapid Multi-Panel Scroll Slices ---")
scroll_times = Float64[]
for i in 1:30
    t0 = time_ns()
    put!(ch, Int64(1)) # scroll delta +1
    sleep(0.016) # ~60 FPS rate
    t1 = time_ns()
    push!(scroll_times, (t1 - t0) / 1e6)
end
sleep(1.0)
println("Scroll dispatch times: mean = $(round(mean(scroll_times), digits=2)) ms")
capture_synced("/workspaces/MedEye3d.jl/data/scr/opt_02_after_30_scrolls.png", "After 30 Scrolls")

# 3. Benchmark Rapid TimePoint Navigation with Multi-Panel Batched Uploads
println("--- Benchmarking Time Point Changes with Batched Uploads ---")
tp_times = Float64[]
for tp in 1:4
    t0 = time_ns()
    put!(ch, ChangeTimePointEvent(tp))
    sleep(0.5)
    t1 = time_ns()
    push!(tp_times, (t1 - t0) / 1e6)
end
sleep(1.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/opt_03_after_tp_change.png", "After TP Navigation")

# Return back to TP 0
put!(ch, ChangeTimePointEvent(0))
sleep(1.0)

# 4. Compare mode toggle with Batched Slices
println("--- Testing Compare Mode with Batched Uploads ---")
put!(ch, CompareTimePointsEvent(true))
sleep(1.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/opt_04_compare_mode_batched.png", "Compare Mode Batched")

put!(ch, CompareTimePointsEvent(false))
sleep(1.0)
capture_synced("/workspaces/MedEye3d.jl/data/scr/opt_05_quadview_restored.png", "Quadview Restored")

println("=== ALL BATCHED OPTIMIZATION TESTS PASSED ===")
