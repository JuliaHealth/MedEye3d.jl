# Real App simulation testing exact user workflow in scripts/app/run_interactive_mrb.jl
include("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.jl")

println("\n=== APP LAUNCHED SUCCESSFULLY ===")
println("Testing Semiauto AI through the running app...")

# Check that inference worker is running
MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
println("Status Observable: $(MEH.ai_status_text[])")

# Simulate user workflow:
# 1. Click "New Lesion"
# 2. Paint in Axial panel
# 3. Trigger HELPNet
# 4. Trigger NNInteractive

ch = mainMedEye3dInstance.channel

# Find first panel's manualModif
tp1_state = SegmentationDisplay.mainMedEye3dInstance.channel # channel
# Get state from SegmentationDisplay
# In the app, we can put events on the channel:

println("Step 1: Put PaintValEvent(32, true) on channel...")
put!(ch, PaintValEvent(32, true))
sleep(0.5)

# Find a lesion coordinate in baseline CT
# In baseline CT, lesion 1 is around (329, 255, 152)
cx, cy, cz = 329, 255, 152

println("Step 2: Simulating mouse drawing via MouseStruct on channel...")
# Or modifying manualModif in stateObjects directly:
# Let's paint by sending MouseStruct or directly into manualModif in tp_data_cache
tp_idx = MEH.current_tp_index[]
println("Current TP index: $tp_idx")
panel1_vols = MEH.tp_data_cache[tp_idx][1]
for entry in panel1_vols
    println("  Panel 1 entry: $(entry[1]) (size $(size(entry[2])))")
    if entry[1] == "manualModif"
        for dx in -2:2, dy in -2:2
            entry[2][cx+dx, cy+dy, cz] = 32.0f0
        end
        println("  -> Painted $(count(entry[2] .> 0)) voxels into manualModif in cache")
    end
end

# Also paint into the live stateObjects' manualModif if available
# We can find manualModif in first_voxelDataTupleVector:
for entry in first_voxelDataTupleVector[1]
    if entry[1] == "manualModif"
        for dx in -2:2, dy in -2:2
            entry[2][cx+dx, cy+dy, cz] = 32.0f0
        end
        println("  -> Painted $(count(entry[2] .> 0)) voxels into live manualModif")
    end
end

# Check if seg_vol has lesion 32 before
seg_vol = nothing
for entry in first_voxelDataTupleVector[1]
    if entry[1] == "Mask"
        global seg_vol = entry[2]
        break
    end
end
before_count = count(seg_vol .== 32.0f0)
println("seg_vol label=32 before AI: $before_count")

# Step 3: Trigger HELPNet via AddAutoPetEvent
println("\nStep 3: Triggering HELPNet via AddAutoPetEvent...")
put!(ch, AddAutoPetEvent("HELPNet (AI)", ch))

# Wait for completion
for i in 1:20
    sleep(1)
    c = count(seg_vol .== 32.0f0)
    if c > before_count
        println("  [SUCCESS] HELPNet completed at $(i)s: $(c - before_count) voxels generated!")
        println("  Status: $(MEH.ai_status_text[])")
        break
    end
    if i % 5 == 0
        println("  ... waiting ($(i)s, status: $(MEH.ai_status_text[]))")
    end
end

after_helpnet = count(seg_vol .== 32.0f0)
println("seg_vol label=32 after HELPNet: $after_helpnet (new: $(after_helpnet - before_count))")

# Step 4: Test NNInteractive
println("\nStep 4: Painting new scribble for NNInteractive (lesion 33)...")
put!(ch, PaintValEvent(33, true))
sleep(0.5)

for entry in first_voxelDataTupleVector[1]
    if entry[1] == "manualModif"
        fill!(entry[2], 0.0f0)
        for dx in -2:2, dy in -2:2
            entry[2][cx+dx, cy+dy, cz] = 33.0f0
        end
        println("  -> Painted $(count(entry[2] .> 0)) voxels into live manualModif for NNInteractive")
    end
end

before_nn = count(seg_vol .== 33.0f0)
println("seg_vol label=33 before NNInteractive: $before_nn")

println("Triggering NNInteractive via AddAutoPetEvent...")
put!(ch, AddAutoPetEvent("NNInteractive", ch))

for i in 1:20
    sleep(1)
    c = count(seg_vol .== 33.0f0)
    if c > before_nn
        println("  [SUCCESS] NNInteractive completed at $(i)s: $(c - before_nn) voxels generated!")
        println("  Status: $(MEH.ai_status_text[])")
        break
    end
    if i % 5 == 0
        println("  ... waiting ($(i)s, status: $(MEH.ai_status_text[]))")
    end
end

after_nn = count(seg_vol .== 33.0f0)
println("seg_vol label=33 after NNInteractive: $after_nn (new: $(after_nn - before_nn))")

println("\n=== SUMMARY ===")
println("HELPNet voxels: $(after_helpnet - before_count)")
println("NNInteractive voxels: $(after_nn - before_nn)")
println("Final status: $(MEH.ai_status_text[])")

