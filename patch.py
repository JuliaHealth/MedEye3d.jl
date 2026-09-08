with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "After mask modification (painting or AI segmentation):" in line:
        start_idx = i - 2
        break

for i, line in enumerate(lines[start_idx:]):
    if "function invalidate_and_recompute_lesion_metrics_async!" in line:
        end_idx = start_idx + i - 1
        break

# The new correct docstring and definition
new_block = [
    "const _async_suv_debounce = Dict{Tuple{Int, Int}, Float64}()\n",
    "\n",
    "\"\"\"\n",
    "    invalidate_and_recompute_lesion_metrics_async!(lesion_id, tp_idx, mask_vol)\n",
    "\n",
    "Called when a lesion is painted or modified.\n",
    "1. Synchronously invalidates caches.\n",
    "2. Schedules a debounced background task to recompute metrics.\n",
    "\"\"\"\n"
]

lines = lines[:start_idx] + new_block + lines[end_idx+1:]

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.writelines(lines)
