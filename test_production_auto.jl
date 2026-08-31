using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

# Run the full production script by including it, but override readline
# We inject our test code BEFORE readline by modifying the script flow

# Include all lines except the readline() block at end
# Instead of `include("run_interactive_mrb.jl")`, we run it manually

# Source the production script
include("/workspaces/MedEye3d.jl/scripts/app/run_interactive_mrb.jl")
