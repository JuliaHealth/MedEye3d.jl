with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.jl", "r") as f:
    content = f.read()

# In entry_to_vdt
content = content.replace(
    'anat_f32 = e.anatomy !== nothing ? Float32.(e.anatomy) : zeros(Float32, size(e.ct))',
    'anat_f32 = MEH.global_ts_atlas[] !== nothing ? Float32.(MEH.global_ts_atlas[]) : zeros(Float32, size(e.ct))'
)

# And we can just skip loading anatomy_vol per TP entirely!
# But actually, leaving it in TpCacheEntry is fine, it will just not be used.
# Let's remove the whole try-catch for max_anat_src in load_single_tp_from_h5 to save RAM and time!
import re
pattern = re.compile(r'            # Load per-TP max_anatomy atlas.*?anatomy_labels_cache\[tp_i\] = .*?end\n.*?end\n.*?end', re.DOTALL)
# Wait, it's safer to just comment it out, or let it load but not use it.
# Actually, the user says "each time point has its own max anatomy mask !". 
# They want to NOT load it!
