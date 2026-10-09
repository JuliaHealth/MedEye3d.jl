import json

with open("data/max_anatomy_to_ontology.json") as f:
    data = json.load(f)

for key, info in data.items():
    loc = info.get("anatomic_location", "")
    ltype = info.get("lesion_type", "")
    
    if key in ["fused_spine", "spinal_cord_dilated_1cm", "thorax_wall", "tissue_fat", "portal_vein"]:
        print(f"{key}: {loc} / {ltype}")

