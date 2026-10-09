import json

with open("data/max_anatomy_to_ontology.json") as f:
    data = json.load(f)

misclassifications = []

for mask_name, info in data.items():
    anatomic_location = info.get("anatomic_location", "")
    lesion_type = info.get("lesion_type", "")
    detailed = info.get("detailed", "")
    
    # Check if a non-bone is classified as a bone
    is_bone = ("Skeleton" in anatomic_location) or (lesion_type == "Bone Meta")
    
    # We should print all bones and their location to check manually or with logic.
    if is_bone:
        print(f"{mask_name} -> {anatomic_location}")

