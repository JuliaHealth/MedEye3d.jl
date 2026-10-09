import json

with open("data/max_anatomy_to_ontology.json") as f:
    data = json.load(f)

for key, info in data.items():
    loc = info.get("anatomic_location", "")
    ltype = info.get("lesion_type", "")
    
    # Are there any typical bone words in the key but it's not a bone?
    bone_words = ["vertebra", "rib", "sternum", "skull", "sacrum", "femur", "humerus", 
                   "tibia", "mandible", "hyoid", "scapula", "styloid", "zygomatic", 
                   "clavicula", "hip", "ilium", "ischium", "pubis", "bone"]
    
    if any(x in key.lower() for x in bone_words):
        if ltype != "Bone Meta" and "Skeleton" not in loc:
            print(f"Potential bone not marked as bone: {key} -> loc: '{loc}', ltype: '{ltype}'")

