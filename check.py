import json

with open("data/max_anatomy_to_ontology.json") as f:
    data = json.load(f)

print("Checking for misclassifications...")

for key, info in data.items():
    loc = info.get("anatomic_location", "")
    ltype = info.get("lesion_type", "")
    
    # Check if cartilages or joints are marked as Bone Meta
    if ltype == "Bone Meta" and any(x in key.lower() for x in ["cartilage", "joint", "disc", "ligament", "tendon"]):
        print(f"Non-bone (cartilage/joint) classified as Bone Meta: {key}")
        
    # Check if standard Appendicular bones are in the Axial label
    if "Axial" in loc:
        if any(x in key.lower() for x in ["clavicula", "hip", "ilium", "ischium", "pubis"]):
            print(f"Appendicular bone assigned to Axial label: {key}")
            
    # Check if any muscle/organ is marked as Bone Meta or Skeleton
    if ltype == "Bone Meta" or "Skeleton" in loc:
        # exclude valid bones and cartilages/joints already caught
        valid_bone_keywords = ["vertebra", "rib", "sternum", "skull", "sacrum", "femur", "humerus", 
                               "tibia", "mandible", "hyoid", "scapula", "styloid", "zygomatic", 
                               "clavicula", "hip", "ilium", "ischium", "pubis", "cartilage", "joint",
                               "sacroiliac"]
        if not any(x in key.lower() for x in valid_bone_keywords):
            print(f"Potential non-bone marked as bone: {key} -> {loc}, {ltype}")

