import os
import sys
import json
import zipfile
import slicer
import vtk

def process_mrb(mrb_path, output_dir):
    print(f"Processing MRB: {mrb_path}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Extract JSONs directly from MRB (which is a ZIP)
    print("Extracting JSONs...")
    with zipfile.ZipFile(mrb_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.json') or file_info.filename.endswith('.txt') or file_info.filename.endswith('.csv'):
                # Extract to output_dir but without the internal folder structure
                basename = os.path.basename(file_info.filename)
                if basename:
                    source = zip_ref.open(file_info)
                    target_path = os.path.join(output_dir, basename)
                    with open(target_path, "wb") as target:
                        target.write(source.read())
                    print(f"  Copied {basename}")

    # 2. Load scene in Slicer
    print("Loading scene...")
    slicer.mrmlScene.Clear(0)
    slicer.util.loadScene(mrb_path)

    # 3. Harden transforms on all nodes
    print("Hardening transforms...")
    transformLogic = slicer.modules.transforms.logic()
    for node in slicer.util.getNodesByClass("vtkMRMLTransformableNode"):
        if node.GetParentTransformNode():
            print(f"  Hardening transform on {node.GetName()}")
            transformLogic.hardenTransform(node)

    # 4. Export Volumes
    print("Exporting volumes...")
    for vol in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        name = vol.GetName()
        # Avoid saving the default generic anatomies if any, but usually it's fine
        out_path = os.path.join(output_dir, f"{name}.nrrd")
        slicer.util.saveNode(vol, out_path)
        print(f"  Saved {name}.nrrd")
    
    for vol in slicer.util.getNodesByClass("vtkMRMLVectorVolumeNode"):
        name = vol.GetName()
        out_path = os.path.join(output_dir, f"{name}.nrrd")
        slicer.util.saveNode(vol, out_path)
        print(f"  Saved {name}.nrrd")

    # 5. Export Segmentations and their names
    print("Exporting segmentations...")
    seg_names_dict = {}
    for segNode in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        name = segNode.GetName()
        out_path = os.path.join(output_dir, f"{name}.seg.nrrd")
        slicer.util.saveNode(segNode, out_path)
        print(f"  Saved {name}.seg.nrrd")

        # Extract segment names and IDs
        segmentation = segNode.GetSegmentation()
        segment_info = {}
        for i in range(segmentation.GetNumberOfSegments()):
            segmentID = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(segmentID)
            # Use segment name
            segment_info[segmentID] = segment.GetName()
            # If there is specific terminology, we could also extract it, but name is requested
        
        seg_names_dict[name] = segment_info

    # Save segmentation names mapping
    seg_names_path = os.path.join(output_dir, "segmentation_names.json")
    with open(seg_names_path, "w") as f:
        json.dump(seg_names_dict, f, indent=4)
    print(f"  Saved segmentation names to {seg_names_path}")
    
    print("MRB Processing Complete.")
    slicer.app.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: Slicer --no-main-window --python-script extract_mrb.py <input_mrb> <output_dir>")
        slicer.app.exit(1)
    
    mrb_path = sys.argv[1]
    output_dir = sys.argv[2]
    process_mrb(mrb_path, output_dir)
