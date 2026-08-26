#!/usr/bin/env python3
"""
Post-process anatomy segmentation outputs:
1. Dilate mandible and subtract from skull (clean jaw-cranium separation)
2. Split bilateral muscles without left/right into _left/_right via connected components
"""

import os
import sys
import logging
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
logger = logging.getLogger("PostprocessAnatomy")

# Muscles that are bilateral and need splitting into _left / _right
BILATERAL_MUSCLES = [
    "deltoid", "subscapularis", "supraspinatus", "infraspinatus",
    "teres_major", "coracobrachial", "triceps_brachii",
    "serratus_anterior", "trapezius", "pectoralis_minor"
]

# Midline structures that should NOT be split
MIDLINE_STRUCTURES = [
    "tongue", "superior_pharyngeal_constrictor",
    "middle_pharyngeal_constrictor", "inferior_pharyngeal_constrictor"
]


def dilate_mandible_subtract_skull(anatomy_dir, dilation_radius=3):
    """Dilate mandible mask and subtract from skull to get clean separation."""
    mandible_path = os.path.join(anatomy_dir, "mandible.nii.gz")
    skull_path = os.path.join(anatomy_dir, "skull.nii.gz")

    if not os.path.exists(mandible_path) or not os.path.exists(skull_path):
        logger.warning("mandible.nii.gz or skull.nii.gz not found, skipping skull cleanup")
        return

    logger.info(f"Dilating mandible by radius={dilation_radius} and subtracting from skull...")
    mandible_img = sitk.ReadImage(mandible_path)
    skull_img = sitk.ReadImage(skull_path)

    # Binary threshold mandible
    mandible_binary = sitk.BinaryThreshold(mandible_img, lowerThreshold=1)

    # Morphological dilation with ball structuring element
    dilated_mandible = sitk.BinaryDilate(mandible_binary, [dilation_radius] * 3)

    # Subtract dilated mandible from skull: skull = skull AND NOT dilated_mandible
    dilated_arr = sitk.GetArrayFromImage(dilated_mandible)
    skull_arr = sitk.GetArrayFromImage(skull_img)

    skull_cleaned = skull_arr.copy()
    skull_cleaned[dilated_arr > 0] = 0

    # Write cleaned skull back
    out_img = sitk.GetImageFromArray(skull_cleaned)
    out_img.CopyInformation(skull_img)
    sitk.WriteImage(out_img, skull_path)

    original_voxels = int(np.sum(skull_arr > 0))
    cleaned_voxels = int(np.sum(skull_cleaned > 0))
    removed_voxels = original_voxels - cleaned_voxels
    logger.info(f"Skull cleaned: {original_voxels} -> {cleaned_voxels} voxels (removed {removed_voxels} in mandible zone)")


def split_bilateral_muscle(anatomy_dir, muscle_name):
    """Split a single bilateral muscle mask into _left and _right using connected components."""
    filepath = os.path.join(anatomy_dir, f"{muscle_name}.nii.gz")
    if not os.path.exists(filepath):
        return False

    left_path = os.path.join(anatomy_dir, f"{muscle_name}_left.nii.gz")
    right_path = os.path.join(anatomy_dir, f"{muscle_name}_right.nii.gz")

    # Skip if already split
    if os.path.exists(left_path) and os.path.exists(right_path):
        logger.debug(f"{muscle_name} already split, skipping")
        return True

    img = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(img)
    binary = (arr > 0).astype(np.uint8)

    if np.sum(binary) == 0:
        logger.warning(f"{muscle_name}: empty mask, skipping")
        return False

    # Connected component labeling
    labeled, num_features = ndimage.label(binary)
    logger.info(f"{muscle_name}: found {num_features} connected components")

    if num_features < 2:
        # Single component - split by sagittal midplane (axis 2 in ZYX = X in physical)
        # Use the image midpoint along X axis
        mid_x = arr.shape[2] // 2
        left_mask = np.zeros_like(arr)
        right_mask = np.zeros_like(arr)

        # In RAS convention, typically right side of patient = lower X indices
        # But we need to check the image direction to be sure
        direction = img.GetDirection()
        # direction[0] is the X component of the first axis
        # If direction[0] > 0, increasing index = increasing X (right in RAS = patient left)
        # We use physical coordinates to be safe

        # Compute centroid of all nonzero voxels
        coords = np.argwhere(binary > 0)  # Z, Y, X indices
        if len(coords) == 0:
            return False

        # Split at midplane
        left_mask[binary > 0] = 0
        right_mask[binary > 0] = 0

        for z, y, x in coords:
            if x < mid_x:
                right_mask[z, y, x] = 1  # Patient right
            else:
                left_mask[z, y, x] = 1   # Patient left

        logger.info(f"{muscle_name}: single component, split at midplane x={mid_x}")
    else:
        # Multiple components - assign by centroid X position
        component_centroids = ndimage.center_of_mass(binary, labeled, range(1, num_features + 1))

        # Find midpoint X across all centroids
        centroid_xs = [c[2] for c in component_centroids]  # X is index 2 in ZYX
        mid_x = np.mean(centroid_xs)

        left_mask = np.zeros_like(arr)
        right_mask = np.zeros_like(arr)

        for comp_id, centroid in enumerate(component_centroids, 1):
            comp_mask = (labeled == comp_id)
            if centroid[2] > mid_x:
                left_mask[comp_mask] = 1   # Higher X = patient left
            else:
                right_mask[comp_mask] = 1  # Lower X = patient right

        logger.info(f"{muscle_name}: {num_features} components, centroids at X={[f'{c[2]:.0f}' for c in component_centroids]}, midline={mid_x:.0f}")

    # Save left and right
    for mask_arr, out_path, side in [(left_mask, left_path, "left"), (right_mask, right_path, "right")]:
        if np.sum(mask_arr) > 0:
            out_img = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
            out_img.CopyInformation(img)
            sitk.WriteImage(out_img, out_path)
            logger.info(f"  Saved {muscle_name}_{side}.nii.gz ({int(np.sum(mask_arr))} voxels)")
        else:
            logger.warning(f"  {muscle_name}_{side} is empty, not saving")

    # Remove original unsided file
    os.remove(filepath)
    logger.info(f"  Removed original {muscle_name}.nii.gz")
    return True


def main(anatomy_dir):
    if not os.path.isdir(anatomy_dir):
        logger.error(f"Directory not found: {anatomy_dir}")
        sys.exit(1)

    logger.info(f"Post-processing anatomy in: {anatomy_dir}")

    # Step 1: Dilate mandible and subtract from skull
    dilate_mandible_subtract_skull(anatomy_dir, dilation_radius=3)

    # Step 2: Split bilateral muscles
    for muscle in BILATERAL_MUSCLES:
        try:
            split_bilateral_muscle(anatomy_dir, muscle)
        except Exception as e:
            logger.warning(f"Failed to split {muscle}: {e}")

    logger.info("Post-processing complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 postprocess_anatomy.py <anatomy_dir>")
        sys.exit(1)
    main(sys.argv[1])
