"""
Generates segmentation masks for all cloth images using the color-based
segmentation method. These masks are required for training the U-Net model.

This script should be run once before the first training run of the
cloth recognition model.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to Python path to allow importing from cutting_edge
sys.path.append(str(Path(__file__).resolve().parent))

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.config import CLOTH, SYSTEM


def create_masks():
    """
    Iterates through all cloth images, generates a segmentation mask for each,
    and saves it to the corresponding location in the 'masks' directory.
    """
    base_dir = Path(SYSTEM["BASE_DIR"])
    image_dir = base_dir / "images" / "cloth"
    mask_dir = base_dir / "masks" / "cloth"

    if not image_dir.exists():
        print(f"Error: Image directory not found at {image_dir}")
        return

    print(f"Generating masks from '{image_dir}' and saving to '{mask_dir}'...")

    # We must use the color-based segmentation to create the initial masks,
    # so we'll temporarily override the config to disable the U-Net.
    original_use_unet = CLOTH["USE_UNET"]
    CLOTH["USE_UNET"] = False
    cloth_module = ClothRecognitionModule()
    CLOTH["USE_UNET"] = original_use_unet  # Restore config setting

    image_paths = list(image_dir.glob("**/*.jpg")) + list(
        image_dir.glob("**/*.png")
    ) + list(image_dir.glob("**/*.jpeg"))

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            shape_info = cloth_module.extract_cloth_shape(str(image_path))
            cloth_mask = shape_info["cloth_mask"]

            if cloth_mask is None or cloth_mask.size == 0:
                print(f"Warning: No mask generated for {image_path}, creating blank mask.")
                # Create a black mask as a fallback
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                cloth_mask = np.zeros(img.shape[:2], dtype=np.uint8)

            # Create the corresponding mask path, mirroring the image structure
            relative_path = image_path.relative_to(image_dir)
            mask_path = mask_dir / relative_path
            mask_path = mask_path.with_suffix(".png")

            # Create parent directory for the mask if it doesn't exist
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the generated mask as a PNG file
            cv2.imwrite(str(mask_path), cloth_mask)

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

    print("\nMask generation complete.")
    print(f"Please check the contents of the '{mask_dir}' directory.")


if __name__ == "__main__":
    create_masks()
