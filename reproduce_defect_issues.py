import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.config import CLOTH

def test_image(image_path):
    print(f"Testing image: {image_path}")
    if not os.path.exists(image_path):
        print("Image not found!")
        return

    crm = ClothRecognitionModule()
    
    # Check if using UNet or fallback
    print(f"Using UNet: {crm.use_unet}")
    print(f"Model Loaded: {crm.model_loaded}")

    result = crm.process_image(image_path)
    
    print(f"Cloth Type: {result.cloth_type}")
    print(f"Total Area: {result.total_area}")
    print(f"Defects Found: {len(result.defects) if result.defects else 0}")
    
    if result.defects_by_type:
        for dtype, defects in result.defects_by_type.items():
            print(f"  - {dtype}: {len(defects)} defects")
            # Only print details if count is low
            if len(defects) < 5:
                for i, d in enumerate(defects):
                    area = cv2.contourArea(d)
                    print(f"    - {dtype} {i}: Area={area:.2f}")
    else:
        print("  No classified defects.")

if __name__ == "__main__":
    stain_img = r"d:\cut\CuttingEdge\images\cloth\Stain\stain_sample_118_699x524.jpg"
    hole_img = r"d:\cut\CuttingEdge\images\cloth\Hole\hole_11_180x135.jpg"
    
    test_image(stain_img)
    print("-" * 50)
    test_image(hole_img)
