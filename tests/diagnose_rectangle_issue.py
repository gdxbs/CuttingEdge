
import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.config import CLOTH

def create_irregular_cloth_image(path):
    """Create a synthetic irregular cloth image (L-shape)."""
    # Create white background
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    # Draw L-shaped blue cloth
    pts = np.array([[100, 100], [400, 100], [400, 250], [250, 250], [250, 400], [100, 400]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (200, 0, 0)) # Blue in BGR
    
    cv2.imwrite(path, img)
    print(f"Created synthetic irregular cloth at {path}")

def test_contour_extraction():
    module = ClothRecognitionModule()
    
    # Create test image
    test_img_path = "test_irregular_cloth.png"
    create_irregular_cloth_image(test_img_path)
    
    # Process
    try:
        cloth = module.process_image(test_img_path)
        
        print("\nAnalysis Results:")
        print(f"Cloth Type: {cloth.cloth_type}")
        print(f"Contour Points: {len(cloth.contour)}")
        
        # Check if it's a rectangle (4 points) or irregular (>4 points)
        if len(cloth.contour) <= 4:
            print("ISSUE DETECTED: Contour has 4 or fewer points. Likely simplified to a rectangle.")
        else:
            print("SUCCESS: Contour has > 4 points. Irregular shape preserved.")
            
        # Check area coverage
        img_area = 500 * 500
        # Expected area of L-shape: (300*300) - (150*150) = 90000 - 22500 = 67500 pixels (approx)
        # Using scale factors, check if meaningful area is returned
        print(f"Detected Area: {cloth.total_area:.2f} cm2")
        
        # Clean up
        if os.path.exists(test_img_path):
            os.remove(test_img_path)
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_contour_extraction()
