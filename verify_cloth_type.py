
import os
import logging
from cutting_edge.cloth_recognition_module import ClothRecognitionModule

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_cotton_detection():
    image_path = r"d:\cut\CuttingEdge\images\cloth\freeform\cloth_4_507x426.jpg"
    print(f"Testing cloth detection on: {image_path}")
    
    if not os.path.exists(image_path):
        print("Image not found!")
        return

    module = ClothRecognitionModule()
    
    # Process the image
    cloth = module.process_image(image_path)
    
    print(f"Detected Type: {cloth.cloth_type}")
    print(f"Is Irregular: {cloth.material_properties.get('is_irregular')}")
    print(f"Shape Complexity: {cloth.material_properties.get('shape_complexity')}")
    
    if cloth.cloth_type == "cotton":
        print("SUCCESS: Detected as cotton")
    else:
        print(f"FAILURE: Detected as {cloth.cloth_type}")

if __name__ == "__main__":
    test_cotton_detection()
