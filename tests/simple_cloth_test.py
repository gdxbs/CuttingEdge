#!/usr/bin/env python3
"""
Simple test of cloth shape extraction
"""

import cv2
import numpy as np

from cutting_edge.cloth_recognition_module import ClothRecognitionModule


def test_cloth_shapes():
    """Test cloth shape extraction"""

    cloth_recognition = ClothRecognitionModule()

    # Test different cloth types
    test_cloths = [
        ("images/cloth/free/cloth_497_862x352.jpg", 862.0, 352.0),
        ("images/cloth/free/cloth_305_862x352.jpg", 862.0, 352.0),
        ("images/cloth/wool/cloth_524_699x524.jpg", 699.0, 524.0),
    ]

    print("Testing cloth shape extraction")
    print("=" * 60)

    for cloth_path, expected_width, expected_height in test_cloths:
        try:
            print(f"\nTesting: {cloth_path.split('/')[-1]}")
            print(f"Expected original: {expected_width:.1f} x {expected_height:.1f} cm")

            # Process cloth
            cloth = cloth_recognition.process_image(cloth_path)

            print(f"Detected scaled: {cloth.width:.1f} x {cloth.height:.1f} cm")
            print(f"Cloth type: {cloth.cloth_type}")

            # Check scaling (should be 0.22x)
            scale_x = cloth.width / expected_width
            scale_y = cloth.height / expected_height
            print(f"Scaling factors: {scale_x:.3f}x, {scale_y:.3f}x")

            if abs(scale_x - 0.22) < 0.05 and abs(scale_y - 0.22) < 0.05:
                print("✓ Scaling factor is correct (~0.22x)")
            else:
                print("✗ Scaling factor is incorrect")

            # Check contour
            if cloth.contour is not None:
                # Convert float coordinates to integer for OpenCV operations
                contour_int = cloth.contour.astype(np.int32)
                contour_area = cv2.contourArea(contour_int)
                cloth_area = cloth.width * cloth.height
                coverage = contour_area / cloth_area * 100
                print(f"Contour area: {contour_area:.1f} cm²")
                print(f"Expected area: {cloth_area:.1f} cm²")
                print(f"Coverage: {coverage:.1f}%")

                if 70 <= coverage <= 110:
                    print("✓ Contour coverage is reasonable")
                else:
                    print("✗ Contour coverage seems off")
            else:
                print("✗ No contour found")

            # Check defects
            if cloth.defects:
                print(f"Defects: {len(cloth.defects)} found")
            else:
                print("No defects detected")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_cloth_shapes()
