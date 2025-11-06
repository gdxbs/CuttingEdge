#!/usr/bin/env python3
"""
Comprehensive test of both pattern and cloth shape extraction
"""

import cv2
import numpy as np

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_recognition_module import PatternRecognitionModule


def test_comprehensive_shapes():
    """Test both pattern and cloth shape extraction"""

    pattern_recognition = PatternRecognitionModule()
    cloth_recognition = ClothRecognitionModule()

    print("Comprehensive Shape Extraction Test")
    print("=" * 80)

    # Test patterns
    print("\n--- PATTERN SHAPE EXTRACTION ---")
    test_patterns = [
        (
            "images/shape/dress_sleeveless/top_front/panel_44x37_dress_sleeveless_Q374Q0YUAY.png",
            44.0,
            37.0,
        ),
        (
            "images/shape/dress_sleeveless/top_front/panel_49x37_dress_sleeveless_WBN7DPNJ13.png",
            49.0,
            37.0,
        ),
        (
            "images/shape/dress_sleeveless/top_front/panel_53x37_dress_sleeveless_5S1089FQ3Q.png",
            53.0,
            37.0,
        ),
        (
            "images/shape/dress_sleeveless/top_front/panel_55x37_dress_sleeveless_70PTJ39TZ3.png",
            55.0,
            37.0,
        ),
    ]

    pattern_results = []
    for pattern_path, expected_width, expected_height in test_patterns:
        try:
            pattern = pattern_recognition.process_image(pattern_path)

            # Check scaling
            scale_x = pattern.width / expected_width
            scale_y = pattern.height / expected_height

            # Check contour area
            if pattern.contour is not None and len(pattern.contour) > 0:
                contour_int = pattern.contour.astype(np.int32)
                contour_area = cv2.contourArea(contour_int)
                expected_area = pattern.width * pattern.height
                coverage = contour_area / expected_area * 100
            else:
                contour_area = 0
                expected_area = pattern.width * pattern.height
                coverage = 0

            pattern_results.append(
                {
                    "file": pattern_path.split("/")[-1],
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "coverage": coverage,
                    "confidence": pattern.confidence,
                }
            )

            print(
                f"✓ {pattern_path.split('/')[-1]}: scale={scale_x:.3f}x, coverage={coverage:.1f}%, confidence={pattern.confidence:.2f}"
            )

        except Exception as e:
            print(f"✗ {pattern_path.split('/')[-1]}: Error - {e}")
            pattern_results.append(
                {"file": pattern_path.split("/")[-1], "error": str(e)}
            )

    # Test cloths
    print("\n--- CLOTH SHAPE EXTRACTION ---")
    test_cloths = [
        ("images/cloth/free/cloth_497_862x352.jpg", 862.0, 352.0),
        ("images/cloth/free/cloth_305_862x352.jpg", 862.0, 352.0),
    ]

    cloth_results = []
    for cloth_path, expected_width, expected_height in test_cloths:
        try:
            cloth = cloth_recognition.process_image(cloth_path)

            # Check scaling
            scale_x = cloth.width / expected_width
            scale_y = cloth.height / expected_height

            # Check contour area
            if cloth.contour is not None and len(cloth.contour) > 0:
                contour_int = cloth.contour.astype(np.int32)
                contour_area = cv2.contourArea(contour_int)
                expected_area = cloth.width * cloth.height
                coverage = contour_area / expected_area * 100
            else:
                contour_area = 0
                expected_area = cloth.width * cloth.height
                coverage = 0

            cloth_results.append(
                {
                    "file": cloth_path.split("/")[-1],
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "coverage": coverage,
                    "defects": len(cloth.defects or []),
                }
            )

            print(
                f"✓ {cloth_path.split('/')[-1]}: scale={scale_x:.3f}x, coverage={coverage:.1f}%, defects={len(cloth.defects or [])}"
            )

        except Exception as e:
            print(f"✗ {cloth_path.split('/')[-1]}: Error - {e}")
            cloth_results.append({"file": cloth_path.split("/")[-1], "error": str(e)})

    # Summary
    print("\n--- SUMMARY ---")

    # Pattern summary
    valid_patterns = [r for r in pattern_results if "error" not in r]
    if valid_patterns:
        avg_scale_pattern = sum(r["scale_x"] for r in valid_patterns) / len(
            valid_patterns
        )
        avg_coverage_pattern = sum(r["coverage"] for r in valid_patterns) / len(
            valid_patterns
        )
        avg_confidence = sum(r["confidence"] for r in valid_patterns) / len(
            valid_patterns
        )

        print(f"Patterns: {len(valid_patterns)}/{len(test_patterns)} successful")
        print(f"  Average scaling: {avg_scale_pattern:.3f}x (target: 0.22x)")
        print(f"  Average coverage: {avg_coverage_pattern:.1f}% (target: 70-110%)")
        print(f"  Average confidence: {avg_confidence:.2f}")

    # Cloth summary
    valid_cloths = [r for r in cloth_results if "error" not in r]
    if valid_cloths:
        avg_scale_cloth = sum(r["scale_x"] for r in valid_cloths) / len(valid_cloths)
        avg_coverage_cloth = sum(r["coverage"] for r in valid_cloths) / len(
            valid_cloths
        )
        total_defects = sum(r["defects"] for r in valid_cloths)

        print(f"Cloths: {len(valid_cloths)}/{len(test_cloths)} successful")
        print(f"  Average scaling: {avg_scale_cloth:.3f}x (target: 0.22x)")
        print(f"  Average coverage: {avg_coverage_cloth:.1f}% (target: 70-110%)")
        print(f"  Total defects detected: {total_defects}")

    # Overall assessment
    print("\n--- OVERALL ASSESSMENT ---")

    pattern_success = len(valid_patterns) / len(test_patterns) * 100
    cloth_success = len(valid_cloths) / len(test_cloths) * 100

    print(f"Pattern extraction success rate: {pattern_success:.1f}%")
    print(f"Cloth extraction success rate: {cloth_success:.1f}%")

    # Calculate averages only if we have valid results
    avg_scale_pattern = 0
    avg_coverage_pattern = 0
    if valid_patterns:
        avg_scale_pattern = sum(r["scale_x"] for r in valid_patterns) / len(
            valid_patterns
        )
        avg_coverage_pattern = sum(r["coverage"] for r in valid_patterns) / len(
            valid_patterns
        )

    avg_scale_cloth = 0
    avg_coverage_cloth = 0
    if valid_cloths:
        avg_scale_cloth = sum(r["scale_x"] for r in valid_cloths) / len(valid_cloths)
        avg_coverage_cloth = sum(r["coverage"] for r in valid_cloths) / len(
            valid_cloths
        )

    if (
        pattern_success >= 75
        and cloth_success >= 75
        and abs(avg_scale_pattern - 1.00) < 0.05  # Patterns should not be scaled
        and abs(avg_scale_cloth - 0.22) < 0.05  # Cloths should be scaled to 0.22x
        and 70 <= avg_coverage_pattern <= 110
        and 70 <= avg_coverage_cloth <= 110
    ):
        print("🎉 ALL TESTS PASSED! Shape extraction is working correctly.")
    else:
        print("⚠️  Some tests failed. Check individual results above.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_comprehensive_shapes()
