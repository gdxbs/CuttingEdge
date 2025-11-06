#!/usr/bin/env python3
"""
Test cloth recognition and pattern fitting with irregular cloth shapes and defects.

This test demonstrates the system's ability to:
1. Detect irregular cloth shapes (remnants, L-shapes, scraps)
2. Identify defects (holes, stains, tears)
3. Fit patterns while avoiding defects and staying within irregular boundaries
"""

import os
import sys

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_fitting_module import PatternFittingModule
from cutting_edge.pattern_recognition_module import PatternRecognitionModule


def create_irregular_cloth_with_defects(
    width: int = 800, height: int = 600, defect_count: int = 3
) -> np.ndarray:
    """
    Create a synthetic irregular cloth image with defects for testing.

    Creates:
    - L-shaped cloth (irregular shape)
    - Multiple defects (holes and stains)
    - Realistic colors and textures
    """
    # Create white background (pure white for clear background detection)
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Create L-shaped cloth (irregular shape) with distinct color
    cloth_color = (180, 140, 120)  # Beige/tan cloth color

    # Vertical part of L (scaled for larger image)
    cv2.rectangle(img, (50, 50), (300, 500), cloth_color, -1)
    # Horizontal part of L
    cv2.rectangle(img, (300, 350), (700, 500), cloth_color, -1)

    # Add subtle texture to cloth (simulate fabric texture)
    # Keep noise small so defects stand out
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    cloth_region = (img[:, :, 0] < 250).astype(np.uint8)  # Non-white areas
    img = np.clip(
        img.astype(np.int16) + noise * cloth_region[:, :, np.newaxis], 0, 255
    ).astype(np.uint8)

    # Add VERY DARK defects (holes - black/near-black for clear detection)
    # Make them larger so they're detectable after scaling
    hole_positions = [
        (150, 200, 30),  # (x, y, radius) - larger defects
        (200, 420, 25),
        (500, 420, 35),
    ]

    for x, y, radius in hole_positions[:defect_count]:
        # Very dark hole (nearly black)
        cv2.circle(img, (x, y), radius, (10, 10, 10), -1)
        # Darker border
        cv2.circle(img, (x, y), radius + 2, (5, 5, 5), 3)

    # Add bright stains (much lighter than cloth)
    bright_stain_positions = [(400, 250, 40), (600, 420, 35)]
    for i, (x, y, radius) in enumerate(bright_stain_positions):
        if i < defect_count - 1:
            # Bright stain (much lighter than cloth)
            stain_color = (240, 220, 200)
            cv2.circle(img, (x, y), radius, stain_color, -1)
            # Add slight border
            cv2.circle(img, (x, y), radius, (235, 215, 195), 2)

    # Add dark stains (darker than cloth but not as dark as holes)
    dark_stain_positions = [(150, 450, 25), (550, 380, 30)]
    for i, (x, y, radius) in enumerate(dark_stain_positions):
        if i < defect_count - 1:
            # Dark stain (brownish/grayish)
            stain_color = (80, 60, 50)
            cv2.circle(img, (x, y), radius, stain_color, -1)
            cv2.circle(img, (x, y), radius, (70, 55, 45), 2)

    return img


def test_irregular_cloth_detection():
    """Test detection of irregular cloth shape with defects."""
    print("\n" + "=" * 80)
    print("TEST: Irregular Cloth Shape Detection with Defects")
    print("=" * 80)

    # Create test image
    test_img = create_irregular_cloth_with_defects(defect_count=3)

    # Save test image
    os.makedirs("output", exist_ok=True)
    test_path = "output/test_irregular_cloth_with_defects.png"
    cv2.imwrite(test_path, test_img)
    print(f"\n✓ Created test image: {test_path}")

    # Initialize cloth recognition with U-Net disabled for testing
    # (U-Net requires training data, color-based segmentation works out of the box)
    cloth_module = ClothRecognitionModule()
    cloth_module.use_unet = False  # Force color-based segmentation for testing

    # Process the image
    print("\n--- Processing Irregular Cloth ---")
    print("(Using color-based defect detection)")
    cloth = cloth_module.process_image(test_path)

    # Verify results
    print("\n--- Cloth Detection Results ---")
    print(f"Cloth Type: {cloth.cloth_type}")
    print(f"Dimensions (bounding box): {cloth.width:.1f} x {cloth.height:.1f} cm")
    print(f"Total Area: {cloth.total_area:.1f} cm²")
    print(f"Usable Area: {cloth.usable_area:.1f} cm²")
    print(f"Contour Points: {len(cloth.contour) if cloth.contour is not None else 0}")
    print(f"Defects Detected: {len(cloth.defects) if cloth.defects is not None else 0}")

    # Check for irregular shape
    if cloth.contour is not None and len(cloth.contour) > 4:
        bbox_area = cloth.width * cloth.height
        shape_ratio = cloth.total_area / bbox_area if bbox_area > 0 else 1.0
        print(f"Shape Complexity: {shape_ratio:.2f}")
        if shape_ratio < 0.85:
            print("✓ IRREGULAR shape detected (L-shape/remnant)")
        else:
            print("Regular rectangular shape")

    # Verify defects were detected
    num_defects = len(cloth.defects) if cloth.defects is not None else 0
    if num_defects >= 3:
        print("✓ Multiple defects detected (holes/stains)")
    else:
        print(f"⚠ Expected 3+ defects, found {num_defects}")

    # Visualize cloth analysis
    vis_path = "output/test_irregular_cloth_visualization.png"
    cloth_module.visualize(cloth, vis_path)
    print(f"\n✓ Visualization saved: {vis_path}")

    return cloth


def test_pattern_fitting_on_irregular_cloth():
    """Test pattern fitting on irregular cloth with defects."""
    print("\n" + "=" * 80)
    print("TEST: Pattern Fitting on Irregular Cloth with Defects")
    print("=" * 80)

    # Get the irregular cloth
    cloth = test_irregular_cloth_detection()

    # Create some test patterns
    print("\n--- Creating Test Patterns ---")
    pattern_module = PatternRecognitionModule()

    # Find some real pattern files if available
    pattern_paths = []
    shape_dir = "images/shape"
    if os.path.exists(shape_dir):
        for root, dirs, files in os.walk(shape_dir):
            for file in files:
                if file.endswith(".png") and "panel_" in file:
                    pattern_paths.append(os.path.join(root, file))
                    if len(pattern_paths) >= 5:
                        break
            if len(pattern_paths) >= 5:
                break

    if len(pattern_paths) < 3:
        print("⚠ Not enough pattern files found, skipping fitting test")
        return

    # Process patterns
    patterns = []
    for i, path in enumerate(pattern_paths[:5]):
        pattern = pattern_module.process_image(path)
        patterns.append(pattern)
        print(
            f"  Pattern {i + 1}: {pattern.pattern_type}, {pattern.width:.1f}x{pattern.height:.1f} cm"
        )

    # Fit patterns onto irregular cloth
    print("\n--- Fitting Patterns onto Irregular Cloth ---")
    fitting_module = PatternFittingModule()

    result = fitting_module.fit_patterns(patterns, cloth)

    # Display results
    print("\n--- Fitting Results ---")
    print(f"Total Patterns: {result['patterns_total']}")
    print(f"Successfully Placed: {result['patterns_placed']}")
    print(f"Failed (couldn't fit): {len(result['failed_patterns'])}")
    print(f"Material Utilization: {result['utilization_percentage']:.1f}%")
    print(f"Success Rate: {result['success_rate']:.1f}%")

    # Check that patterns avoid defects
    print("\n--- Defect Avoidance Verification ---")

    cloth_poly, defect_polys = fitting_module.create_cloth_polygon(cloth)

    defect_intersections = 0
    for placement in result["placed_patterns"]:
        pattern_poly = placement.placement_polygon
        for i, defect in enumerate(defect_polys):
            if pattern_poly.intersects(defect):
                defect_intersections += 1
                print(
                    f"⚠ Pattern '{placement.pattern.name}' intersects defect #{i + 1}!"
                )

    if defect_intersections == 0:
        print(f"✓ All {result['patterns_placed']} patterns successfully avoid defects!")
    else:
        print(f"✗ Found {defect_intersections} pattern-defect intersections")

    # Check that patterns stay within irregular cloth boundary
    print("\n--- Irregular Boundary Verification ---")
    out_of_bounds = 0
    for placement in result["placed_patterns"]:
        pattern_poly = placement.placement_polygon
        if not cloth_poly.contains(pattern_poly):
            # Check coverage
            intersection = cloth_poly.intersection(pattern_poly)
            coverage = intersection.area / pattern_poly.area
            if coverage < 0.99:  # Allow 1% tolerance
                out_of_bounds += 1
                print(
                    f"⚠ Pattern '{placement.pattern.name}' extends outside cloth boundary (coverage: {coverage * 100:.1f}%)"
                )

    if out_of_bounds == 0:
        print(
            f"✓ All {result['patterns_placed']} patterns stay within irregular cloth boundaries!"
        )
    else:
        print(f"✗ Found {out_of_bounds} patterns extending outside boundaries")

    # Visualize results
    vis_path = "output/test_irregular_cloth_fitting_result.png"
    fitting_module.visualize(result, patterns, cloth, vis_path)
    print(f"\n✓ Fitting visualization saved: {vis_path}")

    # Generate report
    report_path = "output/test_irregular_cloth_fitting_report.txt"
    fitting_module.save_report(result, cloth, report_path)
    print(f"✓ Detailed report saved: {report_path}")

    return result


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST: Irregular Cloth with Defects")
    print("=" * 80)
    print("\nThis test verifies the system can handle real-world scenarios with:")
    print("  • Irregular cloth shapes (remnants, L-shapes, scraps)")
    print("  • Multiple defects (holes, stains, tears)")
    print("  • Pattern fitting that avoids defects")
    print("  • Patterns constrained to irregular boundaries")

    try:
        # Test 1: Cloth detection
        test_irregular_cloth_detection()

        # Test 2: Pattern fitting
        test_pattern_fitting_on_irregular_cloth()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nCheck the 'output' directory for visualizations and reports.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
