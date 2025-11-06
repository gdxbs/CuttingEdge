#!/usr/bin/env python3
"""
Comprehensive test of all cloth types with defect detection verification.

This test:
1. Tests each cloth category (free, Hole, Lines, Stain)
2. Verifies defect detection accuracy
3. Checks cloth shape extraction
4. Validates pattern shape extraction
5. Tests pattern fitting on different cloth types
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_fitting_module import PatternFittingModule
from cutting_edge.pattern_recognition_module import PatternRecognitionModule


def test_cloth_category(cloth_dir: str, category_name: str, max_samples: int = 3):
    """Test cloth samples from a specific category."""
    print(f"\n{'=' * 80}")
    print(f"TESTING CLOTH CATEGORY: {category_name.upper()}")
    print(f"{'=' * 80}")

    # Initialize modules with color-based segmentation (U-Net not trained)
    cloth_module = ClothRecognitionModule()
    cloth_module.use_unet = False

    # Get cloth files
    cloth_files = sorted(
        [f for f in os.listdir(cloth_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    )

    if not cloth_files:
        print(f"⚠ No cloth files found in {cloth_dir}")
        return []

    print(f"Found {len(cloth_files)} cloth samples")
    print(f"Testing {min(max_samples, len(cloth_files))} samples...")

    results = []

    for i, cloth_file in enumerate(cloth_files[:max_samples], 1):
        cloth_path = os.path.join(cloth_dir, cloth_file)
        print(
            f"\n--- Sample {i}/{min(max_samples, len(cloth_files))}: {cloth_file} ---"
        )

        try:
            # Process cloth
            cloth = cloth_module.process_image(cloth_path)

            # Check results
            print(f"  Type: {cloth.cloth_type}")
            print(f"  Dimensions: {cloth.width:.1f} x {cloth.height:.1f} cm")
            print(f"  Total Area: {cloth.total_area:.1f} cm²")
            print(f"  Usable Area: {cloth.usable_area:.1f} cm²")
            print(
                f"  Contour Points: {len(cloth.contour) if cloth.contour is not None else 0}"
            )
            print(
                f"  Defects Detected: {len(cloth.defects) if cloth.defects is not None else 0}"
            )

            # Calculate defect percentage
            if cloth.total_area > 0:
                defect_area = (
                    sum(cv2.contourArea(d) for d in (cloth.defects or []))
                    if cloth.defects
                    else 0
                )
                defect_pct = (defect_area / cloth.total_area) * 100
                print(f"  Defect Area %: {defect_pct:.2f}%")

            # Check shape complexity
            if cloth.contour is not None and len(cloth.contour) > 4:
                bbox_area = cloth.width * cloth.height
                shape_ratio = cloth.total_area / bbox_area if bbox_area > 0 else 1.0
                if shape_ratio < 0.85:
                    print(f"  Shape: IRREGULAR (complexity: {shape_ratio:.2f})")
                else:
                    print(f"  Shape: Regular rectangular")

            # Verify defect detection based on category
            expected_defects = {
                "free": 0,  # Clean cloth should have 0 defects
                "Hole": 1,  # Should detect at least 1 hole
                "Lines": 1,  # Should detect at least 1 line defect
                "Stain": 1,  # Should detect at least 1 stain
            }

            detected = len(cloth.defects) if cloth.defects else 0
            expected = expected_defects.get(category_name, 0)

            if category_name == "free":
                status = "✓" if detected == 0 else "⚠"
            else:
                status = "✓" if detected >= expected else "⚠"

            print(
                f"  {status} Expected: {'>=' if category_name != 'free' else '=='}{expected}, Detected: {detected}"
            )

            # Save visualization for inspection
            output_dir = "output/cloth_inspection"
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(
                output_dir, f"{category_name}_{cloth_file.replace('.jpg', '.png')}"
            )
            cloth_module.visualize(cloth, vis_path)
            print(f"  Visualization: {vis_path}")

            results.append(
                {
                    "file": cloth_file,
                    "category": category_name,
                    "detected_defects": detected,
                    "expected_defects": expected,
                    "status": status,
                    "cloth": cloth,
                }
            )

        except Exception as e:
            print(f"  ✗ Error processing {cloth_file}: {e}")
            import traceback

            traceback.print_exc()

    return results


def test_pattern_shapes(max_samples: int = 5):
    """Test pattern shape extraction accuracy."""
    print(f"\n{'=' * 80}")
    print(f"TESTING PATTERN SHAPE EXTRACTION")
    print(f"{'=' * 80}")

    pattern_module = PatternRecognitionModule()

    # Get pattern files from different types
    shape_dir = "images/shape"
    pattern_types = []

    for garment_type in os.listdir(shape_dir):
        garment_path = os.path.join(shape_dir, garment_type)
        if os.path.isdir(garment_path):
            for panel_type in os.listdir(garment_path):
                panel_path = os.path.join(garment_path, panel_type)
                if os.path.isdir(panel_path):
                    patterns = [f for f in os.listdir(panel_path) if f.endswith(".png")]
                    if patterns:
                        pattern_types.append(
                            (
                                garment_type,
                                panel_type,
                                os.path.join(panel_path, patterns[0]),
                            )
                        )
                        if len(pattern_types) >= max_samples:
                            break
        if len(pattern_types) >= max_samples:
            break

    print(f"Testing {len(pattern_types)} pattern types...")

    results = []

    for i, (garment, panel, pattern_path) in enumerate(pattern_types, 1):
        print(f"\n--- Pattern {i}: {garment}/{panel} ---")

        try:
            pattern = pattern_module.process_image(pattern_path)

            print(f"  File: {os.path.basename(pattern_path)}")
            print(
                f"  Type: {pattern.pattern_type} (confidence: {pattern.confidence:.2f})"
            )
            print(f"  Dimensions: {pattern.width:.1f} x {pattern.height:.1f} cm")
            print(f"  Area: {pattern.area:.1f} cm²")
            print(
                f"  Contour Points: {len(pattern.contour) if pattern.contour is not None else 0}"
            )

            # Check if contour matches dimensions
            if pattern.contour is not None and len(pattern.contour) > 2:
                try:
                    # Ensure contour is in the correct format for OpenCV
                    contour_int = pattern.contour.astype(np.int32)
                    bbox = cv2.boundingRect(contour_int)
                    bbox_w, bbox_h = bbox[2], bbox[3]

                    # Allow 10% tolerance (patterns can be irregular)
                    width_match = abs(bbox_w - pattern.width) / pattern.width < 0.10
                    height_match = abs(bbox_h - pattern.height) / pattern.height < 0.10

                    if width_match and height_match:
                        print(f"  ✓ Contour matches dimensions")
                    else:
                        print(
                            f"  ⚠ Contour mismatch: bbox={bbox_w:.1f}x{bbox_h:.1f}, expected={pattern.width:.1f}x{pattern.height:.1f}"
                        )
                except Exception as e:
                    print(f"  ⚠ Could not verify contour: {e}")

            results.append(
                {
                    "garment": garment,
                    "panel": panel,
                    "file": os.path.basename(pattern_path),
                    "pattern": pattern,
                }
            )

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()

    return results


def test_fitting_on_different_cloths(pattern_results, cloth_results):
    """Test pattern fitting on different cloth types."""
    print(f"\n{'=' * 80}")
    print(f"TESTING PATTERN FITTING ON DIFFERENT CLOTH TYPES")
    print(f"{'=' * 80}")

    fitting_module = PatternFittingModule()

    # Select patterns
    patterns = [r["pattern"] for r in pattern_results[:3]]

    if not patterns:
        print("⚠ No patterns available for fitting test")
        return

    print(f"Using {len(patterns)} patterns for fitting tests")

    # Group cloth results by category
    by_category = defaultdict(list)
    for result in cloth_results:
        by_category[result["category"]].append(result)

    # Test one cloth from each category
    for category, results in by_category.items():
        if not results:
            continue

        cloth_result = results[0]  # Take first sample from category
        cloth = cloth_result["cloth"]

        print(f"\n--- Testing on {category.upper()} cloth: {cloth_result['file']} ---")

        try:
            # Fit patterns
            fit_result = fitting_module.fit_patterns(patterns, cloth)

            print(
                f"  Patterns Placed: {fit_result['patterns_placed']}/{fit_result['patterns_total']}"
            )
            print(f"  Utilization: {fit_result['utilization_percentage']:.1f}%")
            print(f"  Success Rate: {fit_result['success_rate']:.1f}%")

            # Save visualization
            output_dir = "output/cloth_inspection"
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(
                output_dir,
                f"fitting_{category}_{cloth_result['file'].replace('.jpg', '.png')}",
            )
            fitting_module.visualize(fit_result, patterns, cloth, vis_path)
            print(f"  Visualization: {vis_path}")

            # Verify defect avoidance
            if cloth.defects and len(cloth.defects) > 0:
                from shapely.geometry import Polygon

                cloth_poly, defect_polys = fitting_module.create_cloth_polygon(cloth)

                intersections = 0
                for placement in fit_result["placed_patterns"]:
                    for defect in defect_polys:
                        if placement.placement_polygon.intersects(defect):
                            intersections += 1

                if intersections == 0:
                    print(f"  ✓ All patterns avoid defects")
                else:
                    print(f"  ⚠ {intersections} pattern-defect intersections found!")

        except Exception as e:
            print(f"  ✗ Fitting error: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Run comprehensive tests on all cloth types and patterns."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE CLOTH & PATTERN TESTING")
    print("=" * 80)
    print("\nThis test will:")
    print("  1. Test all cloth categories (free, Hole, Lines, Stain)")
    print("  2. Verify defect detection accuracy")
    print("  3. Check pattern shape extraction")
    print("  4. Test fitting on different cloth types")
    print("  5. Generate inspection visualizations")

    # Test each cloth category
    cloth_results = []

    cloth_categories = {
        "free": "images/cloth/free",
        "Hole": "images/cloth/Hole",
        "Lines": "images/cloth/Lines",
        "Stain": "images/cloth/Stain",
    }

    for category, directory in cloth_categories.items():
        if os.path.exists(directory):
            results = test_cloth_category(directory, category, max_samples=3)
            cloth_results.extend(results)

    # Test pattern shapes
    pattern_results = test_pattern_shapes(max_samples=5)

    # Test fitting on different cloth types
    test_fitting_on_different_cloths(pattern_results, cloth_results)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 80}")

    # Cloth detection summary
    by_category = defaultdict(list)
    for result in cloth_results:
        by_category[result["category"]].append(result)

    print("\nCloth Defect Detection:")
    for category, results in by_category.items():
        correct = sum(1 for r in results if r["status"] == "✓")
        total = len(results)
        print(f"  {category:10s}: {correct}/{total} correct")

    print(f"\nPattern Extraction: {len(pattern_results)} patterns tested")

    print(f"\nAll visualizations saved to: output/cloth_inspection/")
    print(f"\n{'=' * 80}")
    print("TEST COMPLETE - Check output/cloth_inspection/ for visual verification")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    sys.exit(main() or 0)
