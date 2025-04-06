import argparse
import logging
import os
from typing import Dict, List, Optional

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_fitting_module import PatternFittingModule
from cutting_edge.pattern_recognition_module import PatternRecognitionModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Main entry point for the Cutting Edge system

This application analyzes garment patterns and cloth materials for:
1. Pattern recognition and classification
2. Cloth material analysis
3. Optimal pattern placement on cloth

References:
- GarmentCodeData dataset: https://www.research-collection.ethz.ch/handle/20.500.11850/690432
- "GarmentCode: Physics-based automatic patterning of 3D garment models" (Korosteleva & Lee, 2021)
- "Planning Irregular Object Packing via Hierarchical Reinforcement Learning" (Wang et al., 2022)
"""


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cutting Edge Pattern Recognition")

    # Dataset and model paths
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to GarmentCodeData dataset"
    )
    parser.add_argument(
        "--pattern_model_path",
        type=str,
        default="models/pattern_recognition_model.pth",
        help="Path to save/load pattern recognition model",
    )
    parser.add_argument(
        "--fitting_model_path",
        type=str,
        default="models/pattern_fitting_model.pth",
        help="Path to save/load pattern fitting model",
    )

    # Mode selection
    parser.add_argument(
        "--train", action="store_true", help="Train the model even if it exists"
    )
    parser.add_argument(
        "--train_fitting", action="store_true", help="Train the pattern fitting model"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--fitting_episodes",
        type=int,
        default=1,
        help="Number of fitting training episodes",
    )

    # Input files for inference
    parser.add_argument(
        "--pattern_image",
        type=str,
        default=None,
        help="Path to input pattern image for inference",
    )
    parser.add_argument(
        "--cloth_image",
        type=str,
        default=None,
        help="Path to input cloth image for inference",
    )
    parser.add_argument(
        "--multi_pattern",
        action="store_true",
        help="Process multiple pattern images for fitting",
    )
    parser.add_argument(
        "--pattern_dir",
        type=str,
        default=None,
        help="Directory containing multiple pattern images",
    )

    # Visualization options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization for pattern fitting",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output visualizations",
    )

    return parser.parse_args()


def setup_directories(args: argparse.Namespace) -> None:
    """Create necessary directories for models and output"""
    os.makedirs(os.path.dirname(args.pattern_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.fitting_model_path), exist_ok=True)
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)


def initialize_pattern_recognizer(args: argparse.Namespace) -> PatternRecognitionModule:
    """Initialize and optionally train the pattern recognition module"""
    logger.info("Initializing pattern recognition module...")

    model_path = (
        args.pattern_model_path if os.path.exists(args.pattern_model_path) else None
    )
    pattern_recognizer = PatternRecognitionModule(
        dataset_path=args.dataset_path,
        model_path=model_path,
    )

    # Train model if requested or needed
    if args.train and args.dataset_path:
        logger.info(f"Training pattern recognition model for {args.epochs} epochs...")
        pattern_recognizer.train(args.epochs)
        pattern_recognizer.save_model(args.pattern_model_path)
        logger.info(f"Model saved to {args.pattern_model_path}")
    elif not args.train:
        pattern_recognizer.check_and_train(args.pattern_model_path, args.epochs)

    return pattern_recognizer


def process_patterns(
    pattern_recognizer: PatternRecognitionModule, pattern_paths: List[str]
) -> List[Dict]:
    """Process multiple pattern images"""
    patterns_data = []

    for path in pattern_paths:
        if not os.path.exists(path):
            logger.warning(f"Pattern image not found: {path}")
            continue

        logger.info(f"Processing pattern image: {path}")
        pattern_image = cv2.imread(path)
        if pattern_image is None:
            logger.warning(f"Failed to load pattern image: {path}")
            continue

        pattern_data = pattern_recognizer.process_pattern(pattern_image)
        pattern_data["image_path"] = path
        patterns_data.append(pattern_data)

        logger.info(f"Pattern type: {pattern_data['pattern_type']}")
        logger.info(f"Pattern dimensions: {pattern_data['dimensions']}")

    return patterns_data


def process_multiple_patterns(
    args: argparse.Namespace, pattern_recognizer: PatternRecognitionModule
) -> Optional[List[Dict]]:
    """Process multiple pattern images from a directory"""
    pattern_dir = args.pattern_dir or os.path.dirname(args.pattern_image)
    if not pattern_dir or not os.path.exists(pattern_dir):
        logger.error(f"Pattern directory not found: {pattern_dir}")
        return None

    pattern_paths = [
        os.path.join(pattern_dir, f)
        for f in os.listdir(pattern_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not pattern_paths:
        logger.error(f"No pattern images found in: {pattern_dir}")
        return None

    patterns_data = process_patterns(pattern_recognizer, pattern_paths)
    if not patterns_data:
        logger.error("Failed to process any pattern images")
        return None

    return patterns_data


def process_single_pattern(
    args: argparse.Namespace, pattern_recognizer: PatternRecognitionModule
) -> Optional[List[Dict]]:
    """Process a single pattern image and visualize results"""
    if not os.path.exists(args.pattern_image):
        logger.error(f"Pattern image not found: {args.pattern_image}")
        return None

    logger.info(f"Processing pattern image: {args.pattern_image}")
    pattern_image = cv2.imread(args.pattern_image)
    if pattern_image is None:
        logger.error(f"Failed to load pattern image: {args.pattern_image}")
        return None

    # Convert BGR to RGB for display
    pattern_image_rgb = cv2.cvtColor(pattern_image, cv2.COLOR_BGR2RGB)
    pattern_data = pattern_recognizer.process_pattern(pattern_image)
    patterns_data = [pattern_data]  # Add to list for fitting

    logger.info(f"Pattern type: {pattern_data['pattern_type']}")
    logger.info(f"Pattern dimensions: {pattern_data['dimensions']}")

    # Visualize pattern results
    visualize_pattern_results(
        pattern_image_rgb, pattern_data, args.output_dir, args.train_fitting
    )

    return patterns_data


def visualize_pattern_results(
    pattern_image_rgb, pattern_data, output_dir, skip_show=False
):
    """Visualize pattern recognition results"""
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(231)
    plt.imshow(pattern_image_rgb)
    plt.title("Input Pattern")

    # Draw contours on the pattern image
    if "contours" in pattern_data and pattern_data["contours"]:
        contour_img = pattern_image_rgb.copy()
        cv2.drawContours(contour_img, pattern_data["contours"], -1, (0, 255, 0), 2)
        plt.subplot(232)
        plt.imshow(contour_img)
        plt.title("Detected Pattern Contours")

        # Create a binary mask of the pattern area
        pattern_mask = np.zeros(pattern_image_rgb.shape[:2], dtype=np.uint8)
        cv2.drawContours(pattern_mask, pattern_data["contours"], -1, 255, -1)
        plt.subplot(233)
        plt.imshow(pattern_mask, cmap="gray")
        plt.title("Pattern Mask")

        # Create an isolated pattern image
        pattern_only = cv2.bitwise_and(
            pattern_image_rgb, pattern_image_rgb, mask=pattern_mask
        )
        plt.subplot(234)
        plt.imshow(pattern_only)
        plt.title("Isolated Pattern")

        # Create a filled contour visualization
        filled_vis = np.zeros_like(pattern_image_rgb)
        cv2.drawContours(filled_vis, pattern_data["contours"], -1, (0, 255, 0), -1)
        # Add transparency for better visualization
        overlay = cv2.addWeighted(pattern_image_rgb, 1, filled_vis, 0.5, 0)
        plt.subplot(235)
        plt.imshow(overlay)
        plt.title("Pattern Overlay")

    # Show pattern information
    plt.subplot(236)
    dimensions = pattern_data.get("dimensions", None)
    pattern_type = pattern_data.get("pattern_type", "Unknown")
    info_text = f"Pattern Type: {pattern_type}\n"
    if dimensions is not None:
        info_text += f"Dimensions: {dimensions[0]:.1f} x {dimensions[1]:.1f}"

    plt.text(
        0.5,
        0.5,
        info_text,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        transform=plt.gca().transAxes,
    )
    plt.axis("off")
    plt.title("Pattern Information")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pattern_detection.png"))
    if not skip_show:
        plt.show()
    plt.close()


def process_cloth_image(args: argparse.Namespace):
    """Process cloth image and visualize results"""
    # Initialize cloth recognition module
    logger.info("Initializing cloth recognition module...")
    cloth_recognizer = ClothRecognitionModule()

    # Check if file exists
    if not os.path.exists(args.cloth_image):
        logger.error(f"Cloth image not found: {args.cloth_image}")
        return None, None

    logger.info(f"Processing cloth image: {args.cloth_image}")
    cloth_image = cv2.imread(args.cloth_image)
    if cloth_image is None:
        logger.error(f"Failed to load cloth image: {args.cloth_image}")
        return None, None

    # Process the cloth image
    cloth_data = cloth_recognizer.process_cloth(cloth_image)

    if cloth_data.get("error"):
        logger.error(f"Processing failed with error: {cloth_data['error']}")

    # Log dimensions
    logger.info(f"Cloth dimensions (WxH): {cloth_data['dimensions']}")
    logger.info(f"Cloth area (pixels): {cloth_data['area']:.2f}")

    # Convert BGR to RGB for display
    cloth_image_rgb = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)

    # Visualize cloth results
    visualize_cloth_results(
        cloth_image_rgb,
        cloth_data,
        args.output_dir,
        args.train_fitting,
    )

    return cloth_data, cloth_image_rgb


def visualize_cloth_results(cloth_image_rgb, cloth_data, output_dir, skip_show=False):
    """Visualize cloth recognition results"""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(231)
    plt.imshow(cloth_image_rgb)
    plt.title("Input Cloth")

    # Edge detection
    plt.subplot(232)
    if cloth_data.get("edges") is not None:
        plt.imshow(cloth_data["edges"], cmap="gray")
    plt.title("Cloth Edges")

    # Contours on the original image
    contour_img = cloth_image_rgb.copy()
    if cloth_data.get("contours"):
        cv2.drawContours(contour_img, cloth_data["contours"], -1, (0, 255, 0), 2)
    plt.subplot(233)
    plt.imshow(contour_img)
    plt.title("Cloth Contours")

    # Cloth Mask
    plt.subplot(234)
    if cloth_data.get("cloth_mask") is not None:
        plt.imshow(cloth_data["cloth_mask"], cmap="gray")
    plt.title("Cloth Mask")

    # Empty plot
    plt.subplot(235)
    plt.axis("off")
    plt.title("Empty")

    # Segmentation if available
    plt.subplot(236)
    if cloth_data.get("segmented_image") is not None:
        seg_img = cloth_data["segmented_image"]
        if seg_img.ndim == 3 and seg_img.shape[0] == 1:
            seg_img = seg_img.squeeze(0)
        plt.imshow(seg_img, cmap="viridis")
        plt.title("Semantic Segmentation")
    else:
        plt.text(
            0.5,
            0.5,
            "Segmentation N/A",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title("Semantic Segmentation")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cloth_detection_results.png")
    plt.savefig(save_path)
    logger.info(f"Visualization saved to: {save_path}")
    if not skip_show:
        plt.show()
    plt.close()


def visualize_training_result(training_result, output_dir):
    """Visualize pattern fitting training results"""
    plt.figure(figsize=(10, 10))
    plt.imshow(training_result["best_state"], cmap="gray")
    plt.title(
        f"Best Training Result (Utilization: {training_result['best_utilization']:.4f})"
    )
    plt.colorbar(label="Pattern Index")
    plt.savefig(os.path.join(output_dir, "training_result.png"))
    plt.show()
    plt.close()


def visualize_fitting_result(result, cloth_image, save_path):
    """Visualize pattern fitting results"""
    plt.figure(figsize=(16, 10))

    # Define colors for patterns
    colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "yellow"]

    # Get canvas dimensions
    canvas = result.get("final_state", np.zeros((400, 400), dtype=int))
    cloth_width, cloth_height = result.get(
        "cloth_dims", (canvas.shape[1], canvas.shape[0])
    )

    # Get cloth mask for boundary checking from result or cloth_data
    cloth_mask = None
    if "visualization_mask" in result and result["visualization_mask"] is not None:
        # If the mask is already in the result, use it
        cloth_mask = result.get("visualization_mask")
        # Resize to match visualization dimensions
        if cloth_mask is not None:
            try:
                cloth_mask = cv2.resize(cloth_mask, (cloth_width, cloth_height))
            except Exception as e:
                logger.warning(f"Error resizing cloth mask: {e}")
    elif cloth_image is not None:
        # Fallback: create a mask from the image
        try:
            # Create a binary mask from the cloth image
            if len(cloth_image.shape) == 3:
                # Convert to grayscale if color image
                gray_cloth = cv2.cvtColor(cloth_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_cloth = cloth_image

            # Threshold to get binary mask
            _, cloth_mask = cv2.threshold(gray_cloth, 10, 255, cv2.THRESH_BINARY)

            # Resize to match the cloth dimensions from result
            cloth_mask = cv2.resize(cloth_mask, (cloth_width, cloth_height))
        except Exception as e:
            logger.warning(f"Error creating cloth mask for visualization: {e}")

    # Plot original cloth
    plt.subplot(1, 2, 1)
    if cloth_image is not None:
        plt.imshow(cloth_image)
    else:
        plt.imshow(np.ones((cloth_height, cloth_width, 3)) * 0.8)  # Light gray
    plt.title("Original Cloth")
    plt.axis("off")

    # Create overlay visualization
    plt.subplot(1, 2, 2)

    # Start with cloth image or blank canvas
    if cloth_image is not None:
        # Resize cloth image to match the dimensions from the environment
        try:
            overlay_img = cv2.resize(cloth_image.copy(), (cloth_width, cloth_height))
            if len(overlay_img.shape) == 2:
                overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2RGB)

            # Create background
            background = (
                np.ones((cloth_height, cloth_width, 3), dtype=np.uint8) * 240
            )  # Light gray

            # Create a mask for the cloth part only (255 = cloth, 0 = background)
            if cloth_mask is not None:
                # Create binary mask from cloth
                _, binary_mask = cv2.threshold(cloth_mask, 1, 255, cv2.THRESH_BINARY)

                # Apply the mask - show cloth image only where the mask is non-zero
                for c in range(3):
                    background[:, :, c] = np.where(
                        binary_mask > 0, overlay_img[:, :, c], background[:, :, c]
                    )
                overlay_img = background
        except Exception as e:
            logger.warning(f"Error preparing overlay image: {e}")
            # Fallback to blank canvas
            overlay_img = np.ones((cloth_height, cloth_width, 3), dtype=np.uint8) * 240
    else:
        overlay_img = np.ones((cloth_height, cloth_width, 3), dtype=np.uint8) * 240

    # Draw each placed pattern
    for i, placement in enumerate(result.get("placements", [])):
        pattern_idx = placement["pattern_id"]
        pos_x, pos_y = placement["position"]
        rotation = placement["rotation"]
        color_idx = i % len(colors)
        color = plt.cm.colors.to_rgb(colors[color_idx])

        # Get pattern shape
        if pattern_idx < len(result.get("patterns", [])):
            pattern = result["patterns"][pattern_idx]

            if "contours" in pattern and pattern["contours"]:
                # Draw each contour
                for contour in pattern["contours"]:
                    # Convert to numpy array
                    contour_np = np.array(contour)

                    # Handle reshape if needed
                    if len(contour_np.shape) == 3 and contour_np.shape[1] == 1:
                        contour_np = contour_np.reshape(contour_np.shape[0], 2)

                    # Rotate contour
                    center = np.mean(contour_np, axis=0)
                    angle_rad = np.radians(rotation)
                    rot_mat = np.array(
                        [
                            [np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)],
                        ]
                    )
                    rotated = np.dot(contour_np - center, rot_mat.T) + center

                    # Translate contour
                    translated = rotated + np.array([pos_x, pos_y])

                    # Draw contour on overlay with safety checks
                    try:
                        # Create mask with same dimensions as overlay_img
                        h, w = overlay_img.shape[:2]
                        mask = np.zeros((h, w), dtype=np.uint8)

                        # Convert to integer and constrain to image bounds
                        points = translated.astype(np.int32)
                        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                        points[:, 1] = np.clip(points[:, 1], 0, h - 1)

                        # Only draw if we have valid points
                        if len(points) >= 3:
                            cv2.fillPoly(mask, [points], 1)

                            # If we have a cloth mask, only draw within the cloth area
                            if cloth_mask is not None:
                                # Use a binary cloth mask for proper filtering
                                binary_mask = cloth_mask.copy()
                                if np.max(binary_mask) > 1:
                                    # Normalize to binary 0/1 mask
                                    _, binary_mask = cv2.threshold(
                                        binary_mask, 1, 1, cv2.THRESH_BINARY
                                    )
                                mask = mask * binary_mask

                            # Apply color with transparency
                            for c in range(3):
                                overlay_img[:, :, c] = np.where(
                                    mask > 0,
                                    overlay_img[:, :, c] * 0.3 + color[c] * 255 * 0.7,
                                    overlay_img[:, :, c],
                                )
                    except Exception as e:
                        logger.warning(f"Error drawing contour: {e}")

    # Make a final pass to mask out any pattern fragments outside the cloth
    if cloth_mask is not None:
        try:
            # Create final mask
            _, final_mask = cv2.threshold(cloth_mask, 1, 1, cv2.THRESH_BINARY)

            # Create a background canvas
            background = np.ones((cloth_height, cloth_width, 3), dtype=np.uint8) * 240

            # Apply final mask to the overlay image
            for c in range(3):
                overlay_img[:, :, c] = np.where(
                    final_mask > 0, overlay_img[:, :, c], background[:, :, c]
                )
        except Exception as e:
            logger.warning(f"Error in final masking: {e}")

    # Display the result
    plt.imshow(overlay_img.astype(np.uint8))
    plt.title(f"Pattern Placement (Utilization: {result['utilization']:.1%})")
    plt.axis("off")

    # Add legend
    handles = []
    for i, placement in enumerate(result.get("placements", [])):
        pattern_idx = placement["pattern_id"]
        color_idx = i % len(colors)
        handles.append(
            patches.Patch(color=colors[color_idx], label=f"Pattern {pattern_idx}")
        )

    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {save_path}")

    # Show the plot
    plt.show()
    plt.close()


def fit_patterns_to_cloth(args, patterns_data, cloth_data, cloth_image_rgb):
    """Fit patterns to cloth and visualize results"""
    logger.info("Initializing pattern fitting module...")

    # Check if model exists
    model_path = (
        args.fitting_model_path if os.path.exists(args.fitting_model_path) else None
    )

    # Initialize pattern fitting module
    pattern_fitter = PatternFittingModule(model_path=model_path)

    # Train fitting model if requested
    if args.train_fitting:
        logger.info(
            f"Training pattern fitting model for {args.fitting_episodes} episodes..."
        )
        training_result = pattern_fitter.train(
            cloth_data=cloth_data,
            patterns_data=patterns_data,
            num_episodes=args.fitting_episodes,
        )
        logger.info(
            f"Fitting model trained. Best utilization: {training_result['best_utilization']:.4f}"
        )

        # Visualize training result
        visualize_training_result(training_result, args.output_dir)

    # Perform pattern fitting
    logger.info("Fitting patterns to cloth...")
    fitting_result = pattern_fitter.fit_patterns(
        cloth_data=cloth_data,
        patterns_data=patterns_data,
        visualize=args.visualize,
    )

    # Add patterns to result for visualization
    fitting_result["patterns"] = patterns_data

    # Visualize fitting result
    logger.info(
        f"Pattern fitting complete. Utilization: {fitting_result['utilization']:.4f}"
    )
    visualize_fitting_result(
        result=fitting_result,
        cloth_image=cloth_image_rgb,
        save_path=os.path.join(args.output_dir, "pattern_fitting_result.png"),
    )

    return fitting_result


def main():
    """Main execution function"""
    args = parse_arguments()

    try:
        # Setup directories
        setup_directories(args)

        # Initialize and train pattern recognition model
        pattern_recognizer = initialize_pattern_recognizer(args)

        # Process pattern images
        patterns_data = None

        if args.multi_pattern or args.pattern_dir:
            # Process multiple patterns for fitting
            patterns_data = process_multiple_patterns(args, pattern_recognizer)
            if patterns_data is None:
                return
        elif args.pattern_image:
            # Process single pattern image
            patterns_data = process_single_pattern(args, pattern_recognizer)
            if patterns_data is None:
                return

        # Process cloth image if provided
        cloth_data = None
        cloth_image_rgb = None
        if args.cloth_image:
            cloth_data, cloth_image_rgb = process_cloth_image(args)
            if cloth_data is None:
                return

        # If both pattern and cloth are provided, perform pattern fitting
        if patterns_data and cloth_data and cloth_image_rgb is not None:
            result = fit_patterns_to_cloth(
                args, patterns_data, cloth_data, cloth_image_rgb
            )
            logger.info(
                f"Successfully placed {len(result['placements'])} patterns with {result['utilization']:.2%} utilization"
            )

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU memory exceeded. Try reducing batch size.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Print stack trace for debugging
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
