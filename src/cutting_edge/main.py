import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_fitting_module import PatternFittingModule
from cutting_edge.pattern_recognition_module import PatternRecognitionModule

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Main entry point for the Cutting Edge system
# This application analyzes garment patterns and cloth materials for:
# 1. Pattern recognition and classification
# 2. Cloth material analysis
# 3. Optimal pattern placement on cloth using Hierarchical Reinforcement Learning
#
# References:
# - GarmentCodeData dataset: https://www.research-collection.ethz.ch/handle/20.500.11850/690432
# - "GarmentCode: Physics-based automatic patterning of 3D garment models" (Korosteleva & Lee, 2021)
#   https://doi.org/10.1145/3478513.3480489
# - "Planning Irregular Object Packing via Hierarchical Reinforcement Learning" (Wang et al., 2022)


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


def process_patterns(
    pattern_recognizer: PatternRecognitionModule, pattern_paths: List[str]
) -> List[Dict[str, Any]]:
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


def setup_directories(args: argparse.Namespace) -> None:
    """Create necessary directories for models and output."""
    os.makedirs(os.path.dirname(args.pattern_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.fitting_model_path), exist_ok=True)
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)


def initialize_and_train_pattern_recognizer(
    args: argparse.Namespace,
) -> PatternRecognitionModule:
    """Initialize and optionally train the pattern recognition module."""
    logger.info("Initializing pattern recognition module...")

    # Handle model path properly for type checking
    model_path: Optional[str] = None
    if os.path.exists(args.pattern_model_path):
        model_path = args.pattern_model_path

    pattern_recognizer = PatternRecognitionModule(
        dataset_path=args.dataset_path,
        model_path=model_path,
    )

    # Train pattern recognition model if requested or needed
    if args.train and args.dataset_path:
        logger.info(f"Training pattern recognition model for {args.epochs} epochs...")
        pattern_recognizer.train(args.epochs)
        pattern_recognizer.save_model(args.pattern_model_path)
        logger.info(f"Model saved to {args.pattern_model_path}")
    elif not args.train:
        pattern_recognizer.check_and_train(args.pattern_model_path, args.epochs)

    return pattern_recognizer


def process_multiple_patterns(
    args: argparse.Namespace, pattern_recognizer: PatternRecognitionModule
) -> Optional[List[Dict[str, Any]]]:
    """Process multiple pattern images from a directory."""
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
) -> Optional[List[Dict[str, Any]]]:
    """Process a single pattern image and visualize results."""
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
    pattern_image_rgb: np.ndarray,
    pattern_data: Dict[str, Any],
    output_dir: str,
    skip_show: bool = False,
) -> None:
    """Visualize pattern recognition results with enhanced visualization."""
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(231)
    plt.imshow(pattern_image_rgb)
    plt.title("Input Pattern")

    # Draw contours on a copy of the pattern image
    if "contours" in pattern_data and pattern_data["contours"]:
        contour_img = pattern_image_rgb.copy()
        cv2.drawContours(contour_img, pattern_data["contours"], -1, (0, 255, 0), 2)
        plt.subplot(232)
        plt.imshow(contour_img)
        plt.title("Detected Pattern Contours")
        
        # Create a binary mask of just the pattern area
        pattern_mask = np.zeros(pattern_image_rgb.shape[:2], dtype=np.uint8)
        cv2.drawContours(pattern_mask, pattern_data["contours"], -1, 255, -1)
        plt.subplot(233)
        plt.imshow(pattern_mask, cmap="gray")
        plt.title("Pattern Mask")
        
        # Create an isolated pattern image (just the pattern without background)
        pattern_only = cv2.bitwise_and(
            pattern_image_rgb,
            pattern_image_rgb,
            mask=pattern_mask
        )
        plt.subplot(234)
        plt.imshow(pattern_only)
        plt.title("Isolated Pattern")
        
        # Create a filled contour visualization for better clarity
        filled_vis = np.zeros_like(pattern_image_rgb)
        color = (0, 255, 0)  # Green fill
        cv2.drawContours(filled_vis, pattern_data["contours"], -1, color, -1)
        # Add transparency for better visualization
        alpha = 0.5
        overlay = cv2.addWeighted(pattern_image_rgb, 1, filled_vis, alpha, 0)
        plt.subplot(235)
        plt.imshow(overlay)
        plt.title("Pattern Overlay")
    
    # Show additional pattern information
    plt.subplot(236)
    dimensions = pattern_data.get("dimensions", None)
    pattern_type = pattern_data.get("pattern_type", "Unknown")
    info_text = f"Pattern Type: {pattern_type}\n"
    if dimensions is not None:
        info_text += f"Dimensions: {dimensions[0]:.1f} x {dimensions[1]:.1f}"
    
    # Create a text-only subplot with pattern information
    plt.text(0.5, 0.5, info_text, 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12,
             transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title("Pattern Information")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pattern_detection.png"))
    if not skip_show:
        plt.show()


def process_cloth_image(
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """Process cloth image and visualize results."""
    # Initialize cloth recognition module
    logger.info("Initializing cloth recognition module...")
    # You might need to pass model paths or configs here if they are not hardcoded
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
        # Decide if you want to return partial data or None
        # return None, None # Option 1: Return nothing on error
        # Option 2: Return partial data (current behavior)

    # Log dimensions (use the final dimensions from cloth_data)
    logger.info(f"Cloth dimensions (WxH): {cloth_data['dimensions']}")
    logger.info(f"Cloth area (pixels): {cloth_data['area']:.2f}")


    # Convert BGR to RGB for display *after* processing
    cloth_image_rgb = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)

    # Visualize cloth results
    visualize_cloth_results(
        cloth_image_rgb, cloth_data, args.output_dir, args.train_fitting # Use skip_show from args
    )

    return cloth_data, cloth_image_rgb


def visualize_cloth_results(
    cloth_image_rgb: np.ndarray,
    cloth_data: Dict[str, Any],
    output_dir: str,
    skip_show: bool = False,
) -> None:
    """Visualize cloth recognition results with enhanced mask visualization."""
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(231)
    plt.imshow(cloth_image_rgb)
    plt.title("Input Cloth")

    # Edge detection - Should look better now if mask is good
    plt.subplot(232)
    if cloth_data.get("edges") is not None:
        plt.imshow(cloth_data["edges"], cmap="gray")
    plt.title("Cloth Edges (from Mask)")

    # Contours on the original image
    contour_img = cloth_image_rgb.copy()
    if cloth_data.get("contours"): # Check if list is not empty
        cv2.drawContours(contour_img, cloth_data["contours"], -1, (0, 255, 0), 2)
    plt.subplot(233)
    plt.imshow(contour_img)
    plt.title("Cloth Contours (Largest)")

    # Cloth Mask - Should now show the white cloth area
    plt.subplot(234)
    if cloth_data.get("cloth_mask") is not None:
        plt.imshow(cloth_data["cloth_mask"], cmap="gray")
    plt.title("Cloth Mask (Thresholded & Cleaned)")
    
    # Placeholder for 5th plot if needed, or leave empty
    plt.subplot(235)
    plt.axis('off') # Hide axes if unused
    plt.title("Empty")

    # Show segmentation if available
    plt.subplot(236)
    if cloth_data.get("segmented_image") is not None:
        # Squeeze might be needed depending on how segmentation map is stored
        seg_img = cloth_data["segmented_image"]
        if seg_img.ndim == 3 and seg_img.shape[0] == 1: # Handle potential channel dim
             seg_img = seg_img.squeeze(0)
        plt.imshow(seg_img, cmap="viridis") # Adjust cmap as needed
        plt.title("Semantic Segmentation")
    else:
        plt.text(0.5, 0.5, 'Segmentation N/A', horizontalalignment='center', verticalalignment='center')
        plt.title("Semantic Segmentation")


    plt.tight_layout()
    save_path = os.path.join(output_dir, "cloth_detection_results.png")
    plt.savefig(save_path)
    logger.info(f"Visualization saved to: {save_path}")
    if not skip_show:
        plt.show()
    plt.close() # Close the figure to free memory

def fit_patterns_to_cloth(
    args: argparse.Namespace,
    patterns_data: List[Dict[str, Any]],
    cloth_data: Dict[str, Any],
    cloth_image_rgb: np.ndarray,
) -> None:
    """Fit patterns to cloth and visualize results."""
    logger.info("Initializing pattern fitting module...")

    # Handle model path properly for type checking
    model_path: Optional[str] = None
    if os.path.exists(args.fitting_model_path):
        model_path = args.fitting_model_path

    # Initialize pattern fitting module with model path
    pattern_fitter = PatternFittingModule(model_path=model_path)

    # Train fitting model if requested
    if args.train_fitting:
        train_fitting_model(
            pattern_fitter,
            args.fitting_episodes,
            cloth_data,
            patterns_data,
            args.output_dir,
        )

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
    pattern_fitter.visualize_result(
        result=fitting_result,
        cloth_image=cloth_image_rgb,
        save_path=os.path.join(args.output_dir, "pattern_fitting_result.png"),
    )


def train_fitting_model(
    pattern_fitter: PatternFittingModule,
    num_episodes: int,
    cloth_data: Dict[str, Any],
    patterns_data: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Train the pattern fitting model."""
    logger.info(f"Training pattern fitting model for {num_episodes} episodes...")
    training_result = pattern_fitter.train(
        cloth_data=cloth_data,
        patterns_data=patterns_data,
        num_episodes=num_episodes,
    )
    logger.info(
        f"Fitting model trained. Best utilization: {training_result['best_utilization']:.4f}"
    )

    # Visualize training result
    plt.figure(figsize=(10, 10))
    plt.imshow(training_result["best_state"], cmap="gray")
    plt.title(
        f"Best Training Result (Utilization: {training_result['best_utilization']:.4f})"
    )
    plt.colorbar(label="Pattern Index")
    plt.savefig(os.path.join(output_dir, "training_result.png"))
    plt.show()


def main() -> None:
    """Main execution function"""
    args = parse_arguments()

    try:
        # Setup directories
        setup_directories(args)

        # Initialize and train pattern recognition model
        pattern_recognizer = initialize_and_train_pattern_recognizer(args)

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
            fit_patterns_to_cloth(args, patterns_data, cloth_data, cloth_image_rgb)

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
