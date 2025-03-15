import argparse
import logging
import os

import cv2
import matplotlib.pyplot as plt
import torch

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_recognition_module import PatternRecognitionModule

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Main entry point for the Cutting Edge system
# This application analyzes garment patterns and cloth materials for:
# 1. Pattern recognition and classification
# 2. Cloth material analysis 
# 3. Optimal pattern placement on cloth (future implementation)
#
# References:
# - GarmentCodeData dataset: https://www.research-collection.ethz.ch/handle/20.500.11850/690432
# - "GarmentCode: Physics-based automatic patterning of 3D garment models" (Korosteleva & Lee, 2021)
#   https://doi.org/10.1145/3478513.3480489

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cutting Edge Pattern Recognition")

    # Dataset and model paths
    parser.add_argument("--dataset_path", type=str, default=None, 
                        help="Path to GarmentCodeData dataset")
    parser.add_argument("--model_path", type=str, default="models/pattern_recognition_model.pth", 
                        help="Path to save/load model")

    # Mode selection
    parser.add_argument("--train", action="store_true", 
                        help="Train the model even if it exists")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")

    # Input files for inference
    parser.add_argument("--pattern_image", type=str, default=None, 
                        help="Path to input pattern image for inference")
    parser.add_argument("--cloth_image", type=str, default=None, 
                        help="Path to input cloth image for inference")

    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()

    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

        # Initialize modules
        logger.info("Initializing pattern recognition module...")
        pattern_recognizer = PatternRecognitionModule(
            dataset_path=args.dataset_path, 
            model_path=args.model_path if os.path.exists(args.model_path) else None
        )

        # Train model if requested or needed
        if args.train and args.dataset_path:
            logger.info(f"Training model for {args.epochs} epochs...")
            pattern_recognizer.train(args.epochs)
            pattern_recognizer.save_model(args.model_path)
            logger.info(f"Model saved to {args.model_path}")
        elif not args.train:
            pattern_recognizer.check_and_train(args.model_path, args.epochs)

        # If no input images provided, exit after training
        if not args.pattern_image and not args.cloth_image:
            logger.info("No input images provided. Exiting after model preparation.")
            return

        # Process pattern image if provided
        if args.pattern_image:
            # Check if file exists
            if not os.path.exists(args.pattern_image):
                logger.error(f"Pattern image not found: {args.pattern_image}")
                return

            logger.info(f"Processing pattern image: {args.pattern_image}")
            pattern_image = cv2.imread(args.pattern_image)
            if pattern_image is None:
                logger.error(f"Failed to load pattern image: {args.pattern_image}")
                return

            # Convert BGR to RGB for display
            pattern_image_rgb = cv2.cvtColor(pattern_image, cv2.COLOR_BGR2RGB)
            pattern_data = pattern_recognizer.process_pattern(pattern_image)

            logger.info(f"Pattern type: {pattern_data['pattern_type']}")
            logger.info(f"Pattern dimensions: {pattern_data['dimensions']}")

            # Visualize pattern results
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(pattern_image_rgb)
            plt.title("Input Pattern")

            # Draw contours on a copy of the pattern image
            if 'contours' in pattern_data and pattern_data['contours']:
                contour_img = pattern_image_rgb.copy()
                cv2.drawContours(contour_img, pattern_data['contours'], -1, (0, 255, 0), 2)
                plt.subplot(122)
                plt.imshow(contour_img)
                plt.title("Detected Pattern Contours")

            plt.tight_layout()
            plt.show()

        # Initialize cloth recognition module
        cloth_recognizer = ClothRecognitionModule()

        # Process cloth image if provided
        if args.cloth_image:
            # Check if file exists
            if not os.path.exists(args.cloth_image):
                logger.error(f"Cloth image not found: {args.cloth_image}")
                return

            logger.info(f"Processing cloth image: {args.cloth_image}")
            cloth_image = cv2.imread(args.cloth_image)
            if cloth_image is None:
                logger.error(f"Failed to load cloth image: {args.cloth_image}")
                return

            # Convert BGR to RGB for display
            cloth_image_rgb = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)
            cloth_data = cloth_recognizer.process_cloth(cloth_image)

            logger.info(f"Cloth dimensions: {cloth_data['dimensions']}")

            # Visualize cloth results
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(cloth_image_rgb)
            plt.title("Input Cloth")

            plt.subplot(132)
            plt.imshow(cloth_data['edges'], cmap='gray')
            plt.title("Cloth Edges")

            # Draw contours on a copy of the cloth image
            if 'contours' in cloth_data and cloth_data['contours']:
                contour_img = cloth_image_rgb.copy()
                cv2.drawContours(contour_img, cloth_data['contours'], -1, (0, 255, 0), 2)
                plt.subplot(133)
                plt.imshow(contour_img)
                plt.title("Cloth Contours")

            plt.tight_layout()
            plt.show()

        # If both pattern and cloth are provided, show optimal placement
        if args.pattern_image and args.cloth_image:
            # Here we would implement the optimal pattern placement algorithm
            # For now, just show the images side by side
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(pattern_image_rgb)
            plt.title("Pattern")

            plt.subplot(132)
            plt.imshow(cloth_image_rgb)
            plt.title("Cloth")

            # Placeholder for optimal placement visualization
            plt.subplot(133)
            plt.imshow(cloth_image_rgb)  # Replace with actual placement visualization
            plt.title("Pattern Placement (Not Implemented)")

            plt.tight_layout()
            plt.show()

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
