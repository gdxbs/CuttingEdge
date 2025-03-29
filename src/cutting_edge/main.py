import argparse
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_recognition_module import PatternRecognitionModule
from cutting_edge.pattern_fitting_module import PatternFittingModule

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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cutting Edge Pattern Recognition")

    # Dataset and model paths
    parser.add_argument("--dataset_path", type=str, default=None, 
                        help="Path to GarmentCodeData dataset")
    parser.add_argument("--pattern_model_path", type=str, default="models/pattern_recognition_model.pth", 
                        help="Path to save/load pattern recognition model")
    parser.add_argument("--fitting_model_path", type=str, default="models/pattern_fitting_model.pth", 
                        help="Path to save/load pattern fitting model")

    # Mode selection
    parser.add_argument("--train", action="store_true", 
                        help="Train the model even if it exists")
    parser.add_argument("--train_fitting", action="store_true", 
                        help="Train the pattern fitting model")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--fitting_episodes", type=int, default=100, 
                        help="Number of fitting training episodes")

    # Input files for inference
    parser.add_argument("--pattern_image", type=str, default=None, 
                        help="Path to input pattern image for inference")
    parser.add_argument("--cloth_image", type=str, default=None, 
                        help="Path to input cloth image for inference")
    parser.add_argument("--multi_pattern", action="store_true",
                        help="Process multiple pattern images for fitting")
    parser.add_argument("--pattern_dir", type=str, default=None,
                        help="Directory containing multiple pattern images")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization for pattern fitting")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output visualizations")

    return parser.parse_args()

def process_patterns(pattern_recognizer, pattern_paths):
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

def main():
    """Main execution function"""
    args = parse_arguments()

    try:
        # Create models and output directories if they don't exist
        os.makedirs(os.path.dirname(args.pattern_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.fitting_model_path), exist_ok=True)
        if args.visualize:
            os.makedirs(args.output_dir, exist_ok=True)

        # Initialize pattern recognition module
        logger.info("Initializing pattern recognition module...")
        pattern_recognizer = PatternRecognitionModule(
            dataset_path=args.dataset_path, 
            model_path=args.pattern_model_path if os.path.exists(args.pattern_model_path) else None
        )

        # Train pattern recognition model if requested or needed
        if args.train and args.dataset_path:
            logger.info(f"Training pattern recognition model for {args.epochs} epochs...")
            pattern_recognizer.train(args.epochs)
            pattern_recognizer.save_model(args.pattern_model_path)
            logger.info(f"Model saved to {args.pattern_model_path}")
        elif not args.train:
            pattern_recognizer.check_and_train(args.pattern_model_path, args.epochs)

        # Process pattern images
        pattern_data = None
        patterns_data = []
        
        if args.multi_pattern or args.pattern_dir:
            # Process multiple patterns for fitting
            pattern_dir = args.pattern_dir or os.path.dirname(args.pattern_image)
            if not pattern_dir or not os.path.exists(pattern_dir):
                logger.error(f"Pattern directory not found: {pattern_dir}")
                return
                
            pattern_paths = [os.path.join(pattern_dir, f) for f in os.listdir(pattern_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not pattern_paths:
                logger.error(f"No pattern images found in: {pattern_dir}")
                return
                
            patterns_data = process_patterns(pattern_recognizer, pattern_paths)
            if not patterns_data:
                logger.error("Failed to process any pattern images")
                return
                
        elif args.pattern_image:
            # Process single pattern image
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
            patterns_data = [pattern_data]  # Add to list for fitting

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
            plt.savefig(os.path.join(args.output_dir, "pattern_detection.png"))
            if not args.train_fitting:
                plt.show()

        # Process cloth image if provided
        cloth_data = None
        cloth_image_rgb = None
        if args.cloth_image:
            # Initialize cloth recognition module
            logger.info("Initializing cloth recognition module...")
            cloth_recognizer = ClothRecognitionModule()
            
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
            plt.savefig(os.path.join(args.output_dir, "cloth_detection.png"))
            if not args.train_fitting:
                plt.show()

        # If both pattern and cloth are provided, perform pattern fitting
        if patterns_data and cloth_data:
            logger.info("Initializing pattern fitting module...")
            # Initialize pattern fitting module with model path
            pattern_fitter = PatternFittingModule(
                model_path=args.fitting_model_path if os.path.exists(args.fitting_model_path) else None
            )
            
            # Train fitting model if requested
            if args.train_fitting:
                logger.info(f"Training pattern fitting model for {args.fitting_episodes} episodes...")
                training_result = pattern_fitter.train(
                    cloth_data=cloth_data,
                    patterns_data=patterns_data,
                    num_episodes=args.fitting_episodes
                )
                logger.info(f"Fitting model trained. Best utilization: {training_result['best_utilization']:.4f}")
                
                # Visualize training result
                plt.figure(figsize=(10, 10))
                plt.imshow(training_result['best_state'], cmap='gray')
                plt.title(f"Best Training Result (Utilization: {training_result['best_utilization']:.4f})")
                plt.colorbar(label="Pattern Index")
                plt.savefig(os.path.join(args.output_dir, "training_result.png"))
                plt.show()
            
            # Perform pattern fitting
            logger.info("Fitting patterns to cloth...")
            fitting_result = pattern_fitter.fit_patterns(
                cloth_data=cloth_data,
                patterns_data=patterns_data,
                visualize=args.visualize
            )
            
            # Add patterns to result for visualization
            fitting_result["patterns"] = patterns_data
            
            # Visualize fitting result
            logger.info(f"Pattern fitting complete. Utilization: {fitting_result['utilization']:.4f}")
            pattern_fitter.visualize_result(
                result=fitting_result,
                cloth_image=cloth_image_rgb,
                save_path=os.path.join(args.output_dir, "pattern_fitting_result.png")
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