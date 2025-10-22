"""
Cutting Edge - Main Application

Handles the complete workflow for pattern fitting:
1. Loading patterns and cloth images
2. Pattern recognition and cloth analysis
3. Optimal pattern fitting on cloth
4. Visualization and reporting

This is the consolidated version that combines the best aspects of all previous implementations.
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .cloth_recognition_module import ClothMaterial, ClothRecognitionModule
from .config import SYSTEM, TRAINING
from .pattern_fitting_module import PatternFittingModule

# Import our modules
from .pattern_recognition_module import Pattern, PatternRecognitionModule

# Setup logging
logging.basicConfig(
    level=getattr(logging, SYSTEM["LOG_LEVEL"]), format=SYSTEM["LOG_FORMAT"]
)
logger = logging.getLogger(__name__)


class CuttingEdgeSystem:
    """
    Main system that orchestrates the entire pattern fitting workflow.
    """

    def __init__(self, base_dir: str = None):
        """Initialize the cutting edge system."""
        # Set base directory
        if base_dir is None:
            base_dir = SYSTEM["BASE_DIR"]
        self.base_dir = Path(base_dir)

        # Setup directories
        self.images_dir = self.base_dir / SYSTEM["IMAGES_DIR"]
        self.models_dir = self.base_dir / SYSTEM["MODELS_DIR"]
        self.output_dir = self.base_dir / SYSTEM["OUTPUT_DIR"]
        self.data_dir = self.base_dir / SYSTEM["DATA_DIR"]

        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.output_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize modules
        self.pattern_module = PatternRecognitionModule()
        self.cloth_module = ClothRecognitionModule()
        self.fitting_module = PatternFittingModule()

        logger.info("Cutting Edge System initialized")
        logger.info(f"Base directory: {self.base_dir}")

    def scan_images(self) -> Tuple[List[str], List[str]]:
        """
        Scan for pattern and cloth images in the images directory.
        """
        logger.info("Scanning for images...")

        pattern_dir = self.images_dir / SYSTEM["PATTERN_DIR_NAME"]
        cloth_dir = self.images_dir / SYSTEM["CLOTH_DIR_NAME"]

        # Image extensions to look for
        extensions = SYSTEM["IMAGE_EXTENSIONS"]

        # Find pattern images
        pattern_files = []
        if pattern_dir.exists():
            for ext in extensions:
                pattern_files.extend(list(pattern_dir.glob(f"**/*.{ext}")))

        # Find cloth images
        cloth_files = []
        if cloth_dir.exists():
            for ext in extensions:
                cloth_files.extend(list(cloth_dir.glob(f"**/*.{ext}")))

        logger.info(
            f"Found {len(pattern_files)} pattern images and {len(cloth_files)} cloth images"
        )

        return [str(f) for f in pattern_files], [str(f) for f in cloth_files]

    def split_data(self, pattern_files: List[str], cloth_files: List[str]) -> Dict:
        """
        Split the data into training and testing sets.
        """
        # Randomize order
        random.shuffle(pattern_files)
        random.shuffle(cloth_files)

        # Calculate split indices
        pattern_split = int(len(pattern_files) * TRAINING["TRAIN_RATIO"])
        cloth_split = int(len(cloth_files) * TRAINING["TRAIN_RATIO"])

        # Create split data
        split_data = {
            "pattern_train": pattern_files[:pattern_split],
            "pattern_test": pattern_files[pattern_split:],
            "cloth_train": cloth_files[:cloth_split],
            "cloth_test": cloth_files[cloth_split:],
            "timestamp": datetime.now().isoformat(),
        }

        # Save split info
        split_file = self.data_dir / "data_split.json"
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)

        logger.info(
            f"Data split created: {pattern_split}/{len(pattern_files)} patterns, "
            f"{cloth_split}/{len(cloth_files)} cloths for training"
        )

        return split_data

    def load_or_create_split(
        self, pattern_files: List[str], cloth_files: List[str]
    ) -> Dict:
        """
        Load existing data split or create a new one.
        """
        split_file = self.data_dir / "data_split.json"

        if split_file.exists():
            logger.info("Loading existing data split...")
            with open(split_file, "r") as f:
                split_data = json.load(f)

            # Verify files still exist
            all_files_exist = all(
                os.path.exists(f)
                for f in split_data.get("pattern_train", [])
                + split_data.get("pattern_test", [])
                + split_data.get("cloth_train", [])
                + split_data.get("cloth_test", [])
            )

            if all_files_exist:
                logger.info("Using existing data split")
                return split_data

            logger.warning(
                "Some files in existing split no longer exist, creating new split"
            )

        # Create new split
        return self.split_data(pattern_files, cloth_files)

    def process_patterns(self, pattern_paths: List[str]) -> List[Pattern]:
        """
        Process multiple pattern images.
        """
        patterns = []
        for path in pattern_paths:
            try:
                pattern = self.pattern_module.process_image(path)
                patterns.append(pattern)
            except Exception as e:
                logger.error(f"Failed to process pattern {path}: {e}")

        return patterns

    def process_cloth(self, cloth_path: str) -> ClothMaterial:
        """
        Process a cloth image.
        """
        try:
            cloth = self.cloth_module.process_image(cloth_path)
            return cloth
        except Exception as e:
            logger.error(f"Failed to process cloth {cloth_path}: {e}")
            raise

    def process_fitting(self, patterns: List[Pattern], cloth: ClothMaterial) -> Dict:
        """
        Perform pattern fitting and generate visualization.
        """
        # Fit patterns on cloth
        logger.info(f"Fitting {len(patterns)} patterns onto cloth...")
        result = self.fitting_module.fit_patterns(patterns, cloth)

        # Generate visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = self.output_dir / f"fitting_result_{timestamp}.png"
        self.fitting_module.visualize(result, patterns, cloth, str(viz_path))

        # Add visualization path to result
        result["visualization_path"] = str(viz_path)

        # Also visualize cloth analysis
        cloth_viz_path = self.output_dir / f"cloth_analysis_{timestamp}.png"
        self.cloth_module.visualize(cloth, str(cloth_viz_path))
        result["cloth_analysis_path"] = str(cloth_viz_path)

        return result

    def run_training(self, split_data: Dict) -> Dict:
        """
        Train all models using the training data.
        """
        logger.info("=== TRAINING MODE ===")

        # Load training data
        pattern_train = split_data["pattern_train"]
        cloth_train = split_data["cloth_train"]

        if not pattern_train or not cloth_train:
            logger.error("Insufficient training data")
            return {"status": "failed", "message": "Insufficient training data"}

        # Train pattern recognition module
        logger.info("Training pattern recognition model...")
        pattern_result = self.pattern_module.train(
            pattern_train, epochs=TRAINING["EPOCHS"]
        )

        # Train cloth recognition module
        logger.info("Training cloth recognition model...")
        cloth_result = self.cloth_module.train(cloth_train, epochs=TRAINING["EPOCHS"])

        # Train pattern fitting module
        # (would need pattern-cloth pairs with ground truth for real training)
        logger.info("Training pattern fitting model...")
        fitting_result = self.fitting_module.train([], epochs=TRAINING["EPOCHS"])

        # Save all models
        self.pattern_module.save_model()
        self.cloth_module.save_model()
        self.fitting_module.save_model()

        logger.info("Training complete. Models saved.")

        return {
            "status": "success",
            "pattern_training": pattern_result,
            "cloth_training": cloth_result,
            "fitting_training": fitting_result,
        }

    def run_fitting_task(self, pattern_paths: List[str], cloth_path: str) -> Dict:
        """
        Run a complete fitting task with specified patterns and cloth.
        """
        # Load models if they exist
        self.pattern_module.load_model()
        self.cloth_module.load_model()
        self.fitting_module.load_model()

        # Process patterns
        logger.info(f"Processing {len(pattern_paths)} patterns...")
        patterns = self.process_patterns(pattern_paths)

        if not patterns:
            logger.error("No valid patterns to process")
            return None

        # Process cloth
        logger.info(f"Processing cloth: {cloth_path}")
        cloth = self.process_cloth(cloth_path)

        # Perform fitting
        result = self.process_fitting(patterns, cloth)

        # Add file paths to result
        result["pattern_paths"] = pattern_paths
        result["cloth_path"] = cloth_path

        # Display metrics
        self.display_metrics(result)

        return result

    def run_demo(self, num_patterns: int = 3):
        """
        Run a demonstration with automatically selected patterns and cloth.
        """
        logger.info("=== DEMO MODE ===")

        # Scan for images
        pattern_files, cloth_files = self.scan_images()

        if not pattern_files or not cloth_files:
            logger.error("No images found. Please add pattern and cloth images.")
            return

        # Get data split (or create one)
        split_data = self.load_or_create_split(pattern_files, cloth_files)

        # Use test patterns and cloth for demo
        test_patterns = split_data.get("pattern_test", pattern_files[:num_patterns])
        test_cloths = split_data.get("cloth_test", cloth_files[:1])

        # Ensure we have enough patterns and at least one cloth
        if len(test_patterns) < 1 or len(test_cloths) < 1:
            logger.warning("Not enough test data, using all available images")
            test_patterns = pattern_files[:num_patterns] if pattern_files else []
            test_cloths = cloth_files[:1] if cloth_files else []

        if not test_patterns or not test_cloths:
            logger.error("No images available for demo")
            return

        # Select patterns and cloth
        selected_patterns = test_patterns[: min(num_patterns, len(test_patterns))]
        selected_cloth = test_cloths[0]

        logger.info(f"Selected {len(selected_patterns)} patterns and 1 cloth for demo")

        # Run fitting task
        try:
            result = self.run_fitting_task(selected_patterns, selected_cloth)

            # Save result metrics
            self.save_result(result)

        except Exception as e:
            logger.error(f"Demo failed: {e}")

    def display_metrics(self, result: Dict):
        """
        Display the fitting metrics in a readable format.
        """
        logger.info("\n" + "=" * 50)
        logger.info("FITTING RESULTS")
        logger.info("=" * 50)

        logger.info(
            f"Patterns: {result['patterns_placed']}/{result['patterns_total']} placed"
        )
        logger.info(f"Material utilization: {result['utilization_percentage']:.1f}%")
        logger.info(f"Waste area: {result['waste_area']:.1f} cm²")
        logger.info(f"Success rate: {result['success_rate']:.1f}%")

        if "visualization_path" in result:
            logger.info(f"\nVisualization saved to: {result['visualization_path']}")
        if "cloth_analysis_path" in result:
            logger.info(f"Cloth analysis saved to: {result['cloth_analysis_path']}")

        logger.info("=" * 50)

    def save_result(self, result: Dict):
        """
        Save fitting result to a JSON file.
        """
        # Ensure the result can be serialized to JSON
        serializable_result = {
            "timestamp": datetime.now().isoformat(),
            "patterns_total": result["patterns_total"],
            "patterns_placed": result["patterns_placed"],
            "utilization_percentage": result["utilization_percentage"],
            "success_rate": result["success_rate"],
            "total_pattern_area": result["total_pattern_area"],
            "waste_area": result["waste_area"],
            "cloth_dimensions": result["cloth_dimensions"],
            "cloth_usable_area": result["cloth_usable_area"],
            "visualization_path": result.get("visualization_path"),
            "cloth_analysis_path": result.get("cloth_analysis_path"),
        }

        # Paths to patterns and cloth
        if "pattern_paths" in result:
            serializable_result["pattern_paths"] = result["pattern_paths"]
        if "cloth_path" in result:
            serializable_result["cloth_path"] = result["cloth_path"]

        # Save to file
        results_file = self.output_dir / "fitting_results.json"

        # Load existing results if available
        existing_results = []
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    existing_results = json.load(f)
            except Exception:
                pass

        # Add new result
        existing_results.append(serializable_result)

        # Save
        with open(results_file, "w") as f:
            json.dump(existing_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")


def main():
    """
    Main entry point for the Cutting Edge application.
    """
    parser = argparse.ArgumentParser(description="Cutting Edge Pattern Fitting System")

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["demo", "train", "fit"],
        default="demo",
        help="Operation mode: demo, train, or fit",
    )

    # Pattern and cloth selection
    parser.add_argument("--patterns", nargs="+", help="Paths to pattern images")
    parser.add_argument("--cloth", help="Path to cloth image")
    parser.add_argument("--pattern_dir", help="Directory containing pattern images")

    # Demo options
    parser.add_argument(
        "--num_patterns",
        type=int,
        default=3,
        help="Number of patterns to use in demo mode",
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING["EPOCHS"],
        help="Number of training epochs",
    )

    # Other options
    parser.add_argument("--base_dir", help="Base directory for the application")
    parser.add_argument("--output", help="Custom output directory for results")

    # Parse arguments
    args = parser.parse_args()

    # Initialize system
    system = CuttingEdgeSystem(args.base_dir)

    # Handle custom output directory
    if args.output:
        system.output_dir = Path(args.output)
        system.output_dir.mkdir(parents=True, exist_ok=True)

    # Execute selected mode
    if args.mode == "demo":
        # Run demo mode
        system.run_demo(args.num_patterns)

    elif args.mode == "train":
        # Training mode
        pattern_files, cloth_files = system.scan_images()
        split_data = system.load_or_create_split(pattern_files, cloth_files)
        system.run_training(split_data)

    elif args.mode == "fit":
        # Fitting mode
        # Get patterns (either from --patterns or --pattern_dir)
        pattern_paths = []

        if args.patterns:
            pattern_paths = args.patterns
        elif args.pattern_dir:
            pattern_dir = Path(args.pattern_dir)
            if pattern_dir.exists():
                for ext in SYSTEM["IMAGE_EXTENSIONS"]:
                    pattern_paths.extend([str(f) for f in pattern_dir.glob(f"*.{ext}")])

        # Get cloth path
        cloth_path = args.cloth

        # Validate inputs
        if not pattern_paths or not cloth_path:
            logger.error(
                "Please specify patterns (--patterns or --pattern_dir) and cloth (--cloth)"
            )
            return

        if not os.path.exists(cloth_path):
            logger.error(f"Cloth image not found: {cloth_path}")
            return

        # Filter out non-existent pattern files
        pattern_paths = [p for p in pattern_paths if os.path.exists(p)]

        if not pattern_paths:
            logger.error("No valid pattern images specified")
            return

        # Run fitting task
        system.run_fitting_task(pattern_paths, cloth_path)

    logger.info("\nProcessing complete")


if __name__ == "__main__":
    main()
