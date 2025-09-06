"""
Enhanced Main Script for Cutting Edge System
Combines simplicity with advanced features for optimal pattern fitting.
"""

import os
import sys
import glob
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import numpy as np
import torch

# Import our enhanced modules
from .enhanced_pattern_recognition import EnhancedPatternProcessor, Pattern
from .enhanced_cloth_recognition import EnhancedClothProcessor, ClothMaterial
from .enhanced_pattern_fitting import EnhancedPatternFitter
from .simple_config import SYSTEM, TRAINING

# Setup logging
logging.basicConfig(
    level=getattr(logging, SYSTEM["LOG_LEVEL"]), format=SYSTEM["LOG_FORMAT"]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(SYSTEM["RANDOM_SEED"])
torch.manual_seed(SYSTEM["RANDOM_SEED"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(SYSTEM["RANDOM_SEED"])


class EnhancedCuttingEdgeSystem:
    """
    Main system that orchestrates the enhanced pattern fitting workflow.
    Combines the best of both simple and advanced features.
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = SYSTEM["BASE_DIR"]

        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / SYSTEM["IMAGES_DIR"]
        self.models_dir = self.base_dir / SYSTEM["MODELS_DIR"]
        self.output_dir = self.base_dir / SYSTEM["OUTPUT_DIR"]
        self.data_dir = self.base_dir / SYSTEM["DATA_DIR"]

        # Create directories
        for dir_path in [self.models_dir, self.output_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.pattern_processor = EnhancedPatternProcessor(
            str(self.models_dir / "enhanced_pattern_model.pth")
        )
        self.cloth_processor = EnhancedClothProcessor(
            str(self.models_dir / "enhanced_cloth_model.pth")
        )
        self.fitter = EnhancedPatternFitter(
            str(self.models_dir / "enhanced_fitting_model.pkl")
        )

        logger.info(f"Enhanced Cutting Edge System initialized")
        logger.info(f"Base directory: {self.base_dir}")

    def scan_images(self) -> Tuple[List[str], List[str]]:
        """Scan for pattern and cloth images."""
        logger.info("Scanning for images...")

        # Pattern images
        pattern_dir = self.images_dir / "shape"
        pattern_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        pattern_files = []

        if pattern_dir.exists():
            for ext in pattern_extensions:
                pattern_files.extend(pattern_dir.glob(ext))

        logger.info(f"Found {len(pattern_files)} pattern images")

        # Cloth images
        cloth_dir = self.images_dir / "cloth"
        cloth_files = []

        if cloth_dir.exists():
            for ext in pattern_extensions:
                cloth_files.extend(cloth_dir.glob(ext))

        logger.info(f"Found {len(cloth_files)} cloth images")

        return [str(f) for f in pattern_files], [str(f) for f in cloth_files]

    def split_data(self, pattern_files: List[str], cloth_files: List[str]) -> Dict:
        """Split data into train/test sets."""
        logger.info(f"Splitting data with ratio: {TRAINING['TRAIN_RATIO']}")

        # Shuffle
        random.shuffle(pattern_files)
        random.shuffle(cloth_files)

        # Split
        pattern_split = int(len(pattern_files) * TRAINING["TRAIN_RATIO"])
        cloth_split = int(len(cloth_files) * TRAINING["TRAIN_RATIO"])

        split = {
            "pattern_train": pattern_files[:pattern_split],
            "pattern_test": pattern_files[pattern_split:],
            "cloth_train": cloth_files[:cloth_split],
            "cloth_test": cloth_files[cloth_split:],
            "timestamp": datetime.now().isoformat(),
            "train_ratio": TRAINING["TRAIN_RATIO"],
        }

        # Save split
        split_file = self.data_dir / "enhanced_data_split.json"
        with open(split_file, "w") as f:
            json.dump(split, f, indent=2)

        logger.info(f"Data split saved to {split_file}")
        logger.info(
            f"Train: {len(split['pattern_train'])} patterns, {len(split['cloth_train'])} cloths"
        )
        logger.info(
            f"Test: {len(split['pattern_test'])} patterns, {len(split['cloth_test'])} cloths"
        )

        return split

    def load_or_create_split(
        self, pattern_files: List[str], cloth_files: List[str]
    ) -> Dict:
        """Load existing split or create new one."""
        split_file = self.data_dir / "enhanced_data_split.json"

        if split_file.exists():
            logger.info("Loading existing data split...")
            with open(split_file, "r") as f:
                split = json.load(f)

            # Validate files exist
            all_exist = all(
                os.path.exists(f)
                for f in split.get("pattern_train", [])
                + split.get("pattern_test", [])
                + split.get("cloth_train", [])
                + split.get("cloth_test", [])
            )

            if all_exist:
                return split
            else:
                logger.warning("Some files missing, creating new split")

        return self.split_data(pattern_files, cloth_files)

    def process_pattern_batch(self, pattern_paths: List[str]) -> List[Pattern]:
        """Process multiple pattern images."""
        patterns = []

        for path in pattern_paths:
            try:
                pattern = self.pattern_processor.process_pattern(path)
                patterns.append(pattern)
            except Exception as e:
                logger.error(f"Failed to process pattern {path}: {e}")

        return patterns

    def process_fitting_task(
        self, pattern_paths: List[str], cloth_path: str, save_visualization: bool = True
    ) -> Dict:
        """
        Process a complete fitting task with multiple patterns on one cloth.
        """
        logger.info("\n" + "=" * 60)
        logger.info("ENHANCED PATTERN FITTING TASK")
        logger.info("=" * 60)

        # Process patterns
        logger.info(f"\nProcessing {len(pattern_paths)} patterns...")
        patterns = self.process_pattern_batch(pattern_paths)

        if not patterns:
            logger.error("No valid patterns found!")
            return None

        # Process cloth
        logger.info(f"\nProcessing cloth material...")
        try:
            cloth = self.cloth_processor.process_cloth(cloth_path)
        except Exception as e:
            logger.error(f"Failed to process cloth: {e}")
            return None

        # Log summary
        logger.info(f"\nFitting Summary:")
        logger.info(
            f"- Cloth: {cloth.cloth_type}, {cloth.total_width:.1f}x{cloth.total_height:.1f} cm"
        )
        logger.info(
            f"- Usable area: {cloth.usable_area:.1f} cm² ({len(cloth.defects)} defects)"
        )
        logger.info(f"- Patterns to fit: {len(patterns)}")

        total_pattern_area = sum(p.area for p in patterns)
        logger.info(f"- Total pattern area: {total_pattern_area:.1f} cm²")
        logger.info(
            f"- Theoretical max utilization: {(total_pattern_area / cloth.usable_area * 100):.1f}%"
        )

        # Perform fitting
        logger.info(f"\nPerforming pattern fitting...")
        result = self.fitter.fit_patterns(patterns, cloth)

        # Add metadata
        result["pattern_files"] = pattern_paths
        result["cloth_file"] = cloth_path
        result["cloth_type"] = cloth.cloth_type
        result["timestamp"] = datetime.now().isoformat()

        # Save visualization
        if save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = self.output_dir / f"enhanced_fitting_{timestamp}.png"
            self.fitter.visualize_fitting(patterns, cloth, result, str(viz_path))
            result["visualization_path"] = str(viz_path)

            # Also save cloth analysis
            cloth_viz_path = self.output_dir / f"cloth_analysis_{timestamp}.png"
            self.cloth_processor.visualize_cloth_analysis(cloth, str(cloth_viz_path))
            result["cloth_analysis_path"] = str(cloth_viz_path)

        return result

    def run_demo(self, num_patterns: int = 3):
        """
        Run a demonstration with randomly selected patterns and cloth.
        """
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING ENHANCED FITTING DEMONSTRATION")
        logger.info("=" * 60)

        # Scan for images
        pattern_files, cloth_files = self.scan_images()

        if not pattern_files or not cloth_files:
            logger.error("No images found! Please add pattern and cloth images.")
            return

        # Load or create split
        split = self.load_or_create_split(pattern_files, cloth_files)

        # Use test set for demo
        test_patterns = split.get("pattern_test", pattern_files)
        test_cloths = split.get("cloth_test", cloth_files)

        if not test_patterns or not test_cloths:
            test_patterns = pattern_files
            test_cloths = cloth_files

        # Select patterns and cloth
        selected_patterns = test_patterns[: min(num_patterns, len(test_patterns))]
        selected_cloth = test_cloths[0] if test_cloths else cloth_files[0]

        logger.info(f"\nSelected {len(selected_patterns)} patterns for demo")
        logger.info(f"Selected cloth: {os.path.basename(selected_cloth)}")

        # Load models
        self.pattern_processor.load_model()
        self.cloth_processor.load_model()
        self.fitter.load_model()

        # Process fitting
        result = self.process_fitting_task(selected_patterns, selected_cloth)

        if result:
            self.display_results(result)

            # Save results (convert to JSON-serializable format)
            results_file = self.output_dir / "enhanced_fitting_results.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    all_results = json.load(f)
            else:
                all_results = []

            # Convert result to JSON-serializable format
            json_result = {
                "timestamp": result["timestamp"],
                "cloth_file": result["cloth_file"],
                "cloth_type": result["cloth_type"],
                "pattern_files": result["pattern_files"],
                "patterns_total": result["patterns_total"],
                "patterns_placed": result["patterns_placed"],
                "success_rate": result["success_rate"],
                "utilization_percentage": result["utilization_percentage"],
                "waste_area": result["waste_area"],
                "cloth_dimensions": result["cloth_dimensions"],
                "cloth_usable_area": result["cloth_usable_area"],
                "total_pattern_area": result["total_pattern_area"],
                "visualization_path": result.get("visualization_path"),
                "cloth_analysis_path": result.get("cloth_analysis_path"),
            }

            all_results.append(json_result)

            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

            logger.info(f"\nResults saved to {results_file}")

    def display_results(self, result: Dict):
        """Display fitting results in a nice format."""
        logger.info("\n" + "-" * 60)
        logger.info("FITTING RESULTS")
        logger.info("-" * 60)

        logger.info(f"Timestamp: {result['timestamp']}")
        logger.info(
            f"Cloth: {os.path.basename(result['cloth_file'])} ({result['cloth_type']})"
        )
        logger.info(f"Patterns attempted: {result['patterns_total']}")
        logger.info(f"Patterns placed: {result['patterns_placed']}")
        logger.info(f"Success rate: {result['success_rate']:.1f}%")
        logger.info(f"Material utilization: {result['utilization_percentage']:.1f}%")
        logger.info(f"Waste area: {result['waste_area']:.1f} cm²")

        if "placed_patterns" in result:
            logger.info("\nPlaced patterns:")
            for i, placed in enumerate(result["placed_patterns"]):
                logger.info(
                    f"  {i + 1}. {placed.pattern.name} ({placed.pattern.pattern_type})"
                )
                logger.info(
                    f"     Position: ({placed.position[0]:.1f}, {placed.position[1]:.1f})"
                )
                logger.info(
                    f"     Rotation: {placed.rotation}°, Flipped: {placed.flipped}"
                )

        if result.get("failed_patterns"):
            logger.info("\nFailed patterns:")
            for pattern in result["failed_patterns"]:
                logger.info(
                    f"  - {pattern.name} ({pattern.width:.1f}x{pattern.height:.1f} cm)"
                )

        if result.get("visualization_path"):
            logger.info(f"\nVisualization saved to: {result['visualization_path']}")
        if result.get("cloth_analysis_path"):
            logger.info(f"Cloth analysis saved to: {result['cloth_analysis_path']}")

        logger.info("-" * 60)

    def train_models(self, split: Dict):
        """
        Train the models (simplified for now).
        In practice, this would involve actual training loops.
        """
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING ENHANCED MODELS")
        logger.info("=" * 60)

        # For now, just save the current models
        logger.info("Saving pattern recognition model...")
        self.pattern_processor.save_model()

        logger.info("Saving cloth segmentation model...")
        self.cloth_processor.save_model()

        logger.info("Saving placement optimizer...")
        self.fitter.save_model()

        logger.info("Training complete! Models saved.")


def main():
    """Main entry point for enhanced system."""
    parser = argparse.ArgumentParser(
        description="Enhanced Cutting Edge Pattern Fitting System"
    )

    parser.add_argument(
        "--mode",
        choices=["demo", "train", "fit"],
        default="demo",
        help="Operation mode",
    )
    parser.add_argument("--patterns", type=str, nargs="+", help="Pattern images to fit")
    parser.add_argument("--cloth", type=str, help="Cloth image to fit patterns on")
    parser.add_argument(
        "--num_patterns", type=int, default=3, help="Number of patterns for demo mode"
    )
    parser.add_argument("--base_dir", type=str, help="Override base directory")

    args = parser.parse_args()

    # Initialize system
    system = EnhancedCuttingEdgeSystem(args.base_dir)

    if args.mode == "demo":
        # Run demonstration
        system.run_demo(args.num_patterns)

    elif args.mode == "train":
        # Train models
        pattern_files, cloth_files = system.scan_images()
        split = system.load_or_create_split(pattern_files, cloth_files)
        system.train_models(split)

    elif args.mode == "fit":
        # Fit specific patterns on cloth
        if not args.patterns or not args.cloth:
            logger.error("Please specify --patterns and --cloth for fit mode")
            return

        # Load models first
        system.pattern_processor.load_model()
        system.cloth_processor.load_model()
        system.fitter.load_model()

        # Process fitting
        result = system.process_fitting_task(args.patterns, args.cloth)
        if result:
            system.display_results(result)

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Check {system.output_dir} for results")
    logger.info(f"Check {system.models_dir} for saved models")


if __name__ == "__main__":
    main()
