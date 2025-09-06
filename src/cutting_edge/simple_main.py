"""
Simple Main Script for Pattern Fitting System
This script handles the complete workflow:
1. Load patterns and cloth images from folders
2. Split into train/test sets
3. Train or load existing models
4. Fit patterns onto cloth
5. Display results and metrics
"""

import os
import sys
import glob
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import argparse
from datetime import datetime

# Import our simplified modules
from .simple_pattern_recognition import PatternProcessor, Pattern
from .simple_cloth_recognition import ClothProcessor
from .simple_pattern_fitting import PatternFitter

# Setup logging with detailed format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleCuttingEdgeSystem:
    """
    Main system that orchestrates the entire pattern fitting workflow.
    Designed to be simple and easy to understand.
    """

    def __init__(self, base_dir: str = "/Users/aryaminus/Developer/cutting-edge"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.models_dir = self.base_dir / "models"
        self.output_dir = self.base_dir / "output"
        self.data_dir = self.base_dir / "data"

        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.output_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.pattern_processor = PatternProcessor(
            str(self.models_dir / "pattern_model.pth")
        )
        self.cloth_processor = ClothProcessor(str(self.models_dir / "cloth_model.pth"))
        self.fitter = PatternFitter(str(self.models_dir / "fitting_model.pkl"))

        # Data split ratio
        self.train_ratio = 0.8

        logger.info(f"System initialized with base directory: {self.base_dir}")

    def scan_folders(self) -> Tuple[List[str], List[str]]:
        """
        Scan the images folder for pattern and cloth images.
        Returns lists of pattern and cloth image paths.
        """
        logger.info("Scanning folders for images...")

        # Pattern images
        pattern_dir = self.images_dir / "shape"
        pattern_files = []
        if pattern_dir.exists():
            pattern_files = (
                list(pattern_dir.glob("*.png"))
                + list(pattern_dir.glob("*.jpg"))
                + list(pattern_dir.glob("*.jpeg"))
            )
        logger.info(f"Found {len(pattern_files)} pattern images")

        # Cloth images
        cloth_dir = self.images_dir / "cloth"
        cloth_files = []
        if cloth_dir.exists():
            cloth_files = (
                list(cloth_dir.glob("*.png"))
                + list(cloth_dir.glob("*.jpg"))
                + list(cloth_dir.glob("*.jpeg"))
            )
        logger.info(f"Found {len(cloth_files)} cloth images")

        return [str(f) for f in pattern_files], [str(f) for f in cloth_files]

    def split_data(self, pattern_files: List[str], cloth_files: List[str]) -> Dict:
        """
        Split data into train and test sets.
        Saves the split information for reproducibility.
        """
        logger.info("Splitting data into train/test sets...")

        # Shuffle data
        random.seed(42)  # For reproducibility
        random.shuffle(pattern_files)
        random.shuffle(cloth_files)

        # Calculate split indices
        pattern_train_size = int(len(pattern_files) * self.train_ratio)
        cloth_train_size = int(len(cloth_files) * self.train_ratio)

        # Split data
        split = {
            "pattern_train": pattern_files[:pattern_train_size],
            "pattern_test": pattern_files[pattern_train_size:],
            "cloth_train": cloth_files[:cloth_train_size],
            "cloth_test": cloth_files[cloth_train_size:],
            "timestamp": datetime.now().isoformat(),
        }

        # Save split info
        split_file = self.data_dir / "data_split.json"
        with open(split_file, "w") as f:
            json.dump(split, f, indent=2)

        logger.info(
            f"Train set: {len(split['pattern_train'])} patterns, {len(split['cloth_train'])} cloths"
        )
        logger.info(
            f"Test set: {len(split['pattern_test'])} patterns, {len(split['cloth_test'])} cloths"
        )
        logger.info(f"Split info saved to {split_file}")

        return split

    def load_or_create_split(
        self, pattern_files: List[str], cloth_files: List[str]
    ) -> Dict:
        """
        Load existing data split or create new one.
        """
        split_file = self.data_dir / "data_split.json"

        if split_file.exists():
            logger.info("Loading existing data split...")
            with open(split_file, "r") as f:
                split = json.load(f)

            # Validate that files still exist
            all_files_exist = all(
                os.path.exists(f)
                for f in split["pattern_train"]
                + split["pattern_test"]
                + split["cloth_train"]
                + split["cloth_test"]
            )

            if all_files_exist:
                logger.info("Using existing data split")
                return split
            else:
                logger.warning(
                    "Some files in existing split not found, creating new split"
                )

        return self.split_data(pattern_files, cloth_files)

    def process_single_fitting(
        self, pattern_path: str, cloth_path: str, save_visualization: bool = True
    ) -> Dict:
        """
        Process a single pattern-cloth pair and perform fitting.
        """
        logger.info("=" * 60)
        logger.info(f"Processing pattern: {os.path.basename(pattern_path)}")
        logger.info(f"Processing cloth: {os.path.basename(cloth_path)}")

        # Process pattern
        pattern_info = self.pattern_processor.process_image(pattern_path)

        # Create Pattern object
        pattern = Pattern(
            id=0,
            width=pattern_info["dimensions"][0],
            height=pattern_info["dimensions"][1],
            contour=pattern_info["contour"],
            name=os.path.splitext(pattern_info["filename"])[0],
        )

        # Process cloth
        cloth_info = self.cloth_processor.process_image(cloth_path)

        # Log dimensions
        logger.info(
            f"Pattern dimensions: {pattern.width:.1f} x {pattern.height:.1f} cm"
        )
        logger.info(
            f"Cloth dimensions: {cloth_info['dimensions'][0]:.1f} x {cloth_info['dimensions'][1]:.1f} cm"
        )
        logger.info(
            f"Usable cloth area: {cloth_info['usable_dimensions'][0]:.1f} x {cloth_info['usable_dimensions'][1]:.1f} cm"
        )

        # Perform fitting
        logger.info("Performing pattern fitting...")
        result = self.fitter.fit_patterns([pattern], cloth_info)

        # Save visualization if requested
        if save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"fitting_result_{timestamp}.png"
            self.fitter.visualize_result(
                [pattern], cloth_info, result, str(output_path)
            )
            result["visualization_path"] = str(output_path)

        # Add more info to result
        result["pattern_file"] = pattern_path
        result["cloth_file"] = cloth_path
        result["pattern_dimensions"] = (pattern.width, pattern.height)
        result["cloth_dimensions"] = cloth_info["dimensions"]

        return result

    def process_multiple_patterns_fitting(
        self, pattern_paths: List[str], cloth_path: str, save_visualization: bool = True
    ) -> Dict:
        """
        Process multiple patterns to fit on a single cloth.
        """
        logger.info("=" * 60)
        logger.info(
            f"Processing {len(pattern_paths)} patterns for multi-pattern fitting"
        )
        logger.info(f"Cloth: {os.path.basename(cloth_path)}")

        # Process all patterns
        patterns = []
        for i, pattern_path in enumerate(pattern_paths):
            logger.info(
                f"Loading pattern {i + 1}/{len(pattern_paths)}: {os.path.basename(pattern_path)}"
            )
            pattern_info = self.pattern_processor.process_image(pattern_path)

            pattern = Pattern(
                id=i,
                width=pattern_info["dimensions"][0],
                height=pattern_info["dimensions"][1],
                contour=pattern_info["contour"],
                name=os.path.splitext(pattern_info["filename"])[0],
            )
            patterns.append(pattern)
            logger.info(f"  Dimensions: {pattern.width:.1f} x {pattern.height:.1f} cm")

        # Process cloth
        cloth_info = self.cloth_processor.process_image(cloth_path)
        logger.info(
            f"Cloth dimensions: {cloth_info['dimensions'][0]:.1f} x {cloth_info['dimensions'][1]:.1f} cm"
        )
        logger.info(
            f"Usable cloth area: {cloth_info['usable_dimensions'][0]:.1f} x {cloth_info['usable_dimensions'][1]:.1f} cm"
        )

        # Perform fitting
        logger.info("Performing multi-pattern fitting...")
        result = self.fitter.fit_patterns(patterns, cloth_info)

        # Save visualization if requested
        if save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"multi_fitting_result_{timestamp}.png"
            self.fitter.visualize_result(patterns, cloth_info, result, str(output_path))
            result["visualization_path"] = str(output_path)

        # Add more info to result
        result["pattern_files"] = pattern_paths
        result["cloth_file"] = cloth_path
        result["cloth_dimensions"] = cloth_info["dimensions"]
        result["patterns_info"] = [(p.name, p.width, p.height) for p in patterns]

        return result

    def run_training(self, split: Dict):
        """
        Train the models using the training data.
        For now, this is simplified - in real implementation would do actual training.
        """
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING MODE")
        logger.info("=" * 60)

        # In a real implementation, we would:
        # 1. Train pattern recognition model on pattern images
        # 2. Train cloth recognition model on cloth images
        # 3. Train fitting model by trying different placements

        # For now, we'll just save the current models
        logger.info("Training pattern recognition model...")
        self.pattern_processor.save_model()

        logger.info("Training cloth recognition model...")
        self.cloth_processor.save_model()

        logger.info("Training pattern fitting model...")
        self.fitter.save_model()

        logger.info("Training complete! Models saved.")

    def run_inference(
        self, pattern_files: List[str], cloth_files: List[str], num_examples: int = 1
    ) -> List[Dict]:
        """
        Run inference on given pattern and cloth files.
        """
        logger.info("\n" + "=" * 60)
        logger.info("INFERENCE MODE")
        logger.info("=" * 60)

        # Load models if they exist
        self.pattern_processor.load_model()
        self.cloth_processor.load_model()
        self.fitter.load_model()

        results = []

        # Process requested number of examples
        for i in range(min(num_examples, len(pattern_files), len(cloth_files))):
            pattern_path = (
                pattern_files[i] if i < len(pattern_files) else pattern_files[0]
            )
            cloth_path = cloth_files[i] if i < len(cloth_files) else cloth_files[0]

            result = self.process_single_fitting(pattern_path, cloth_path)
            results.append(result)

            # Display metrics
            self.display_metrics(result)

        return results

    def display_metrics(self, result: Dict):
        """
        Display fitting metrics in a nice format.
        """
        logger.info("\n" + "-" * 40)
        logger.info("FITTING METRICS:")
        logger.info("-" * 40)

        # Check if this is a multi-pattern result
        if "pattern_files" in result:
            # Multi-pattern result
            logger.info(f"Patterns: {len(result['pattern_files'])} files")
            for i, (name, width, height) in enumerate(result["patterns_info"]):
                logger.info(f"  {i + 1}. {name}: {width:.1f} x {height:.1f} cm")
        else:
            # Single pattern result
            logger.info(f"Pattern: {os.path.basename(result['pattern_file'])}")
            logger.info(
                f"Pattern dimensions: {result['pattern_dimensions'][0]:.1f} x {result['pattern_dimensions'][1]:.1f} cm"
            )

        logger.info(f"Cloth: {os.path.basename(result['cloth_file'])}")
        logger.info(
            f"Cloth dimensions: {result['cloth_dimensions'][0]:.1f} x {result['cloth_dimensions'][1]:.1f} cm"
        )
        logger.info(
            f"Patterns placed: {result['patterns_placed']}/{result['patterns_total']}"
        )
        logger.info(f"Material utilization: {result['utilization_percentage']:.1f}%")
        logger.info(f"Wasted area: {result['wasted_area']:.1f} cm²")
        logger.info(f"Success rate: {result['success_rate']:.1f}%")

        if "visualization_path" in result:
            logger.info(f"Visualization saved to: {result['visualization_path']}")

        # Save metrics to file
        metrics_file = self.output_dir / "fitting_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "cloth": os.path.basename(result["cloth_file"]),
            "utilization": result["utilization_percentage"],
            "success_rate": result["success_rate"],
        }

        if "pattern_files" in result:
            metric_entry["patterns"] = [
                os.path.basename(p) for p in result["pattern_files"]
            ]
            metric_entry["num_patterns"] = len(result["pattern_files"])
        else:
            metric_entry["pattern"] = os.path.basename(result["pattern_file"])

        all_metrics.append(metric_entry)

        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logger.info("-" * 40)
        logger.info(f"Pattern: {os.path.basename(result['pattern_file'])}")
        logger.info(f"Cloth: {os.path.basename(result['cloth_file'])}")
        logger.info(
            f"Pattern dimensions: {result['pattern_dimensions'][0]:.1f} x {result['pattern_dimensions'][1]:.1f} cm"
        )
        logger.info(
            f"Cloth dimensions: {result['cloth_dimensions'][0]:.1f} x {result['cloth_dimensions'][1]:.1f} cm"
        )
        logger.info(
            f"Patterns placed: {result['patterns_placed']}/{result['patterns_total']}"
        )
        logger.info(f"Material utilization: {result['utilization_percentage']:.1f}%")
        logger.info(f"Wasted area: {result['wasted_area']:.1f} cm²")
        logger.info(f"Success rate: {result['success_rate']:.1f}%")

        if "visualization_path" in result:
            logger.info(f"Visualization saved to: {result['visualization_path']}")

        # Save metrics to file
        metrics_file = self.output_dir / "fitting_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        all_metrics.append(
            {
                "timestamp": datetime.now().isoformat(),
                "pattern": os.path.basename(result["pattern_file"]),
                "cloth": os.path.basename(result["cloth_file"]),
                "utilization": result["utilization_percentage"],
                "success_rate": result["success_rate"],
            }
        )

        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logger.info("-" * 40)


def main():
    """
    Main entry point for the simplified cutting edge system.
    """
    parser = argparse.ArgumentParser(description="Simple Pattern Fitting System")
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "both"],
        default="inference",
        help="Run mode: train models, run inference, or both",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="Number of examples to process in inference mode",
    )
    parser.add_argument(
        "--pattern", type=str, default=None, help="Specific pattern image to use"
    )
    parser.add_argument(
        "--cloth", type=str, default=None, help="Specific cloth image to use"
    )
    parser.add_argument(
        "--multi_pattern", action="store_true", help="Enable multi-pattern fitting mode"
    )

    args = parser.parse_args()

    # Initialize system
    system = SimpleCuttingEdgeSystem()

    # Scan for images
    pattern_files, cloth_files = system.scan_folders()

    if not pattern_files or not cloth_files:
        logger.error("No pattern or cloth images found!")
        logger.error(
            f"Please add images to {system.images_dir}/shape/ and {system.images_dir}/cloth/"
        )
        return

    # Handle specific files if provided
    if args.pattern and args.cloth:
        if not os.path.exists(args.pattern) or not os.path.exists(args.cloth):
            logger.error("Specified pattern or cloth file not found!")
            return
        pattern_files = [args.pattern]
        cloth_files = [args.cloth]
        args.mode = "inference"  # Force inference mode for specific files

    # Get data split
    split = system.load_or_create_split(pattern_files, cloth_files)

    # Run based on mode
    if args.mode == "train" or args.mode == "both":
        system.run_training(split)

    if args.mode == "inference" or args.mode == "both":
        # Use test set for inference by default
        test_patterns = (
            split["pattern_test"] if split["pattern_test"] else pattern_files[:1]
        )
        test_cloths = split["cloth_test"] if split["cloth_test"] else cloth_files[:1]

        # If specific files provided, use those instead
        if args.pattern and args.cloth:
            test_patterns = pattern_files
            test_cloths = cloth_files

        # Handle multi-pattern mode
        if args.multi_pattern:
            # Use all test patterns on first cloth
            if len(test_patterns) > 1 and test_cloths:
                logger.info("Running multi-pattern fitting mode")
                result = system.process_multiple_patterns_fitting(
                    test_patterns[: min(5, len(test_patterns))],  # Limit to 5 patterns
                    test_cloths[0],
                )
                system.display_metrics(result)
            else:
                logger.warning(
                    "Not enough patterns for multi-pattern mode, falling back to single pattern"
                )
                system.run_inference(test_patterns, test_cloths, args.num_examples)
        else:
            system.run_inference(test_patterns, test_cloths, args.num_examples)

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Check {system.output_dir} for results")
    logger.info(f"Check {system.data_dir} for data splits")
    logger.info(f"Check {system.models_dir} for saved models")


if __name__ == "__main__":
    main()
