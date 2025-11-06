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
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cloth_recognition_module import ClothMaterial, ClothRecognitionModule
from .config import SYSTEM, TRAINING
from .pattern_fitting_module import PatternFittingModule
from .pattern_recognition_module import Pattern, PatternRecognitionModule
from .training_optimizer import HeuristicOptimizer

# Setup logging
logging.basicConfig(
    level=getattr(logging, SYSTEM["LOG_LEVEL"]), format=SYSTEM["LOG_FORMAT"]
)
logger = logging.getLogger(__name__)


class CuttingEdgeSystem:
    """
    Main system that orchestrates the entire pattern fitting workflow.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the cutting edge system."""
        # Set base directory
        if base_dir is None:
            base_dir = SYSTEM["BASE_DIR"]
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

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

    def select_suitable_cloth(
        self, pattern_files: List[str], cloth_files: List[str]
    ) -> Optional[str]:
        """
        Select a cloth that can fit the given patterns.
        Returns the first cloth that has enough area for all patterns.
        """
        if not cloth_files:
            return None

        # Calculate total pattern area
        total_pattern_area = 0
        for pattern_file in pattern_files:
            try:
                # Extract dimensions from filename
                filename = os.path.basename(pattern_file)
                width, height = self.pattern_module.extract_dimensions_from_filename(
                    filename
                )
                if width and height:
                    total_pattern_area += width * height
            except Exception:
                continue

        # Find cloth with enough area (at least 3x pattern area for fitting)
        min_cloth_area = total_pattern_area * 3

        for cloth_file in cloth_files:
            try:
                # Extract cloth dimensions
                filename = os.path.basename(cloth_file)
                width, height = self.cloth_module.extract_dimensions_from_filename(
                    filename
                )
                if width and height:
                    cloth_area = width * height
                    if cloth_area >= min_cloth_area:
                        logger.info(
                            f"Selected cloth {filename}: {width}x{height}cm (area: {cloth_area:.0f}cm², patterns need: {total_pattern_area:.0f}cm²)"
                        )
                        return cloth_file
            except Exception:
                continue

        # If no suitable cloth found, return the largest one
        largest_cloth = None
        max_area = 0
        for cloth_file in cloth_files:
            try:
                filename = os.path.basename(cloth_file)
                width, height = self.cloth_module.extract_dimensions_from_filename(
                    filename
                )
                if width and height:
                    cloth_area = width * height
                    if cloth_area > max_area:
                        max_area = cloth_area
                        largest_cloth = cloth_file
            except Exception:
                continue

        if largest_cloth:
            logger.warning(
                f"No ideal cloth found, using largest available: {os.path.basename(largest_cloth)}"
            )
            return largest_cloth

        return None

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
        Split the data into training and testing sets with balanced cloth sizes.
        Uses stratified sampling to ensure both small and large cloths are in train/test.
        """
        # Randomize patterns
        random.shuffle(pattern_files)

        # For cloths, stratify by size to ensure balance
        cloth_sizes = []
        for cloth_file in cloth_files:
            try:
                # Extract dimensions from filename
                filename = os.path.basename(cloth_file)
                width, height = self.cloth_module.extract_dimensions_from_filename(
                    filename
                )
                if width and height:
                    area = width * height
                    cloth_sizes.append(
                        {
                            "file": cloth_file,
                            "area": area,
                            "width": width,
                            "height": height,
                        }
                    )
                else:
                    # If can't extract dimensions, assign to medium category
                    cloth_sizes.append(
                        {"file": cloth_file, "area": 1000, "width": 0, "height": 0}
                    )
            except Exception:
                cloth_sizes.append(
                    {"file": cloth_file, "area": 1000, "width": 0, "height": 0}
                )

        # Sort cloths by area
        cloth_sizes.sort(key=lambda x: x["area"])

        # Split into size categories (small, medium, large)
        n_cloths = len(cloth_sizes)
        small_cloths = cloth_sizes[: n_cloths // 3]  # Smallest 33%
        large_cloths = cloth_sizes[2 * n_cloths // 3 :]  # Largest 33%
        medium_cloths = cloth_sizes[n_cloths // 3 : 2 * n_cloths // 3]  # Middle 34%

        # Calculate split ratios for each category
        train_ratio = TRAINING["TRAIN_RATIO"]

        def split_category(cloths_list):
            """Split a category into train/test maintaining ratio"""
            split_idx = int(len(cloths_list) * train_ratio)
            train = [c["file"] for c in cloths_list[:split_idx]]
            test = [c["file"] for c in cloths_list[split_idx:]]
            return train, test

        # Split each category
        small_train, small_test = split_category(small_cloths)
        medium_train, medium_test = split_category(medium_cloths)
        large_train, large_test = split_category(large_cloths)

        # Combine splits
        cloth_train = small_train + medium_train + large_train
        cloth_test = small_test + medium_test + large_test

        # Shuffle final lists to mix sizes within each split
        random.shuffle(cloth_train)
        random.shuffle(cloth_test)

        # Calculate pattern split
        pattern_split = int(len(pattern_files) * train_ratio)

        # Create split data
        split_data = {
            "pattern_train": pattern_files[:pattern_split],
            "pattern_test": pattern_files[pattern_split:],
            "cloth_train": cloth_train,
            "cloth_test": cloth_test,
            "timestamp": datetime.now().isoformat(),
            "split_info": {
                "total_cloths": n_cloths,
                "train_cloths": len(cloth_train),
                "test_cloths": len(cloth_test),
                "small_train": len(small_train),
                "small_test": len(small_test),
                "medium_train": len(medium_train),
                "medium_test": len(medium_test),
                "large_train": len(large_train),
                "large_test": len(large_test),
            },
        }

        # Save split info
        split_file = self.data_dir / "data_split.json"
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)

        logger.info(
            f"Balanced data split created: {pattern_split}/{len(pattern_files)} patterns, "
            f"{len(cloth_train)}/{len(cloth_files)} cloths for training"
        )
        logger.info(f"  Small cloths: {len(small_train)} train, {len(small_test)} test")
        logger.info(
            f"  Medium cloths: {len(medium_train)} train, {len(medium_test)} test"
        )
        logger.info(f"  Large cloths: {len(large_train)} train, {len(large_test)} test")

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

    def run_fitting_task(
        self, pattern_paths: List[str], cloth_path: str
    ) -> Optional[Dict]:
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

    def _apply_best_config(self, config: Dict):
        """Apply best configuration to fitting module."""
        if "grid_size" in config:
            self.fitting_module.grid_size = config["grid_size"]
        if "rotation_angles" in config:
            self.fitting_module.rotation_angles = config["rotation_angles"]
        if "allow_flipping" in config:
            self.fitting_module.allow_flipping = config["allow_flipping"]
        if "max_attempts" in config:
            self.fitting_module.max_attempts = config["max_attempts"]
        logger.info(f"Applied configuration: {config}")

    def run_training(self, epochs: int = 10, batch_size: int = 15):
        """
        Run hyperparameter optimization for heuristic fitting algorithm.

        Optimizes parameters by grid searching and evaluating on train/val sets.
        Saves best configuration to output/best_config.json.

        Args:
            epochs: Not used (kept for compatibility)
            batch_size: Number of training samples to use per config evaluation
        """
        logger.info("=== HYPERPARAMETER OPTIMIZATION MODE ===")
        logger.info("Optimizing heuristic parameters for pattern fitting")

        # Scan for training data
        pattern_files, cloth_files = self.scan_images()

        if not pattern_files or not cloth_files:
            logger.error("No training data found")
            return

        logger.info(
            f"Available: {len(pattern_files)} patterns, {len(cloth_files)} cloths"
        )

        # Split data
        split_data = self.load_or_create_split(pattern_files, cloth_files)

        train_cloth = split_data.get("train", {}).get(
            "cloth", split_data.get("cloth_train", [])
        )
        train_pattern = split_data.get("train", {}).get(
            "pattern", split_data.get("pattern_train", [])
        )
        val_cloth = split_data.get("val", {}).get(
            "cloth", split_data.get("cloth_test", [])
        )[:15]
        val_pattern = split_data.get("val", {}).get(
            "pattern", split_data.get("pattern_test", [])
        )

        # Create training samples
        train_samples = []
        for cloth_file in train_cloth[:batch_size]:
            num_patterns = random.randint(3, 6)
            selected_patterns = random.sample(
                train_pattern, min(num_patterns, len(train_pattern))
            )
            train_samples.append({"patterns": selected_patterns, "cloth": cloth_file})

        # Create validation samples
        val_samples = []
        for cloth_file in val_cloth:
            num_patterns = random.randint(3, 6)
            selected_patterns = random.sample(
                val_pattern, min(num_patterns, len(val_pattern))
            )
            val_samples.append({"patterns": selected_patterns, "cloth": cloth_file})

        logger.info(f"Training samples: {len(train_samples)}")
        logger.info(f"Validation samples: {len(val_samples)}")

        # Load models
        self.pattern_module.load_model()
        self.cloth_module.load_model()
        self.fitting_module.load_model()

        # Initialize optimizer
        optimizer = HeuristicOptimizer(self)

        # Define search space (smaller for faster training)
        param_grid = {
            "grid_size": [20, 25],  # Grid resolution
            "rotation_angles": [
                [0, 90, 180, 270],  # Orthogonal
                [0, 45, 90, 135, 180, 225, 270, 315],  # 8-way
            ],
            "allow_flipping": [True],  # Always allow flipping
            "max_attempts": [500],  # Fixed for speed
        }

        # Run optimization
        results = optimizer.optimize(train_samples, val_samples, param_grid)

        # Apply best configuration
        if results["best_config"]:
            self._apply_best_config(results["best_config"])

        # Save results
        optimizer.save_results(str(self.output_dir))

        # Save fitted module with best config
        self.fitting_module.save_model()

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best score: {results['best_score']:.2f}")
        logger.info(f"Configurations tested: {len(results['training_history'])}")
        logger.info("=" * 70)

        return results

    def run_evaluation(self):
        """
        Run comprehensive evaluation on test set with best configuration.

        Loads best trained configuration and evaluates on held-out test data.
        Generates detailed performance reports.
        """
        import time

        logger.info("=== ENHANCED EVALUATION MODE ===")

        # Load test data
        pattern_files, cloth_files = self.scan_images()
        split_data = self.load_or_create_split(pattern_files, cloth_files)

        # Load best models and configuration
        self.pattern_module.load_model()
        self.cloth_module.load_model()
        self.fitting_module.load_model()

        # Load best config if available
        config_path = os.path.join(self.output_dir, "best_config.json")
        if os.path.exists(config_path):
            logger.info(f"Loading best configuration from {config_path}")
            with open(config_path, "r") as f:
                best_config = json.load(f)
            self._apply_best_config(best_config)
        else:
            logger.warning("No best config found, using default parameters")

        # Get test data
        test_cloth = split_data.get("test", {}).get(
            "cloth", split_data.get("cloth_test", [])
        )
        test_pattern = split_data.get("test", {}).get(
            "pattern", split_data.get("pattern_test", [])
        )

        logger.info(f"Test set: {len(test_cloth)} cloths, {len(test_pattern)} patterns")

        # Evaluate on test set
        test_results = []
        test_samples = []

        for cloth_file in test_cloth[:20]:  # Test on 20 cloths
            num_patterns = random.randint(3, 7)
            selected_patterns = random.sample(
                test_pattern, min(num_patterns, len(test_pattern))
            )
            test_samples.append({"patterns": selected_patterns, "cloth": cloth_file})

        logger.info(f"Evaluating on {len(test_samples)} test samples...")

        for i, sample in enumerate(test_samples, 1):
            logger.info(f"Testing sample {i}/{len(test_samples)}...")

            start_time = time.time()

            # Process patterns
            patterns = [
                self.pattern_module.process_image(p) for p in sample["patterns"]
            ]
            patterns = [p for p in patterns if p is not None]

            if not patterns:
                continue

            # Process cloth
            cloth = self.cloth_module.process_image(sample["cloth"])

            # Fit patterns
            result = self.fitting_module.fit_patterns(patterns, cloth)

            elapsed_time = time.time() - start_time

            # Store detailed results
            test_results.append(
                {
                    "cloth_file": os.path.basename(sample["cloth"]),
                    "cloth_type": cloth.cloth_type,
                    "cloth_size": f"{cloth.width:.1f}x{cloth.height:.1f}cm",
                    "num_patterns": len(patterns),
                    "patterns_placed": result["patterns_placed"],
                    "utilization": result["utilization_percentage"],
                    "success_rate": result["success_rate"],
                    "waste_area": result["waste_area"],
                    "processing_time": elapsed_time,
                }
            )

        # Calculate comprehensive metrics
        if test_results:
            avg_utilization = np.mean([r["utilization"] for r in test_results])
            avg_success = np.mean([r["success_rate"] for r in test_results])
            avg_waste = np.mean([r["waste_area"] for r in test_results])
            avg_time = np.mean([r["processing_time"] for r in test_results])
            total_placed = sum([r["patterns_placed"] for r in test_results])
            total_attempted = sum([r["num_patterns"] for r in test_results])
        else:
            avg_utilization = avg_success = avg_waste = avg_time = 0
            total_placed = total_attempted = 0

        # Log results
        logger.info("\n" + "=" * 70)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Samples evaluated: {len(test_results)}")
        logger.info(f"Average utilization: {avg_utilization:.1f}%")
        logger.info(f"Average success rate: {avg_success:.1f}%")
        logger.info(f"Average waste: {avg_waste:.1f} cm²")
        logger.info(f"Average processing time: {avg_time:.2f}s")
        logger.info(f"Total patterns placed: {total_placed}/{total_attempted}")
        logger.info("=" * 70)

        # Save results
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "num_samples": len(test_results),
                        "avg_utilization": avg_utilization,
                        "avg_success_rate": avg_success,
                        "avg_waste": avg_waste,
                        "avg_time": avg_time,
                        "total_placed": total_placed,
                        "total_attempted": total_attempted,
                    },
                    "detailed_results": test_results,
                },
                f,
                indent=2,
            )
        logger.info(f"Detailed results saved to {results_path}")

        # Create summary report
        summary_path = os.path.join(self.output_dir, "evaluation_summary.txt")
        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("EVALUATION SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Samples evaluated: {len(test_results)}\n")
            f.write(f"Average utilization: {avg_utilization:.1f}%\n")
            f.write(f"Average success rate: {avg_success:.1f}%\n")
            f.write(f"Average waste: {avg_waste:.1f} cm²\n")
            f.write(f"Average processing time: {avg_time:.2f}s\n")
            f.write(f"Total patterns placed: {total_placed}/{total_attempted}\n")
        logger.info(f"Summary report saved to {summary_path}")

        return test_results

    def save_all_results(self, all_results: List[Dict]):
        """
        Save multiple fitting results to a JSON file.
        """
        # Create serializable results
        serializable_results = []
        for i, result in enumerate(all_results):
            serializable_result = {
                "cloth_index": i + 1,
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
                "cloth_path": result.get("cloth_path"),
            }
            serializable_results.append(serializable_result)

        # Save to file
        results_file = self.output_dir / "all_cloths_max_patterns_results.json"
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"All results saved to {results_file}")

    def run_all_cloths_max_patterns(self, max_patterns_per_cloth: int = 20):
        """
        Process all cloth images and fit as many patterns as possible on each.
        Ensures patterns are not repeated across different cloths.
        Shows only successfully placed patterns in visualizations.
        """
        logger.info("=== ALL CLOTHS - MAX PATTERNS MODE ===")

        # Scan for images
        pattern_files, cloth_files = self.scan_images()

        if not pattern_files or not cloth_files:
            logger.error("No images found. Please add pattern and cloth images.")
            return

        logger.info(
            f"Processing {len(cloth_files)} cloth images with up to {max_patterns_per_cloth} patterns each"
        )
        logger.info(f"Total available patterns: {len(pattern_files)}")

        # Limit total patterns to process to avoid excessive runtime
        max_total_patterns = min(
            len(pattern_files), 800
        )  # Process max 800 unique patterns to cover more cloths
        logger.info(
            f"Will process maximum {max_total_patterns} unique patterns across all cloths"
        )

        # Load models once
        self.pattern_module.load_model()
        self.cloth_module.load_model()
        self.fitting_module.load_model()

        # Track used patterns globally to avoid repetition
        used_patterns = set()
        all_results = []

        # Process each cloth
        for i, cloth_file in enumerate(cloth_files):
            logger.info(
                f"\n--- Processing cloth {i + 1}/{len(cloth_files)}: {os.path.basename(cloth_file)} ---"
            )

            try:
                # Process cloth
                cloth = self.process_cloth(cloth_file)

                # Filter out already used patterns
                available_patterns = [
                    p for p in pattern_files if p not in used_patterns
                ]

                # Also limit by total patterns processed
                if len(used_patterns) >= max_total_patterns:
                    logger.info(
                        f"Reached maximum pattern limit ({max_total_patterns}). Stopping."
                    )
                    break

                if not available_patterns:
                    logger.warning(
                        f"No more unused patterns available for {os.path.basename(cloth_file)}"
                    )
                    break

                logger.info(
                    f"Available patterns for this cloth: {len(available_patterns)}"
                )

                # Select patterns that might fit this cloth from available ones
                suitable_patterns = self.select_patterns_for_cloth(
                    available_patterns, cloth, max_patterns_per_cloth
                )

                if not suitable_patterns:
                    logger.warning(
                        f"No suitable patterns found for {os.path.basename(cloth_file)}"
                    )
                    continue

                logger.info(f"Selected {len(suitable_patterns)} patterns for fitting")

                # Process patterns
                patterns = self.process_patterns(suitable_patterns)

                if not patterns:
                    logger.warning(
                        f"No valid patterns processed for {os.path.basename(cloth_file)}"
                    )
                    continue

                # Perform fitting
                result = self.fitting_module.fit_patterns(patterns, cloth)

                # Mark used patterns (both attempted and successfully placed)
                for pattern_file in suitable_patterns:
                    used_patterns.add(pattern_file)

                logger.info(
                    f"Total patterns used so far: {len(used_patterns)}/{len(pattern_files)}"
                )

                # Generate visualization with only successful patterns
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                viz_path = (
                    self.output_dir
                    / f"max_fit_{i + 1:02d}_{os.path.splitext(os.path.basename(cloth_file))[0]}_{timestamp}.png"
                )

                # Get only successfully placed patterns for visualization
                successful_patterns = []
                for placed_pattern in result.get("placed_patterns", []):
                    # placed_pattern is a PlacementResult object with a 'pattern' attribute
                    successful_patterns.append(placed_pattern.pattern)

                # Visualize only successful placements
                if successful_patterns:
                    self.fitting_module.visualize(
                        result, successful_patterns, cloth, str(viz_path)
                    )
                else:
                    logger.warning(
                        f"No patterns successfully placed on {os.path.basename(cloth_file)}"
                    )
                    continue

                # Also visualize cloth analysis
                cloth_viz_path = (
                    self.output_dir
                    / f"cloth_analysis_{i + 1:02d}_{os.path.splitext(os.path.basename(cloth_file))[0]}_{timestamp}.png"
                )
                self.cloth_module.visualize(cloth, str(cloth_viz_path))

                # Add paths to result
                result["visualization_path"] = str(viz_path)
                result["cloth_analysis_path"] = str(cloth_viz_path)
                result["cloth_path"] = cloth_file
                result["pattern_paths"] = suitable_patterns

                # Display metrics
                logger.info(f"Cloth {i + 1} Results:")
                logger.info(
                    f"  Patterns placed: {result['patterns_placed']}/{result['patterns_total']}"
                )
                logger.info(
                    f"  Material utilization: {result['utilization_percentage']:.1f}%"
                )
                logger.info(f"  Waste area: {result['waste_area']:.1f} cm²")
                logger.info(f"  Success rate: {result['success_rate']:.1f}%")

                all_results.append(result)

            except Exception as e:
                logger.error(
                    f"Failed to process cloth {os.path.basename(cloth_file)}: {e}"
                )
                continue

        # Save all results
        if all_results:
            self.save_all_results(all_results)

            # Summary statistics
            total_patterns = sum(r["patterns_total"] for r in all_results)
            total_placed = sum(r["patterns_placed"] for r in all_results)
            avg_utilization = sum(
                r["utilization_percentage"] for r in all_results
            ) / len(all_results)

            logger.info("\n" + "=" * 60)
            logger.info("ALL CLOTHS PROCESSING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Cloths processed: {len(all_results)}/{len(cloth_files)}")
            logger.info(f"Total patterns attempted: {total_patterns}")
            logger.info(f"Total patterns placed: {total_placed}")
            logger.info(
                f"Overall success rate: {100 * total_placed / total_patterns:.1f}%"
            )
            logger.info(f"Average utilization: {avg_utilization:.1f}%")
            logger.info("=" * 60)

        return all_results

    def select_patterns_for_cloth(
        self, pattern_files: List[str], cloth: ClothMaterial, max_patterns: int
    ) -> List[str]:
        """
        Select patterns that are likely to fit on the given cloth.
        Prioritizes smaller patterns first to maximize count.
        Optimized to avoid processing all patterns.
        """
        # Get cloth dimensions
        cloth_width = cloth.width
        cloth_height = cloth.height
        cloth_area = cloth_width * cloth_height

        # Limit the number of evaluate for performance
        # Take a reasonable sample that still provides variety
        sample_size = min(
            len(pattern_files), max_patterns * 8
        )  # Sample 8x what we need for efficiency

        # Randomly sample patterns to evaluate (for variety and performance)
        import random

        sampled_patterns = random.sample(
            pattern_files, min(sample_size, len(pattern_files))
        )

        # Calculate pattern areas and sort by size (smallest first)
        pattern_info = []
        for pattern_file in sampled_patterns:
            try:
                filename = os.path.basename(pattern_file)
                width, height = self.pattern_module.extract_dimensions_from_filename(
                    filename
                )
                if width and height:
                    pattern_area = width * height
                    # Only include patterns that are smaller than the cloth
                    if (
                        pattern_area < cloth_area * 0.5
                    ):  # Pattern should be less than half the cloth area
                        pattern_info.append(
                            {
                                "file": pattern_file,
                                "area": pattern_area,
                                "width": width,
                                "height": height,
                            }
                        )
            except Exception:
                continue

        # If we don't have enough suitable patterns, try a larger sample
        if len(pattern_info) < max_patterns and sample_size < len(pattern_files):
            additional_sample_size = min(
                len(pattern_files) - sample_size, max_patterns * 5
            )
            additional_patterns = [
                p for p in pattern_files if p not in sampled_patterns
            ]
            additional_sampled = random.sample(
                additional_patterns, additional_sample_size
            )

            for pattern_file in additional_sampled:
                try:
                    filename = os.path.basename(pattern_file)
                    width, height = (
                        self.pattern_module.extract_dimensions_from_filename(filename)
                    )
                    if width and height:
                        pattern_area = width * height
                        if pattern_area < cloth_area * 0.5:
                            pattern_info.append(
                                {
                                    "file": pattern_file,
                                    "area": pattern_area,
                                    "width": width,
                                    "height": height,
                                }
                            )
                except Exception:
                    continue

        # Sort by area (smallest first) to maximize count
        pattern_info.sort(key=lambda x: x["area"])

        # Add some variety: take 70% smallest, 30% random from remaining suitable patterns
        num_small = min(int(max_patterns * 0.7), len(pattern_info))
        num_random = min(max_patterns - num_small, len(pattern_info) - num_small)

        selected_small = [p["file"] for p in pattern_info[:num_small]]

        # Random selection from remaining patterns for variety
        remaining_patterns = pattern_info[num_small:]
        if remaining_patterns and num_random > 0:
            random.shuffle(remaining_patterns)
            selected_random = [p["file"] for p in remaining_patterns[:num_random]]
        else:
            selected_random = []

        selected_patterns = selected_small + selected_random

        logger.info(
            f"Selected {len(selected_patterns)} patterns ({num_small} smallest + {len(selected_random)} varied) from {len(sampled_patterns)} sampled for {cloth_width}x{cloth_height}cm cloth"
        )
        return selected_patterns

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

        # Select patterns
        selected_patterns = test_patterns[: min(num_patterns, len(test_patterns))]

        # Select appropriately sized cloth for the patterns
        selected_cloth = self.select_suitable_cloth(selected_patterns, test_cloths)

        if not selected_cloth:
            logger.error("No suitable cloth found for demo patterns")
            return

        logger.info(f"Selected {len(selected_patterns)} patterns and 1 cloth for demo")

        # Run fitting task
        try:
            result = self.run_fitting_task(selected_patterns, selected_cloth)

            # Save result metrics
            if result:
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
        choices=["demo", "train", "eval", "fit", "all_cloths"],
        default="demo",
        help="Operation mode: demo, train, eval, fit, or all_cloths",
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

    # All cloths mode options
    parser.add_argument(
        "--max_patterns_per_cloth",
        type=int,
        default=50,
        help="Maximum patterns to attempt fitting on each cloth",
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
        system.run_training(args.epochs)

    elif args.mode == "eval":
        # Evaluation mode
        system.run_evaluation()

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

    elif args.mode == "all_cloths":
        # All cloths mode - maximize pattern fitting
        system.run_all_cloths_max_patterns(args.max_patterns_per_cloth)

    logger.info("\nProcessing complete")


if __name__ == "__main__":
    main()
