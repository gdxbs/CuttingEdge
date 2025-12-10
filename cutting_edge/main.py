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
import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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

    def __init__(self, base_dir: Optional[str] = None, auto_scale: Optional[bool] = None):
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

        # Create logs directory
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        self._setup_file_logging()

        # Initialize modules
        self.pattern_module = PatternRecognitionModule()
        self.cloth_module = ClothRecognitionModule()
        self.fitting_module = PatternFittingModule(auto_scale=auto_scale)

        logger.info("Cutting Edge System initialized")
        logger.info(f"Base directory: {self.base_dir}")

    def _setup_file_logging(self):
        """Setup file logging with timestamps for research paper documentation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"cutting_edge_{timestamp}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter with timestamps
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        # Add handler to root logger only if not already present
        root_logger = logging.getLogger()
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        
        if not has_file_handler:
            root_logger.addHandler(file_handler)
            logger.info(f"Log file created: {log_file}")
        else:
            logger.info(f"Logging to existing handlers: {[type(h).__name__ for h in root_logger.handlers]}")

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

    def process_fitting(
        self, patterns: List[Pattern], cloth: ClothMaterial, baseline_mode: bool = False
    ) -> Dict:
        """
        Perform pattern fitting and generate visualization.
        """
        # Fit patterns on cloth
        logger.info(f"Fitting {len(patterns)} patterns onto cloth...")
        if baseline_mode:
            logger.info("Using BLF Baseline mode (Simple Greedy, Bottom-Left, 90° rotations)")
            
        result = self.fitting_module.fit_patterns(patterns, cloth, baseline_mode=baseline_mode)

        # Generate visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = self.output_dir / f"fitting_result_{timestamp}.png"
        # Pass cloth name if available
        cloth_name = getattr(cloth, "source_path", None)
        if cloth_name:
            cloth_name = os.path.basename(cloth_name)
        self.fitting_module.visualize(
            result, patterns, cloth, str(viz_path), cloth_image_name=cloth_name
        )

        # Add visualization path to result
        result["visualization_path"] = str(viz_path)

        # Also visualize cloth analysis
        cloth_viz_path = self.output_dir / f"cloth_analysis_{timestamp}.png"
        self.cloth_module.visualize(cloth, str(cloth_viz_path))
        result["cloth_analysis_path"] = str(cloth_viz_path)

        return result

    def run_fitting_task(
        self, pattern_paths: List[str], cloth_path: str, baseline_mode: bool = False
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
        result = self.process_fitting(patterns, cloth, baseline_mode=baseline_mode)

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

    def _save_evaluation_outputs(
        self, test_results: List[Dict], test_samples: List[Dict], extra_metrics: Dict = None
    ):
        """Save comprehensive evaluation outputs for research paper."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create subdirectories
        eval_charts_dir = os.path.join(self.output_dir, "evaluation_charts")
        eval_logs_dir = os.path.join(self.output_dir, "evaluation_logs")
        eval_images_dir = os.path.join(self.output_dir, "evaluation_images")
        os.makedirs(eval_charts_dir, exist_ok=True)
        os.makedirs(eval_logs_dir, exist_ok=True)
        os.makedirs(eval_images_dir, exist_ok=True)

        # Calculate summary statistics
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

        # Save JSON results
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "summary": {
                        "num_samples": len(test_results),
                        "avg_utilization": float(avg_utilization),
                        "avg_success_rate": float(avg_success),
                        "avg_waste": float(avg_waste),
                        "avg_time": float(avg_time),
                        "total_placed": int(total_placed),
                        "total_attempted": int(total_attempted),
                    },
                    "detailed_results": test_results,
                },
                f,
                indent=2,
            )
        logger.info(f"Detailed results saved to {results_path}")

        # Save CSV for Excel/LaTeX tables
        csv_path = os.path.join(eval_logs_dir, f"evaluation_metrics_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Sample_ID",
                    "Cloth_File",
                    "Cloth_Type",
                    "Cloth_Size_cm",
                    "Num_Patterns",
                    "Patterns_Placed",
                    "Utilization_%",
                    "Success_Rate_%",
                    "Waste_Area_cm2",
                    "Processing_Time_sec",
                    "Visualization_File",
                ]
            )
            for result in test_results:
                writer.writerow(
                    [
                        result.get("sample_id", "N/A"),
                        result["cloth_file"],
                        result["cloth_type"],
                        result["cloth_size"],
                        result["num_patterns"],
                        result["patterns_placed"],
                        f"{result['utilization']:.2f}",
                        f"{result['success_rate']:.2f}",
                        f"{result['waste_area']:.2f}",
                        f"{result['processing_time']:.3f}",
                        result.get("visualization_file", "N/A"),
                    ]
                )
        logger.info(f"Evaluation metrics CSV saved to {csv_path}")

        # Save detailed text summary
        summary_path = os.path.join(
            eval_logs_dir, f"evaluation_summary_{timestamp}.txt"
        )
        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("EVALUATION SUMMARY REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples evaluated: {len(test_results)}\n\n")
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average utilization: {avg_utilization:.2f}%\n")
            f.write(f"Average success rate: {avg_success:.2f}%\n")
            f.write(f"Average waste: {avg_waste:.2f} cm²\n")
            f.write(f"Average processing time: {avg_time:.3f}s\n")
            f.write(f"Total patterns placed: {total_placed}/{total_attempted}\n")
            f.write(
                f"Overall success rate: {(total_placed / total_attempted * 100) if total_attempted > 0 else 0:.2f}%\n\n"
            )

            # Detailed per-sample results
            f.write("DETAILED RESULTS PER SAMPLE:\n")
            f.write("-" * 70 + "\n")
            for i, result in enumerate(test_results, 1):
                f.write(f"\nSample {i}: {result['cloth_file']}\n")
                f.write(f"  Cloth: {result['cloth_type']}, {result['cloth_size']}\n")
                f.write(
                    f"  Patterns: {result['patterns_placed']}/{result['num_patterns']} placed\n"
                )
                f.write(f"  Utilization: {result['utilization']:.2f}%\n")
                f.write(f"  Waste: {result['waste_area']:.2f} cm²\n")
                f.write(f"  Time: {result['processing_time']:.3f}s\n")
        
            # Advanced Metrics Section (Moved from Charts)
            if extra_metrics:
                f.write("\n" + "=" * 70 + "\n")
                f.write("ADVANCED METRICS:\n")
                f.write("-" * 70 + "\n")
                
                # Defect Detection F1 Scores
                defect_f1 = extra_metrics.get("defect_f1", {})
                if defect_f1:
                    f.write(f"DEFECT DETECTION F1 SCORES:\n")
                    for dtype in ["hole", "stain", "line", "freeform"]:
                        f1_val = defect_f1.get(dtype, 0)
                        f.write(f"  {dtype.capitalize()}: {f1_val:.3f}\n")
                    f.write("\n")
                
                # Grain Direction
                grain_errs = extra_metrics.get("grain_errors", [])
                if grain_errs:
                    f.write(f"GRAIN DIRECTION ERROR:\n")
                    f.write(f"  Mean: {np.mean(grain_errs):.1f}°\n")
                    f.write(f"  Median: {np.median(grain_errs):.1f}°\n")
                    f.write(f"  Max: {np.max(grain_errs):.1f}°\n")
                    f.write(f"  Samples: {len(grain_errs)}\n\n")
                
                # Classification Accuracy
                cls_stats = extra_metrics.get("class_acc_stats", {})
                if cls_stats:
                    total_correct = sum(s["correct"] for s in cls_stats.values())
                    total_patterns = sum(s["total"] for s in cls_stats.values())
                    acc = total_correct / total_patterns if total_patterns > 0 else 0
                    f.write(f"PATTERN CLASSIFICATION:\n")
                    f.write(f"  Overall Accuracy: {acc*100:.1f}% ({total_correct}/{total_patterns})\n")
                    sorted_cats = sorted(cls_stats.items(), key=lambda x: x[1]['total'], reverse=True)
                    for cat, stat in sorted_cats:
                        if stat['total'] > 0:
                            cat_acc = stat['correct'] / stat['total']
                            f.write(f"    - {cat}: {cat_acc*100:.1f}% ({stat['correct']}/{stat['total']})\n")
                    f.write("\n")

                # Dimension Prediction
                dim_errs = extra_metrics.get("dim_errors", [])
                if dim_errs:
                    f.write(f"DIMENSION PREDICTION ERROR (MAE):\n")
                    f.write(f"  Mean: {np.mean(dim_errs):.2f} cm\n")
                    f.write(f"  Median: {np.median(dim_errs):.2f} cm\n")
                    f.write(f"  Max: {np.max(dim_errs):.2f} cm\n")
                    f.write(f"  Samples: {len(dim_errs)}\n")
        logger.info(f"Evaluation summary saved to {summary_path}")

        # Generate visualization charts
        self._generate_evaluation_charts(test_results, eval_charts_dir, extra_metrics)
        logger.info(f"Evaluation charts saved to {eval_charts_dir}/")

    def _generate_evaluation_charts(self, test_results: List[Dict], charts_dir: str, extra_metrics: Dict = None):
        """Generate comprehensive 3x3 visualization charts for evaluation results."""
        if not test_results:
            return

        plt.figure(figsize=(18, 15))
        
        # --- Row 1: Distributions ---

        # Chart 1: Utilization distribution
        plt.subplot(3, 3, 1)
        utilizations = [r["utilization"] for r in test_results]
        plt.hist(utilizations, bins=15, alpha=0.7, color="steelblue", edgecolor="black")
        mean_util = float(np.mean(utilizations))
        plt.axvline(
            mean_util,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_util:.1f}%",
        )
        plt.xlabel("Utilization (%)", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.title("Fabric Utilization Distribution", fontsize=11, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Chart 2: Success rate distribution
        plt.subplot(3, 3, 2)
        success_rates = [r["success_rate"] for r in test_results]
        plt.hist(success_rates, bins=15, alpha=0.7, color="green", edgecolor="black")
        mean_success = float(np.mean(success_rates))
        plt.axvline(
            mean_success,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_success:.1f}%",
        )
        plt.xlabel("Success Rate (%)", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.title("Pattern Placement Success Rate", fontsize=11, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Chart 3: Processing time distribution
        plt.subplot(3, 3, 3)
        times = [r["processing_time"] for r in test_results]
        plt.hist(times, bins=15, alpha=0.7, color="coral", edgecolor="black")
        mean_time = float(np.mean(times))
        plt.axvline(
            mean_time,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_time:.2f}s",
        )
        plt.xlabel("Processing Time (s)", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.title("Computational Efficiency", fontsize=11, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # --- Row 2: Training vs Testing ---
        
        # Load training logs
        train_logs = []
        try:
            log_dir = os.path.join(os.path.dirname(charts_dir), "training_logs")
            log_files = sorted([f for f in os.listdir(log_dir) if f.startswith("training_metrics")])
            if log_files:
                last_log = os.path.join(log_dir, log_files[-1])
                with open(last_log, 'r') as f:
                    reader = csv.DictReader(f)
                    train_logs = list(reader)
        except Exception as e:
            logger.warning(f"Could not load training logs: {e}")

        # Chart 4: Train vs Test Utilization
        plt.subplot(3, 3, 4)
        if train_logs:
            iters = [int(r["Config_ID"]) for r in train_logs]
            tr_util = [float(r["Train_Utilization"]) for r in train_logs]
            val_util = [float(r["Val_Utilization"]) for r in train_logs]
            plt.plot(iters, tr_util, 'o-', label='Train', color='steelblue', alpha=0.7)
            plt.plot(iters, val_util, 's-', label='Test (Val)', color='orange', alpha=0.7)
            plt.axhline(mean_util, color='green', linestyle='--', label='Current Test Set')
            plt.xlabel("Training Iteration", fontsize=10)
            plt.ylabel("Utilization (%)", fontsize=10)
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No Training Logs Found", ha='center', va='center')
        plt.title("Train vs Test: Utilization", fontsize=11, fontweight="bold")
        plt.grid(True, alpha=0.3)
        
        # Chart 5: Train vs Test Success Rate
        plt.subplot(3, 3, 5)
        if train_logs:
            tr_succ = [float(r["Train_Success_Rate"]) for r in train_logs]
            val_succ = [float(r["Val_Success_Rate"]) for r in train_logs]
            plt.plot(iters, tr_succ, 'o-', label='Train', color='steelblue', alpha=0.7)
            plt.plot(iters, val_succ, 's-', label='Test (Val)', color='orange', alpha=0.7)
            plt.axhline(mean_success, color='green', linestyle='--', label='Current Test Set')
            plt.xlabel("Training Iteration", fontsize=10)
            plt.ylabel("Success Rate (%)", fontsize=10)
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No Training Logs Found", ha='center', va='center')
        plt.title("Train vs Test: Success Rate", fontsize=11, fontweight="bold")
        plt.grid(True, alpha=0.3)

        # --- Row 3: Detail Metrics ---
        if not extra_metrics:
            extra_metrics = {}

        # Chart 6: Defect F1 Scores
        plt.subplot(3, 3, 6)
        f1_scores = extra_metrics.get("defect_f1", {"hole": 0, "stain": 0, "line": 0, "freeform": 0})
        labels = list(f1_scores.keys())
        values = list(f1_scores.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Added red for freeform
        plt.bar(labels, values, color=colors[:len(labels)], alpha=0.7, edgecolor='black')
        plt.ylim(0, 1.0)
        plt.ylabel("F1 Score", fontsize=10)
        plt.title("Defect Detection Performance", fontsize=11, fontweight="bold")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)

        # Chart 7: Utilization vs Pattern Count
        plt.subplot(3, 3, 7)
        if test_results:
            counts = [r["num_patterns"] for r in test_results]
            utils = [r["utilization"] for r in test_results]
            # Add some jitter to x for visibility if counts are integers
            jitter = np.random.normal(0, 0.1, len(counts))
            plt.scatter(np.array(counts) + jitter, utils, c='steelblue', alpha=0.6, edgecolors='k')
            
            # Trend line
            if len(counts) > 1:
                z = np.polyfit(counts, utils, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(counts), max(counts), 10)
                plt.plot(x_range, p(x_range), "r--", alpha=0.8, label=f"Trend")
                
            plt.xlabel("Number of Patterns", fontsize=10)
            plt.ylabel("Utilization (%)", fontsize=10)
            plt.title("Utilization vs Pattern Count", fontsize=11, fontweight="bold")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No Data", ha='center')

        # Chart 8: Per Sample Utilization
        plt.subplot(3, 3, 8)
        if test_results:
            sample_ids = [r["sample_id"] for r in test_results]
            utils = [r["utilization"] for r in test_results]
            colors = plt.cm.viridis(np.linspace(0, 1, len(utils)))
            bars = plt.bar(sample_ids, utils, color=colors, alpha=0.8)
            plt.axhline(mean_util, color='r', linestyle='--', label='Mean')
            plt.xlabel("Sample ID", fontsize=10)
            plt.ylabel("Utilization (%)", fontsize=10)
            plt.title("Per Sample Utilization", fontsize=11, fontweight="bold")
            # Limit x-ticks if too many samples
            if len(sample_ids) > 20:
                plt.xticks(sample_ids[::int(len(sample_ids)/10)], rotation=45)
            else:
                plt.xticks(sample_ids)
            plt.grid(True, axis='y', alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No Data", ha='center')

        # Chart 9: Per Sample Success Rate
        plt.subplot(3, 3, 9)
        if test_results:
            sample_ids = [r["sample_id"] for r in test_results]
            success = [r["success_rate"] for r in test_results]
            bars = plt.bar(sample_ids, success, color='lightgreen', edgecolor='green', alpha=0.7)
            plt.axhline(mean_success, color='r', linestyle='--', label='Mean')
            plt.xlabel("Sample ID", fontsize=10)
            plt.ylabel("Success Rate (%)", fontsize=10)
            plt.title("Per Sample Success Rate", fontsize=11, fontweight="bold")
             # Limit x-ticks if too many samples
            if len(sample_ids) > 20:
                plt.xticks(sample_ids[::int(len(sample_ids)/10)], rotation=45)
            else:
                plt.xticks(sample_ids)
            plt.grid(True, axis='y', alpha=0.3)
        else:
             plt.text(0.5, 0.5, "No Data", ha='center')

        plt.tight_layout()
        plt.savefig(
            os.path.join(charts_dir, "evaluation_comprehensive.png"),
            dpi=150, # Lower DPI slightly for large image
            bbox_inches="tight",
        )
        plt.close()

        # Create a summary performance chart
        plt.figure(figsize=(10, 6))
        metrics = ["Utilization\n(%)", "Success\nRate (%)", "Patterns\nPlaced (%)"]
        values = [
            np.mean(utilizations),
            np.mean(success_rates),
            (
                sum([r["patterns_placed"] for r in test_results])
                / sum([r["num_patterns"] for r in test_results])
                * 100
            ),
        ]
        colors = ["steelblue", "green", "coral"]
        bars = plt.bar(
            metrics, values, alpha=0.8, color=colors, edgecolor="black", linewidth=1.5
        )

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        plt.ylabel("Performance (%)", fontsize=13)
        plt.title(
            "Overall Evaluation Performance Summary", fontsize=14, fontweight="bold"
        )
        plt.ylim(0, 105)
        plt.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(charts_dir, "evaluation_summary.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def run_training(
        self, epochs: int = 10, batch_size: int = 15, optimizer_type: str = "heuristic"
    ):
        """
        Run hyperparameter optimization.
        
        Args:
            epochs: Not used
            batch_size: Training batch size
            optimizer_type: "heuristic" or "bayesian"
        """
        logger.info("=== HYPERPARAMETER OPTIMIZATION MODE ===")
        logger.info(f"Optimizing parameters using {optimizer_type.upper()} optimizer")
        
        # Import optimizers
        from .training_optimizer import HeuristicOptimizer, BayesianOptimizer

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
        if optimizer_type == "bayesian":
            optimizer = BayesianOptimizer(self)
        else:
            optimizer = HeuristicOptimizer(self)

        if optimizer_type == "bayesian":
            # Bayesian handles its own search space
            results = optimizer.optimize(train_samples, val_samples)
        else:
            # Define search space for heuristic (smaller for faster training)
            param_grid = {
                "grid_size": [20, 25],  # Grid resolution
                "rotation_angles": [
                    [0, 90, 180, 270],  # Orthogonal
                    [0, 45, 90, 135, 180, 225, 270, 315],  # 8-way
                ],
                "allow_flipping": [True],  # Always allow flipping
                "max_attempts": [500],  # Fixed for speed
            }
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

        # Stats counters
        defect_stats = {
            "hole": {"TP": 0, "FP": 0, "FN": 0},
            "stain": {"TP": 0, "FP": 0, "FN": 0},
            "line": {"TP": 0, "FP": 0, "FN": 0},
            "freeform": {"TP": 0, "FP": 0, "FN": 0}
        }
        grain_errors = []
        
        # New Stats for Charts
        class_acc_stats = {ptype: {"correct": 0, "total": 0} for ptype in self.pattern_module.pattern_types}
        dim_errors = [] # List of MAE values (cm)

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
                
            # --- Pattern Analysis (Class & Dimensions) ---
            for pat, pat_path in zip(patterns, sample["patterns"]):
                 # Ground Truth from filename
                 fname = os.path.basename(pat_path)
                 gt_w, gt_h = self.pattern_module.extract_dimensions_from_filename(fname)
                 
                 # Classification GT (simple heuristic from filename)
                 # Type aliases: map common filename patterns to canonical types
                 type_aliases = {
                     "tee": "shirt", "t_shirt": "shirt", "blouse": "shirt", "top": "shirt",
                     "jacket": "other", "coat": "other", "hood": "other", "vest": "other",
                     "trouser": "pants", "jean": "pants", "short": "pants",
                     "gown": "dress", "frock": "dress",
                     "arm": "sleeve", "cuff": "sleeve",
                     "neck": "collar", "neckline": "collar",
                     "front": "bodice", "back": "bodice", "panel": "other"
                 }
                 gt_type = "other"
                 fname_lower = fname.lower()
                 # First check canonical types
                 for ptype in self.pattern_module.pattern_types:
                     if ptype in fname_lower:
                         gt_type = ptype
                         break
                 # Then check aliases if no match
                 if gt_type == "other":
                     for alias, canon_type in type_aliases.items():
                         if alias in fname_lower:
                             gt_type = canon_type
                             break
                 
                 # Accuracy
                 if gt_type in class_acc_stats:
                    if pat.pattern_type == gt_type:
                        class_acc_stats[gt_type]["correct"] += 1
                    class_acc_stats[gt_type]["total"] += 1
                 
                 # Dimension Error
                 if gt_w and gt_h:
                     err_w = abs(pat.width - gt_w)
                     err_h = abs(pat.height - gt_h)
                     dim_errors.append((err_w + err_h) / 2)

            # Process cloth
            cloth = self.cloth_module.process_image(sample["cloth"])
            
            # --- Result Analysis and Scoring ---
            # Infer ground truth from directory name
            cloth_dir = os.path.basename(os.path.dirname(sample["cloth"]))
            
            gt_defects = {"hole": False, "stain": False, "line": False, "freeform": False}
            if "Hole" in cloth_dir:
                gt_defects["hole"] = True
            elif "Stain" in cloth_dir:
                gt_defects["stain"] = True
            elif any(x in cloth_dir for x in ["Lines", "Horizontal", "Vertical"]):
                gt_defects["line"] = True
            elif "freeform" in cloth_dir or "free" in cloth_dir:
                gt_defects["freeform"] = True
            
            # Determine predicted defects
            pred_defects = {"hole": False, "stain": False, "line": False, "freeform": False}
            if cloth.defects_by_type:
                for dtype, defects_list in cloth.defects_by_type.items():
                    if len(defects_list) > 0:
                        if dtype in pred_defects:
                            pred_defects[dtype] = True
            
            # Check for freeform/irregular shape
            if cloth.material_properties and cloth.material_properties.get("is_irregular", False):
                pred_defects["freeform"] = True
            # Fallback for legacy compatibility
            elif cloth.defects and len(cloth.defects) > 0:
                pass 

            # Update Defect Stats
            for dtype in ["hole", "stain", "line", "freeform"]:
                if gt_defects[dtype] and pred_defects[dtype]:
                    defect_stats[dtype]["TP"] += 1
                elif not gt_defects[dtype] and pred_defects[dtype]:
                    defect_stats[dtype]["FP"] += 1
                elif gt_defects[dtype] and not pred_defects[dtype]:
                    defect_stats[dtype]["FN"] += 1
            
            # Grain Direction Analysis
            if "Horizontal" in cloth_dir:
                gt_grain = 0.0
            elif "Vertical" in cloth_dir:
                gt_grain = 90.0
            else:
                gt_grain = None
                
            if gt_grain is not None:
                pred_grain = cloth.material_properties.get("grain_direction", 0)
                error = abs(pred_grain - gt_grain)
                if error > 90: # normalize 180 periodicity
                    error = 180 - error
                grain_errors.append(error)

            # Fit patterns
            result = self.fitting_module.fit_patterns(patterns, cloth)

            elapsed_time = time.time() - start_time

            # Save visualization for this test case
            eval_images_dir = os.path.join(self.output_dir, "evaluation_images")
            os.makedirs(eval_images_dir, exist_ok=True)
            viz_filename = f"test_sample_{i:03d}_{os.path.basename(sample['cloth']).split('.')[0]}.png"
            viz_path = os.path.join(eval_images_dir, viz_filename)

            try:
                # Pass cloth image name for display
                cloth_name = os.path.basename(sample["cloth"])
                self.fitting_module.visualize(
                    result,
                    patterns,
                    cloth,
                    output_path=viz_path,
                    cloth_image_name=cloth_name,
                )
                logger.info(f"  Saved visualization to {viz_filename}")
            except Exception as e:
                logger.warning(f"  Could not save visualization: {e}")

            # Store detailed results
            test_results.append(
                {
                    "sample_id": i,
                    "cloth_file": os.path.basename(sample["cloth"]),
                    "cloth_type": cloth.cloth_type,
                    "cloth_size": f"{cloth.width:.1f}x{cloth.height:.1f}cm",
                    "num_patterns": len(patterns),
                    "patterns_placed": result["patterns_placed"],
                    "utilization": result["utilization_percentage"],
                    "success_rate": result["success_rate"],
                    "waste_area": result["waste_area"],
                    "processing_time": elapsed_time,
                    "visualization_file": viz_filename,
                }
            )

        # Calculate comprehensive metrics
        metrics = {}
        if test_results:
            avg_utilization = np.mean([r["utilization"] for r in test_results])
            avg_success = np.mean([r["success_rate"] for r in test_results])
            avg_waste = np.mean([r["waste_area"] for r in test_results])
            avg_time = np.mean([r["processing_time"] for r in test_results])
            total_placed = sum([r["patterns_placed"] for r in test_results])
            total_attempted = sum([r["num_patterns"] for r in test_results])
            
            # F1 Scores
            for dtype in ["hole", "stain", "line", "freeform"]:
                tp = defect_stats[dtype]["TP"]
                fp = defect_stats[dtype]["FP"]
                fn = defect_stats[dtype]["FN"]
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                metrics[f"{dtype}_f1"] = f1
            
            # Grain Error
            if grain_errors:
                metrics["grain_error_mean"] = getattr(np, "mean")(grain_errors)
                metrics["grain_error_max"] = getattr(np, "max")(grain_errors)
            else:
                metrics["grain_error_mean"] = 0
                metrics["grain_error_max"] = 0
                
            # Class Accuracy
            total_correct = sum(s["correct"] for s in class_acc_stats.values())
            total_patterns_cls = sum(s["total"] for s in class_acc_stats.values())
            metrics["class_accuracy"] = total_correct / total_patterns_cls if total_patterns_cls > 0 else 0
            
            # Dim Error
            metrics["dim_mae"] = getattr(np, "mean")(dim_errors) if dim_errors else 0

        else:
            avg_utilization = avg_success = avg_waste = avg_time = 0
            total_placed = total_attempted = 0
            metrics = {k: 0 for k in ["hole_f1", "stain_f1", "line_f1", "freeform_f1", "grain_error_mean", "class_accuracy", "dim_mae"]}

        # Log results
        logger.info("\n" + "=" * 70)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Evaluated {len(test_results)} samples")
        logger.info(f"Avg Utilization: {avg_utilization:.2f}%")
        logger.info(f"Avg Success Rate: {avg_success:.2f}%")
        logger.info(f"Hole Detection F1: {metrics['hole_f1']:.3f}")
        logger.info(f"Stain Detection F1: {metrics['stain_f1']:.3f}")
        logger.info(f"Line Defect F1: {metrics['line_f1']:.3f}")
        logger.info(f"Freeform Detection F1: {metrics['freeform_f1']:.3f}")
        logger.info(f"Grain Direction Error: {metrics['grain_error_mean']:.1f}°")
        logger.info(f"Pattern Class Acc: {metrics['class_accuracy']*100:.1f}%")
        logger.info(f"Dimension MAE: {metrics['dim_mae']:.1f} cm")
        logger.info("=" * 70)

        # Save comprehensive results for research paper
        extra_metrics = {
            "defect_f1": {k: metrics[f"{k}_f1"] for k in ["hole", "stain", "line", "freeform"]},
            "grain_errors": grain_errors,
            "class_acc_stats": class_acc_stats,
            "dim_errors": dim_errors
        }
        self._save_evaluation_outputs(test_results, test_samples, extra_metrics)

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
        max_total_patterns = len(pattern_files)  # Process all available patterns
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
                    # Pass cloth image name for display
                    cloth_name = os.path.basename(cloth_file)
                    self.fitting_module.visualize(
                        result,
                        successful_patterns,
                        cloth,
                        str(viz_path),
                        cloth_image_name=cloth_name,
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
        choices=["demo", "train", "eval", "fit", "all_cloths", "baseline"],
        default="demo",
        help="Operation mode: demo, train, eval, fit, or all_cloths, baseline",
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
    parser.add_argument(
        "--optimizer",
        choices=["heuristic", "bayesian"],
        default="heuristic",
        help="Optimizer type for training mode",
    )

    # Other options
    parser.add_argument("--base_dir", help="Base directory for the application")
    parser.add_argument("--output", help="Custom output directory for results")
    parser.add_argument(
        "--no_scale",
        action="store_true",
        help="Disable automatic pattern scaling (force 1.0x scale)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Determine auto_scale setting
    auto_scale = False if args.no_scale else None

    # Initialize system
    system = CuttingEdgeSystem(args.base_dir, auto_scale=auto_scale)

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
        system.run_training(args.epochs, optimizer_type=args.optimizer)

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

    elif args.mode == "baseline":
        # Baseline mode (same as fit but with baseline_mode=True)
        pattern_paths = []
        if args.patterns:
            pattern_paths = args.patterns
        elif args.pattern_dir:
            pattern_dir = Path(args.pattern_dir)
            if pattern_dir.exists():
                for ext in SYSTEM["IMAGE_EXTENSIONS"]:
                    pattern_paths.extend([str(f) for f in pattern_dir.glob(f"*.{ext}")])

        if not pattern_paths or not args.cloth:
            logger.error("Please specify patterns and cloth for baseline mode")
            return

        system.run_fitting_task(pattern_paths, args.cloth, baseline_mode=True)

    elif args.mode == "all_cloths":
        # All cloths mode - maximize pattern fitting
        system.run_all_cloths_max_patterns(args.max_patterns_per_cloth)

    logger.info("\nProcessing complete")


if __name__ == "__main__":
    main()
