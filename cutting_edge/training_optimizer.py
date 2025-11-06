"""
Advanced Training and Hyperparameter Optimization System

This module implements a modern hyperparameter optimization approach for
the heuristic pattern fitting algorithm, including:
- Grid search over key parameters
- Bayesian optimization (optional)
- Comprehensive metrics tracking
- Best configuration saving/loading
"""

import csv
import json
import logging
import os
import time
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class HeuristicOptimizer:
    """
    Optimizes heuristic parameters for pattern fitting using grid search.
    """

    def __init__(self, system):
        """Initialize with CuttingEdgeSystem instance."""
        self.system = system
        self.best_config = None
        self.best_score = float("-inf")
        self.training_history = []

    def optimize(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        param_grid: Dict[str, List] | None = None,
    ) -> Dict:
        """
        Run hyperparameter optimization.

        Args:
            train_samples: List of training samples (patterns + cloth combos)
            val_samples: List of validation samples
            param_grid: Dictionary of parameters to search over

        Returns:
            Dictionary with best_config, best_score, and all_results
        """
        grid = param_grid if param_grid is not None else self._get_default_param_grid()

        logger.info("\n" + "=" * 70)
        logger.info("HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 70)
        logger.info(f"Training samples: {len(train_samples)}")
        logger.info(f"Validation samples: {len(val_samples)}")
        logger.info("\nParameter Search Space:")
        for param, values in grid.items():
            logger.info(
                f"  {param}: {values if len(str(values)) < 60 else f'{len(values)} options'}"
            )
        logger.info("=" * 70)

        # Calculate total combinations
        total_combinations = 1
        for values in grid.values():
            total_combinations *= len(values)

        logger.info(f"\nTotal configurations to test: {total_combinations}\n")

        # Grid search
        iteration = 0
        for config_values in product(*grid.values()):
            iteration += 1
            config = dict(zip(grid.keys(), config_values))

            logger.info(f"--- Config {iteration}/{total_combinations} ---")
            logger.info(f"Testing: {self._format_config(config)}")

            # Apply configuration
            self._apply_config(config)

            # Evaluate on train and validation
            start_time = time.time()
            train_metrics = self._evaluate(train_samples)
            val_metrics = self._evaluate(val_samples)
            eval_time = time.time() - start_time

            # Store results
            result = {
                "iteration": iteration,
                "config": config,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "eval_time": eval_time,
            }
            self.training_history.append(result)

            # Log results
            logger.info(
                f"Train: util={train_metrics['utilization']:.1f}% "
                f"success={train_metrics['success_rate']:.1f}% "
                f"time={train_metrics['avg_time']:.2f}s"
            )
            logger.info(
                f"Val:   util={val_metrics['utilization']:.1f}% "
                f"success={val_metrics['success_rate']:.1f}% "
                f"time={val_metrics['avg_time']:.2f}s"
            )

            # Track best (based on validation utilization + success rate)
            val_score = (
                val_metrics["utilization"] * 0.7 + val_metrics["success_rate"] * 0.3
            )
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_config = config
                logger.info(f"✓ NEW BEST! Score: {val_score:.2f}\n")
            else:
                logger.info("")

        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best validation score: {self.best_score:.2f}")
        logger.info(f"Best configuration:")
        best_cfg = self.best_config or {}
        for key, value in best_cfg.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 70)

        return {
            "best_config": self.best_config,
            "best_score": self.best_score,
            "training_history": self.training_history,
        }

    def _get_default_param_grid(self) -> Dict:
        """Get default hyperparameter search space."""
        return {
            "grid_size": [15, 20, 25],
            "rotation_angles": [
                [0, 90, 180, 270],  # Orthogonal
                [0, 45, 90, 135, 180, 225, 270, 315],  # 8-way
                list(range(0, 360, 30)),  # 12-way
            ],
            "allow_flipping": [True, False],
            "max_attempts": [300, 500],
        }

    def _apply_config(self, config: Dict):
        """Apply configuration to fitting module."""
        fitting_module = self.system.fitting_module

        if "grid_size" in config:
            fitting_module.grid_size = config["grid_size"]
        if "rotation_angles" in config:
            fitting_module.rotation_angles = config["rotation_angles"]
        if "allow_flipping" in config:
            fitting_module.allow_flipping = config["allow_flipping"]
        if "max_attempts" in config:
            fitting_module.max_attempts = config["max_attempts"]

    def _evaluate(self, samples: List[Dict]) -> Dict:
        """
        Evaluate performance on samples.

        Returns:
            Dictionary with utilization, success_rate, waste, avg_time
        """
        metrics = {
            "utilization": [],
            "success_rate": [],
            "waste": [],
            "times": [],
            "patterns_placed": [],
            "patterns_total": [],
        }

        for sample in samples:
            start = time.time()

            # Process patterns
            patterns = [
                self.system.pattern_module.process_image(p) for p in sample["patterns"]
            ]
            patterns = [p for p in patterns if p is not None]

            if not patterns:
                continue

            # Process cloth
            cloth = self.system.cloth_module.process_image(sample["cloth"])

            # Fit patterns
            result = self.system.fitting_module.fit_patterns(patterns, cloth)

            elapsed = time.time() - start

            # Collect metrics
            metrics["utilization"].append(result["utilization_percentage"])
            metrics["success_rate"].append(result["success_rate"])
            metrics["waste"].append(result["waste_area"])
            metrics["times"].append(elapsed)
            metrics["patterns_placed"].append(result["patterns_placed"])
            metrics["patterns_total"].append(result["patterns_total"])

        # Calculate averages
        return {
            "utilization": np.mean(metrics["utilization"])
            if metrics["utilization"]
            else 0,
            "success_rate": np.mean(metrics["success_rate"])
            if metrics["success_rate"]
            else 0,
            "waste": np.mean(metrics["waste"]) if metrics["waste"] else 0,
            "avg_time": np.mean(metrics["times"]) if metrics["times"] else 0,
            "avg_placed": np.mean(metrics["patterns_placed"])
            if metrics["patterns_placed"]
            else 0,
            "avg_total": np.mean(metrics["patterns_total"])
            if metrics["patterns_total"]
            else 0,
            "num_samples": len(metrics["utilization"]),
        }

    def _format_config(self, config: Dict) -> str:
        """Format configuration for logging."""
        formatted = []
        for key, value in config.items():
            if key == "rotation_angles":
                formatted.append(f"{key}={len(value)}_angles")
            else:
                formatted.append(f"{key}={value}")
        return ", ".join(formatted)

    def save_results(self, output_dir: str):
        """Save comprehensive optimization results for research paper."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create subdirectories
        charts_dir = os.path.join(output_dir, "training_charts")
        logs_dir = os.path.join(output_dir, "training_logs")
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Save best config (JSON)
        config_path = os.path.join(output_dir, "best_config.json")
        with open(config_path, "w") as f:
            json.dump(self.best_config, f, indent=2)
        logger.info(f"Best config saved to {config_path}")

        # Save full training history (JSON)
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

        # Save training metrics as CSV (for Excel/LaTeX tables)
        csv_path = os.path.join(logs_dir, f"training_metrics_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Config_ID",
                    "Grid_Size",
                    "Rotation_Angles",
                    "Allow_Flipping",
                    "Max_Attempts",
                    "Train_Utilization",
                    "Train_Success_Rate",
                    "Val_Utilization",
                    "Val_Success_Rate",
                    "Eval_Time_Sec",
                ]
            )
            for result in self.training_history:
                config = result["config"]
                writer.writerow(
                    [
                        result["iteration"],
                        config.get("grid_size", "N/A"),
                        len(config.get("rotation_angles", [])),
                        config.get("allow_flipping", "N/A"),
                        config.get("max_attempts", "N/A"),
                        f"{result['train_metrics']['utilization']:.2f}",
                        f"{result['train_metrics']['success_rate']:.2f}",
                        f"{result['val_metrics']['utilization']:.2f}",
                        f"{result['val_metrics']['success_rate']:.2f}",
                        f"{result['eval_time']:.2f}",
                    ]
                )
        logger.info(f"Training metrics CSV saved to {csv_path}")

        # Generate comparison charts
        self._generate_training_charts(charts_dir)

        # Save detailed text summary
        summary_path = os.path.join(logs_dir, f"training_summary_{timestamp}.txt")
        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("HYPERPARAMETER OPTIMIZATION SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total configurations tested: {len(self.training_history)}\n")
            f.write(f"Best validation score: {self.best_score:.2f}\n\n")
            f.write("Best configuration:\n")
            best_cfg = self.best_config or {}
            for key, value in best_cfg.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            # Top 5 configs
            f.write("Top 5 Configurations (by validation utilization):\n")
            f.write("-" * 70 + "\n")
            sorted_history = sorted(
                self.training_history,
                key=lambda x: x["val_metrics"]["utilization"],
                reverse=True,
            )
            for i, result in enumerate(sorted_history[:5], 1):
                f.write(
                    f"\n{i}. Validation Utilization: {result['val_metrics']['utilization']:.1f}%\n"
                )
                f.write(
                    f"   Validation Success Rate: {result['val_metrics']['success_rate']:.1f}%\n"
                )
                f.write(f"   Config: {self._format_config(result['config'])}\n")
                f.write(f"   Eval Time: {result['eval_time']:.2f}s\n")

        logger.info(f"Training summary saved to {summary_path}")
        logger.info(f"Training charts saved to {charts_dir}/")

    def _generate_training_charts(self, charts_dir: str):
        """Generate visualization charts for training results."""
        if not self.training_history:
            return

        # Extract data
        iterations = [r["iteration"] for r in self.training_history]
        train_util = [r["train_metrics"]["utilization"] for r in self.training_history]
        val_util = [r["val_metrics"]["utilization"] for r in self.training_history]
        train_success = [
            r["train_metrics"]["success_rate"] for r in self.training_history
        ]
        val_success = [r["val_metrics"]["success_rate"] for r in self.training_history]
        eval_times = [r["eval_time"] for r in self.training_history]

        # Chart 1: Utilization Comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(
            iterations, train_util, "o-", label="Training", linewidth=2, markersize=8
        )
        plt.plot(
            iterations, val_util, "s-", label="Validation", linewidth=2, markersize=8
        )
        plt.xlabel("Configuration #", fontsize=12)
        plt.ylabel("Utilization (%)", fontsize=12)
        plt.title(
            "Fabric Utilization Across Configurations", fontsize=14, fontweight="bold"
        )
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Chart 2: Success Rate Comparison
        plt.subplot(1, 2, 2)
        plt.plot(
            iterations, train_success, "o-", label="Training", linewidth=2, markersize=8
        )
        plt.plot(
            iterations, val_success, "s-", label="Validation", linewidth=2, markersize=8
        )
        plt.xlabel("Configuration #", fontsize=12)
        plt.ylabel("Success Rate (%)", fontsize=12)
        plt.title("Pattern Placement Success Rate", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(charts_dir, "training_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Chart 3: Bar chart of top configurations
        sorted_history = sorted(
            self.training_history,
            key=lambda x: x["val_metrics"]["utilization"],
            reverse=True,
        )[:5]

        plt.figure(figsize=(12, 6))
        config_labels = [f"Config {r['iteration']}" for r in sorted_history]
        util_values = [r["val_metrics"]["utilization"] for r in sorted_history]
        success_values = [r["val_metrics"]["success_rate"] for r in sorted_history]

        x = np.arange(len(config_labels))
        width = 0.35

        plt.bar(x - width / 2, util_values, width, label="Utilization %", alpha=0.8)
        plt.bar(x + width / 2, success_values, width, label="Success Rate %", alpha=0.8)

        plt.xlabel("Configuration", fontsize=12)
        plt.ylabel("Performance (%)", fontsize=12)
        plt.title("Top 5 Configurations Performance", fontsize=14, fontweight="bold")
        plt.xticks(x, config_labels)
        plt.legend(fontsize=11)
        plt.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(charts_dir, "top_configs_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Chart 4: Evaluation time analysis
        plt.figure(figsize=(10, 5))
        plt.bar(iterations, eval_times, alpha=0.7, color="steelblue")
        plt.xlabel("Configuration #", fontsize=12)
        plt.ylabel("Evaluation Time (seconds)", fontsize=12)
        plt.title(
            "Computational Efficiency per Configuration", fontsize=14, fontweight="bold"
        )
        plt.grid(True, axis="y", alpha=0.3)
        mean_time = float(np.mean(eval_times))
        plt.axhline(
            y=mean_time,
            color="r",
            linestyle="--",
            label=f"Mean: {mean_time:.2f}s",
        )
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(
            os.path.join(charts_dir, "eval_time_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def load_best_config(config_path: str) -> Dict:
        """Load best configuration from file."""
        with open(config_path, "r") as f:
            return json.load(f)
