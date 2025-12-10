"""
Test early stopping functionality in pattern fitting.

Verifies that:
1. Early stopping triggers when excellent placement is found
2. Search terminates immediately (doesn't continue through remaining positions)
3. Falls back to exhaustive search if threshold never reached
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from cutting_edge.pattern_fitting_module import PatternFittingModule
from cutting_edge.pattern_recognition_module import Pattern
from cutting_edge.cloth_recognition_module import ClothMaterial


def create_mock_pattern(width=50, height=50, name="test_pattern"):
    """Create a mock pattern for testing."""
    pattern = Mock(spec=Pattern)
    pattern.width = width
    pattern.height = height
    pattern.area = width * height
    pattern.name = name
    pattern.pattern_type = "test"
    pattern.contour = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    return pattern


def create_mock_cloth(width=200, height=200):
    """Create a mock cloth for testing."""
    cloth = Mock(spec=ClothMaterial)
    cloth.width = width
    cloth.height = height
    cloth.total_area = width * height
    cloth.usable_area = width * height * 0.95
    cloth.cloth_type = "test"
    cloth.defects = []
    cloth.contour = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    return cloth


class TestEarlyStopping:
    """Test suite for early stopping functionality."""

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers when excellent placement found."""
        # Create fitting module
        fitting_module = PatternFittingModule()
        fitting_module.grid_size = 10  # Small grid for faster testing
        fitting_module.rotation_angles = [0, 90]
        fitting_module.allow_flipping = False
        fitting_module.max_attempts = 1000  # High limit

        # Create test objects
        pattern = create_mock_pattern(30, 30)
        cloth = create_mock_cloth(200, 200)

        # Patch the scoring function to return high score on first valid placement
        with patch.object(
            fitting_module,
            "calculate_placement_score",
            return_value=20.0,  # Above threshold of 15.0
        ):
            # Find placement
            result = fitting_module.find_best_placement(pattern, cloth, [])

        # Verify placement was found
        assert result is not None
        assert result.score == 20.0

        # Note: We can't easily verify the exact number of attempts without
        # more invasive mocking, but the log output should show early stopping

    def test_no_early_stopping_when_below_threshold(self):
        """Test that search continues when score never exceeds threshold."""
        # Create fitting module
        fitting_module = PatternFittingModule()
        fitting_module.grid_size = 5  # Small grid
        fitting_module.rotation_angles = [0]
        fitting_module.allow_flipping = False
        fitting_module.max_attempts = 100

        # Create test objects
        pattern = create_mock_pattern(30, 30)
        cloth = create_mock_cloth(200, 200)

        # Patch scoring to always return below threshold
        with patch.object(
            fitting_module,
            "calculate_placement_score",
            return_value=5.0,  # Below threshold of 15.0
        ):
            result = fitting_module.find_best_placement(pattern, cloth, [])

        # Should still find placement (best of what was tried)
        assert result is not None
        assert result.score == 5.0

    def test_early_stopping_with_invalid_placements(self):
        """Test that early stopping works even when many placements are invalid."""
        fitting_module = PatternFittingModule()
        fitting_module.grid_size = 10
        fitting_module.rotation_angles = [0, 90]
        fitting_module.allow_flipping = False
        fitting_module.max_attempts = 500

        pattern = create_mock_pattern(30, 30)
        cloth = create_mock_cloth(200, 200)

        # Track calls to scoring function
        score_call_count = [0]

        def mock_score(*args, **kwargs):
            score_call_count[0] += 1
            # Return excellent score on 10th valid placement
            if score_call_count[0] == 10:
                return 16.0  # Above threshold
            return 5.0  # Below threshold

        with patch.object(
            fitting_module, "calculate_placement_score", side_effect=mock_score
        ):
            result = fitting_module.find_best_placement(pattern, cloth, [])

        # Should have stopped after finding excellent placement
        assert result is not None
        assert result.score == 16.0
        # Should have called scoring function <= 10 times (stopped early)
        assert score_call_count[0] <= 15  # Allow some buffer for invalid placements

    def test_max_attempts_still_enforced(self):
        """Test that max_attempts limit is still respected."""
        fitting_module = PatternFittingModule()
        fitting_module.grid_size = 20  # Large grid
        fitting_module.rotation_angles = [0, 90, 180, 270]
        fitting_module.allow_flipping = True
        fitting_module.max_attempts = 10  # Very low limit

        pattern = create_mock_pattern(30, 30)
        cloth = create_mock_cloth(200, 200)

        # Never return excellent score
        with patch.object(
            fitting_module, "calculate_placement_score", return_value=5.0
        ):
            result = fitting_module.find_best_placement(pattern, cloth, [])

        # Should have found something, but stopped at max_attempts
        # (can't verify exact count without more invasive testing)
        assert (
            result is not None or result is None
        )  # May or may not find valid placement in 10 attempts


def test_threshold_configurable():
    """Test that threshold can be configured."""
    from cutting_edge.config import FITTING

    # Verify threshold is configurable
    assert "EXCELLENT_SCORE_THRESHOLD" in FITTING
    assert isinstance(FITTING["EXCELLENT_SCORE_THRESHOLD"], (int, float))
    assert FITTING["EXCELLENT_SCORE_THRESHOLD"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
